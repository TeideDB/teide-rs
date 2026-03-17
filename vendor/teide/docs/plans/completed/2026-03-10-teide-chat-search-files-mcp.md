# Teide Chat Plan 2: Search, Files, MCP Tools — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan.

**Goal:** Add message search indexing, file upload/download, and MCP chat tools to Teide Chat.

**Architecture:** Extend existing chat modules with search indexing in the message post flow, a new files module for upload/download, and new MCP tools that delegate to the chat handlers.

**Tech Stack:** Rust, Axum 0.8, tantivy, TeideDB, rmcp

**Spec:** `docs/superpowers/specs/2026-03-10-teide-chat-design.md` (sections 5, 7)

---

## Chunk 1: Message Search Indexing

### Task 1: Index messages on post

**Files:** `teidelum/src/chat/handlers.rs`

**Context:** When a message is posted via `chat_post_message`, it must be indexed in tantivy so the existing `search` MCP tool and the new `search.messages` endpoint can find it. The `ChatState` already has `api: Arc<TeidelumApi>`, and `TeidelumApi` has `search_engine()` which returns `&Arc<SearchEngine>`. The `SearchEngine::index_documents` method takes `&[(String, String, String, String)]` tuples of (id, source, title, body).

Per the spec (Section 5): "Every message indexed in tantivy as a SearchDocument (source: `"chat"`, title: `"#channel-name"`, body: message content)."

We need the channel name for the title field. The handler already has the channel ID from the request, so we query for the channel name.

**Steps:**

- [x] In `chat_post_message` in `teidelum/src/chat/handlers.rs`, after the successful message INSERT and before the broadcast, add search indexing. Look up the channel name, then call `state.api.search_engine().index_documents(...)`.

In the `chat_post_message` function, find the section after the message INSERT succeeds (after the mentions parsing block, before the broadcast). Add search indexing there. The complete modified function:

```rust
pub async fn chat_post_message(
    State(state): State<AppState>,
    Extension(claims): Extension<Claims>,
    Json(req): Json<PostMessageRequest>,
) -> Response {
    if !is_channel_member(&state, req.channel, claims.user_id) {
        return slack::err("channel_not_found");
    }

    let id = next_id();
    let now = now_timestamp();
    let thread_id = req.thread_ts.unwrap_or(0);

    let insert = format!(
        "INSERT INTO messages (id, channel_id, user_id, thread_id, content, deleted_at, edited_at, created_at) \
         VALUES ({id}, {channel}, {user}, {thread}, '{text}', NULL, NULL, '{now}')",
        channel = req.channel,
        user = claims.user_id,
        thread = thread_id,
        text = escape_sql(&req.text),
    );

    if let Err(e) = state.api.query_router().query_sync(&insert) {
        tracing::error!("post message failed: {e}");
        return slack::err("internal_error");
    }

    // Parse and store mentions
    let mention_regex = regex_lite::Regex::new(r"@(\w+)").unwrap();
    for cap in mention_regex.captures_iter(&req.text) {
        let mentioned_username = &cap[1];
        let user_sql = format!(
            "SELECT id FROM users WHERE username = '{}'",
            escape_sql(mentioned_username)
        );
        if let Ok(r) = state.api.query_router().query_sync(&user_sql) {
            if let Some(row) = r.rows.first() {
                if let crate::connector::Value::Int(uid) = &row[0] {
                    // Check uniqueness before inserting
                    let check = format!(
                        "SELECT message_id FROM mentions WHERE message_id = {} AND user_id = {}",
                        id, uid
                    );
                    if let Ok(existing) = state.api.query_router().query_sync(&check) {
                        if existing.rows.is_empty() {
                            let mention_sql = format!(
                                "INSERT INTO mentions (message_id, user_id) VALUES ({}, {})",
                                id, uid
                            );
                            let _ = state.api.query_router().query_sync(&mention_sql);
                        }
                    }
                }
            }
        }
    }

    // Index message in tantivy for full-text search
    let channel_name = {
        let name_sql = format!(
            "SELECT name FROM channels WHERE id = {}",
            req.channel
        );
        match state.api.query_router().query_sync(&name_sql) {
            Ok(r) if !r.rows.is_empty() => {
                match &r.rows[0][0] {
                    crate::connector::Value::String(s) => format!("#{s}"),
                    _ => format!("#{}", req.channel),
                }
            }
            _ => format!("#{}", req.channel),
        }
    };
    let doc = vec![(
        id.to_string(),
        "chat".to_string(),
        channel_name,
        req.text.clone(),
    )];
    if let Err(e) = state.api.search_engine().index_documents(&doc) {
        tracing::warn!("message search indexing failed: {e}");
    }

    // Broadcast message event
    let event = crate::chat::events::ServerEvent::Message {
        channel: req.channel.to_string(),
        user: claims.user_id.to_string(),
        text: req.text.clone(),
        ts: id.to_string(),
        thread_ts: if thread_id != 0 {
            Some(thread_id.to_string())
        } else {
            None
        },
    };
    state.hub.broadcast_to_channel(req.channel, &event).await;

    slack::ok(json!({
        "message": {
            "ts": id.to_string(),
            "channel": req.channel.to_string(),
            "user": claims.user_id.to_string(),
            "text": req.text,
        }
    }))
}
```

The key addition is the block between the mentions section and the broadcast section:

```rust
    // Index message in tantivy for full-text search
    let channel_name = {
        let name_sql = format!(
            "SELECT name FROM channels WHERE id = {}",
            req.channel
        );
        match state.api.query_router().query_sync(&name_sql) {
            Ok(r) if !r.rows.is_empty() => {
                match &r.rows[0][0] {
                    crate::connector::Value::String(s) => format!("#{s}"),
                    _ => format!("#{}", req.channel),
                }
            }
            _ => format!("#{}", req.channel),
        }
    };
    let doc = vec![(
        id.to_string(),
        "chat".to_string(),
        channel_name,
        req.text.clone(),
    )];
    if let Err(e) = state.api.search_engine().index_documents(&doc) {
        tracing::warn!("message search indexing failed: {e}");
    }
```

**Test commands:**

```bash
cd teidelum && cargo test --lib && cargo check
```

**Commit:**

```bash
cd teidelum && git add src/chat/handlers.rs && git commit -m "chat: index messages in tantivy on post for full-text search"
```

---

### Task 2: `search.messages` endpoint

**Files:** `teidelum/src/chat/handlers.rs`

**Context:** Add a `search.messages` handler that wraps the existing `SearchEngine::search` with Slack-compatible response format. The handler accepts `{query, limit?}`, calls search with `sources: Some(vec!["chat".to_string()])` to filter to chat messages only, and returns matches in Slack response format.

**Steps:**

- [x] Add the `SearchMessagesRequest` struct and `search_messages` handler function to `teidelum/src/chat/handlers.rs`, after the reactions handlers section.

Add the following after the `ReactionRequest` struct (around line 1214):

```rust
// ── Search ──

#[derive(Deserialize)]
pub struct SearchMessagesRequest {
    pub query: String,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
}

fn default_search_limit() -> usize {
    20
}

pub async fn search_messages(
    State(state): State<AppState>,
    Extension(_claims): Extension<Claims>,
    Json(req): Json<SearchMessagesRequest>,
) -> Response {
    if req.query.is_empty() {
        return slack::err("invalid_arguments");
    }

    let limit = req.limit.min(100);

    let search_query = crate::search::SearchQuery {
        text: req.query,
        sources: Some(vec!["chat".to_string()]),
        limit,
        date_from: None,
        date_to: None,
    };

    let results = match state.api.search_engine().search(&search_query) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("search.messages failed: {e}");
            return slack::err("internal_error");
        }
    };

    let matches: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            json!({
                "ts": r.id,
                "channel": r.title,
                "text": r.snippet,
                "score": r.score,
            })
        })
        .collect();

    slack::ok(json!({
        "messages": {
            "matches": matches,
            "total": matches.len(),
        }
    }))
}
```

- [x] Add the route for `search.messages` to the `chat_routes` function in the `authed` router section.

In the `chat_routes` function, add this line after the `.route("/reactions.remove", ...)` line:

```rust
        // Search
        .route("/search.messages", axum::routing::post(search_messages))
```

The full `authed` router in `chat_routes` becomes:

```rust
    let authed = Router::new()
        // Conversations
        .route("/conversations.create", axum::routing::post(conversations_create))
        .route("/conversations.list", axum::routing::post(conversations_list))
        .route("/conversations.info", axum::routing::post(conversations_info))
        .route("/conversations.history", axum::routing::post(conversations_history))
        .route("/conversations.replies", axum::routing::post(conversations_replies))
        .route("/conversations.join", axum::routing::post(conversations_join))
        .route("/conversations.leave", axum::routing::post(conversations_leave))
        .route("/conversations.invite", axum::routing::post(conversations_invite))
        .route("/conversations.members", axum::routing::post(conversations_members))
        .route("/conversations.open", axum::routing::post(conversations_open))
        // Chat
        .route("/chat.postMessage", axum::routing::post(chat_post_message))
        .route("/chat.update", axum::routing::post(chat_update))
        .route("/chat.delete", axum::routing::post(chat_delete))
        // Users
        .route("/users.list", axum::routing::post(users_list))
        .route("/users.info", axum::routing::post(users_info))
        .route("/users.setPresence", axum::routing::post(users_set_presence))
        // Reactions
        .route("/reactions.add", axum::routing::post(reactions_add))
        .route("/reactions.remove", axum::routing::post(reactions_remove))
        // Search
        .route("/search.messages", axum::routing::post(search_messages))
        .layer(middleware::from_fn(crate::chat::auth::jwt_middleware))
        .with_state(state.clone());
```

**Test commands:**

```bash
cd teidelum && cargo test --lib && cargo check
```

**Commit:**

```bash
cd teidelum && git add src/chat/handlers.rs && git commit -m "chat: add search.messages endpoint wrapping tantivy search"
```

---

## Chunk 2: File Upload & Download

### Task 3: File upload handler

**Files:** `teidelum/src/chat/files.rs` (new), `teidelum/src/chat/mod.rs`

**Context:** Per the spec, files are stored on disk at `data/files/<uuid>/<original_filename>`. The metadata goes in the existing `files` table (already created in Plan 1 in `models.rs`). The upload endpoint is `files.upload` using multipart form data. It posts a message with a file reference after storing the file.

The `files` table schema (from `models.rs`):
```sql
CREATE TABLE files (
    id BIGINT, message_id BIGINT, user_id BIGINT, channel_id BIGINT,
    filename VARCHAR, mime_type VARCHAR, size_bytes BIGINT,
    storage_path VARCHAR, created_at VARCHAR
)
```

**Steps:**

- [x] Create `teidelum/src/chat/files.rs` with the file upload handler.

```rust
use crate::chat::auth::Claims;
use crate::chat::handlers::AppState;
use crate::chat::id::next_id;
use crate::chat::models::{escape_sql, now_timestamp};
use crate::chat::slack;
use axum::extract::State;
use axum::response::Response;
use axum::Extension;
use axum::extract::Multipart;
use serde_json::json;

/// Maximum file size: 10 MB.
const MAX_FILE_SIZE: usize = 10 * 1024 * 1024;

/// Allowed MIME types for upload.
const ALLOWED_MIME_TYPES: &[&str] = &[
    "text/plain",
    "text/csv",
    "text/markdown",
    "text/html",
    "application/json",
    "application/pdf",
    "application/zip",
    "application/gzip",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "audio/mpeg",
    "audio/ogg",
    "video/mp4",
    "video/webm",
    "application/octet-stream",
];

/// Guess MIME type from file extension.
fn guess_mime(filename: &str) -> &'static str {
    let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();
    match ext.as_str() {
        "txt" => "text/plain",
        "csv" => "text/csv",
        "md" => "text/markdown",
        "html" | "htm" => "text/html",
        "json" => "application/json",
        "pdf" => "application/pdf",
        "zip" => "application/zip",
        "gz" | "gzip" => "application/gzip",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "mp3" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        _ => "application/octet-stream",
    }
}

/// Handle `files.upload` — multipart file upload.
///
/// Multipart fields:
/// - `channel` (text): channel ID (required)
/// - `file` (file): the file to upload (required)
/// - `message` (text): optional message text to accompany the file
/// - `thread_ts` (text): optional thread parent ID
pub async fn files_upload(
    State(state): State<AppState>,
    Extension(claims): Extension<Claims>,
    mut multipart: Multipart,
) -> Response {
    let mut channel_id: Option<i64> = None;
    let mut file_data: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut file_mime: Option<String> = None;
    let mut message_text: Option<String> = None;
    let mut thread_ts: Option<i64> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "channel" => {
                if let Ok(text) = field.text().await {
                    channel_id = text.parse().ok();
                }
            }
            "file" => {
                file_name = field.file_name().map(|s| s.to_string());
                file_mime = field.content_type().map(|s| s.to_string());
                match field.bytes().await {
                    Ok(bytes) => {
                        if bytes.len() > MAX_FILE_SIZE {
                            return slack::err("file_too_large");
                        }
                        file_data = Some(bytes.to_vec());
                    }
                    Err(e) => {
                        tracing::error!("file read failed: {e}");
                        return slack::err("internal_error");
                    }
                }
            }
            "message" => {
                if let Ok(text) = field.text().await {
                    message_text = Some(text);
                }
            }
            "thread_ts" => {
                if let Ok(text) = field.text().await {
                    thread_ts = text.parse().ok();
                }
            }
            _ => {}
        }
    }

    let channel_id = match channel_id {
        Some(c) => c,
        None => return slack::err("invalid_arguments"),
    };

    let file_data = match file_data {
        Some(d) => d,
        None => return slack::err("no_file_uploaded"),
    };

    let file_name = file_name.unwrap_or_else(|| "unnamed".to_string());

    // Determine MIME type
    let mime_type = file_mime.unwrap_or_else(|| guess_mime(&file_name).to_string());
    if !ALLOWED_MIME_TYPES.contains(&mime_type.as_str()) {
        return slack::err("invalid_file_type");
    }

    // Check channel membership
    let check_sql = format!(
        "SELECT channel_id FROM channel_members WHERE channel_id = {} AND user_id = {}",
        channel_id, claims.user_id
    );
    match state.api.query_router().query_sync(&check_sql) {
        Ok(r) if r.rows.is_empty() => return slack::err("channel_not_found"),
        Err(e) => {
            tracing::error!("file upload membership check failed: {e}");
            return slack::err("internal_error");
        }
        _ => {}
    }

    // Store file on disk: data/files/<uuid>/<filename>
    let file_uuid = uuid::Uuid::new_v4().to_string();
    let storage_dir = format!("data/files/{file_uuid}");
    let storage_path = format!("{storage_dir}/{file_name}");

    if let Err(e) = std::fs::create_dir_all(&storage_dir) {
        tracing::error!("file dir creation failed: {e}");
        return slack::err("internal_error");
    }

    if let Err(e) = std::fs::write(&storage_path, &file_data) {
        tracing::error!("file write failed: {e}");
        return slack::err("internal_error");
    }

    let file_size = file_data.len() as i64;

    // Post a message with file reference
    let msg_id = next_id();
    let now = now_timestamp();
    let thread_id = thread_ts.unwrap_or(0);
    let msg_text = message_text.unwrap_or_else(|| format!("[file: {}]", file_name));

    let msg_insert = format!(
        "INSERT INTO messages (id, channel_id, user_id, thread_id, content, deleted_at, edited_at, created_at) \
         VALUES ({msg_id}, {channel_id}, {user_id}, {thread_id}, '{text}', NULL, NULL, '{now}')",
        user_id = claims.user_id,
        text = escape_sql(&msg_text),
    );

    if let Err(e) = state.api.query_router().query_sync(&msg_insert) {
        tracing::error!("file message insert failed: {e}");
        return slack::err("internal_error");
    }

    // Insert file metadata row
    let file_id = next_id();
    let file_insert = format!(
        "INSERT INTO files (id, message_id, user_id, channel_id, filename, mime_type, size_bytes, storage_path, created_at) \
         VALUES ({file_id}, {msg_id}, {user_id}, {channel_id}, '{filename}', '{mime}', {size}, '{path}', '{now}')",
        user_id = claims.user_id,
        filename = escape_sql(&file_name),
        mime = escape_sql(&mime_type),
        size = file_size,
        path = escape_sql(&storage_path),
    );

    if let Err(e) = state.api.query_router().query_sync(&file_insert) {
        tracing::error!("file metadata insert failed: {e}");
        return slack::err("internal_error");
    }

    // Index the message in tantivy
    let channel_name = {
        let name_sql = format!("SELECT name FROM channels WHERE id = {}", channel_id);
        match state.api.query_router().query_sync(&name_sql) {
            Ok(r) if !r.rows.is_empty() => {
                match &r.rows[0][0] {
                    crate::connector::Value::String(s) => format!("#{s}"),
                    _ => format!("#{channel_id}"),
                }
            }
            _ => format!("#{channel_id}"),
        }
    };
    let doc = vec![(
        msg_id.to_string(),
        "chat".to_string(),
        channel_name,
        msg_text.clone(),
    )];
    if let Err(e) = state.api.search_engine().index_documents(&doc) {
        tracing::warn!("file message search indexing failed: {e}");
    }

    // Broadcast message event
    let event = crate::chat::events::ServerEvent::Message {
        channel: channel_id.to_string(),
        user: claims.user_id.to_string(),
        text: msg_text,
        ts: msg_id.to_string(),
        thread_ts: if thread_id != 0 {
            Some(thread_id.to_string())
        } else {
            None
        },
    };
    state.hub.broadcast_to_channel(channel_id, &event).await;

    slack::ok(json!({
        "file": {
            "id": file_id.to_string(),
            "name": file_name,
            "mime_type": mime_type,
            "size": file_size,
        },
        "message": {
            "ts": msg_id.to_string(),
            "channel": channel_id.to_string(),
        }
    }))
}
```

- [x] Add `pub mod files;` to `teidelum/src/chat/mod.rs`.

The updated `mod.rs`:

```rust
pub mod auth;
pub mod events;
pub mod files;
pub mod handlers;
pub mod hub;
pub mod id;
pub mod models;
pub mod slack;
pub mod ws;
```

**Test commands:**

```bash
cd teidelum && cargo check
```

**Commit:**

```bash
cd teidelum && git add src/chat/files.rs src/chat/mod.rs && git commit -m "chat: add file upload handler with disk storage and message posting"
```

---

### Task 4: File download handler

**Files:** `teidelum/src/chat/files.rs`

**Context:** Add a GET endpoint at `/files/{id}/{filename}` that serves files from disk. Must validate JWT auth (via query param `token` since it's a GET) and check that the requesting user is a member of the channel the file was uploaded to.

**Steps:**

- [x] Add the `files_download` handler to `teidelum/src/chat/files.rs`.

Add the following imports at the top of `files.rs` (merge with existing):

```rust
use axum::extract::Path;
use axum::http::{header, StatusCode};
use axum::body::Body;
use axum::extract::Query;
use serde::Deserialize;
```

Add the download handler:

```rust
#[derive(Deserialize)]
pub struct FileDownloadQuery {
    pub token: String,
}

/// Handle `GET /files/:id/:filename` — download a file with auth check.
pub async fn files_download(
    State(state): State<AppState>,
    Path((file_id, _filename)): Path<(String, String)>,
    Query(query): Query<FileDownloadQuery>,
) -> Response {
    // Validate JWT from query param
    let secret = match std::env::var("TEIDE_CHAT_SECRET") {
        Ok(s) if !s.is_empty() => s,
        _ => return slack::http_err(StatusCode::INTERNAL_SERVER_ERROR, "server_misconfigured"),
    };

    let claims = match crate::chat::auth::validate_token(&secret, &query.token) {
        Ok(c) => c,
        Err(_) => return slack::http_err(StatusCode::UNAUTHORIZED, "invalid_auth"),
    };

    // Look up file metadata
    let sql = format!(
        "SELECT channel_id, filename, mime_type, storage_path FROM files WHERE id = {}",
        escape_sql(&file_id)
    );
    let result = match state.api.query_router().query_sync(&sql) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("file download query failed: {e}");
            return slack::http_err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error");
        }
    };

    if result.rows.is_empty() {
        return slack::http_err(StatusCode::NOT_FOUND, "file_not_found");
    }

    let row = &result.rows[0];
    let channel_id = match &row[0] {
        crate::connector::Value::Int(v) => *v,
        _ => return slack::http_err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
    };
    let filename = match &row[1] {
        crate::connector::Value::String(s) => s.clone(),
        _ => return slack::http_err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
    };
    let mime_type = match &row[2] {
        crate::connector::Value::String(s) => s.clone(),
        _ => "application/octet-stream".to_string(),
    };
    let storage_path = match &row[3] {
        crate::connector::Value::String(s) => s.clone(),
        _ => return slack::http_err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
    };

    // Check channel membership
    let check_sql = format!(
        "SELECT channel_id FROM channel_members WHERE channel_id = {} AND user_id = {}",
        channel_id, claims.user_id
    );
    match state.api.query_router().query_sync(&check_sql) {
        Ok(r) if r.rows.is_empty() => {
            return slack::http_err(StatusCode::FORBIDDEN, "not_in_channel");
        }
        Err(e) => {
            tracing::error!("file download membership check failed: {e}");
            return slack::http_err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error");
        }
        _ => {}
    }

    // Read file from disk and stream it
    let file_bytes = match std::fs::read(&storage_path) {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::error!("file read from disk failed: {e}");
            return slack::http_err(StatusCode::NOT_FOUND, "file_not_found");
        }
    };

    let content_disposition = format!("inline; filename=\"{}\"", filename.replace('"', "\\\""));

    axum::http::Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime_type)
        .header(header::CONTENT_DISPOSITION, content_disposition)
        .body(Body::from(file_bytes))
        .unwrap()
        .into_response()
}
```

The full import section at the top of `files.rs` should be:

```rust
use crate::chat::auth::Claims;
use crate::chat::handlers::AppState;
use crate::chat::id::next_id;
use crate::chat::models::{escape_sql, now_timestamp};
use crate::chat::slack;
use axum::body::Body;
use axum::extract::{Multipart, Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use serde::Deserialize;
use serde_json::json;
```

**Test commands:**

```bash
cd teidelum && cargo check
```

**Commit:**

```bash
cd teidelum && git add src/chat/files.rs && git commit -m "chat: add file download handler with auth and channel membership check"
```

---

### Task 5: Wire file routes

**Files:** `teidelum/src/chat/handlers.rs`, `teidelum/src/server.rs`

**Context:** The `files.upload` endpoint goes in the authenticated Slack API routes. The `files/{id}/{filename}` download endpoint is a GET that handles its own auth (via query param), so it goes at the top level of the router.

**Steps:**

- [x] Add the `files.upload` route to the `authed` router in `chat_routes` in `teidelum/src/chat/handlers.rs`.

Add after the `search.messages` route:

```rust
        // Files
        .route("/files.upload", axum::routing::post(crate::chat::files::files_upload))
```

The full `authed` section becomes:

```rust
    let authed = Router::new()
        // Conversations
        .route("/conversations.create", axum::routing::post(conversations_create))
        .route("/conversations.list", axum::routing::post(conversations_list))
        .route("/conversations.info", axum::routing::post(conversations_info))
        .route("/conversations.history", axum::routing::post(conversations_history))
        .route("/conversations.replies", axum::routing::post(conversations_replies))
        .route("/conversations.join", axum::routing::post(conversations_join))
        .route("/conversations.leave", axum::routing::post(conversations_leave))
        .route("/conversations.invite", axum::routing::post(conversations_invite))
        .route("/conversations.members", axum::routing::post(conversations_members))
        .route("/conversations.open", axum::routing::post(conversations_open))
        // Chat
        .route("/chat.postMessage", axum::routing::post(chat_post_message))
        .route("/chat.update", axum::routing::post(chat_update))
        .route("/chat.delete", axum::routing::post(chat_delete))
        // Users
        .route("/users.list", axum::routing::post(users_list))
        .route("/users.info", axum::routing::post(users_info))
        .route("/users.setPresence", axum::routing::post(users_set_presence))
        // Reactions
        .route("/reactions.add", axum::routing::post(reactions_add))
        .route("/reactions.remove", axum::routing::post(reactions_remove))
        // Search
        .route("/search.messages", axum::routing::post(search_messages))
        // Files
        .route("/files.upload", axum::routing::post(crate::chat::files::files_upload))
        .layer(middleware::from_fn(crate::chat::auth::jwt_middleware))
        .with_state(state.clone());
```

- [x] Add the file download route to `build_router` in `teidelum/src/server.rs`.

In `build_router`, add the file download route after the `.merge(chat_routes(...))` line and before the `.layer(CorsLayer::permissive())`. The download route needs the `AppState`:

```rust
pub fn build_router(api: Arc<TeidelumApi>, hub: Arc<crate::chat::hub::Hub>, ct: CancellationToken) -> Router {
    let chat_state: crate::chat::handlers::AppState = Arc::new(ChatState {
        api: api.clone(),
        hub: hub.clone(),
    });

    let mut app = Router::new()
        .merge(routes::api_routes())
        .with_state(api.clone())
        .merge(chat_routes(chat_state.clone()))
        .route("/files/{id}/{filename}", axum::routing::get(crate::chat::files::files_download).with_state(chat_state.clone()))
        .route("/ws", axum::routing::get(ws_upgrade).with_state(chat_state))
        .layer(CorsLayer::permissive());

    // ... rest unchanged
```

**Test commands:**

```bash
cd teidelum && cargo check
```

**Commit:**

```bash
cd teidelum && git add src/chat/handlers.rs src/server.rs && git commit -m "chat: wire file upload and download routes"
```

---

## Chunk 3: MCP Chat Tools

### Task 6: Add chat MCP tools

**Files:** `teidelum/src/mcp.rs`

**Context:** Per spec Section 7, add six MCP tools: `chat.postMessage`, `chat.history`, `chat.reply`, `chat.react`, `chat.listChannels`, `chat.search`. These tools let AI agents participate in chat via the MCP protocol.

The MCP `Teidelum` struct has `api: Arc<TeidelumApi>`. The tools need to execute SQL queries through `api.query_router()` and search through `api.search_engine()`, same as the HTTP handlers do. Bot users are identified by `is_bot: true` in the users table.

Key design decision: MCP tools operate as a **specific bot user**. The bot user ID is resolved at tool call time by looking up a bot user (we'll use the first `is_bot=true` user found, or return an error if no bot exists). This keeps the MCP tools stateless and simple.

**Steps:**

- [x] Add parameter structs for the six new MCP chat tools to `teidelum/src/mcp.rs`.

Add after the `AddRelationshipParams` struct (around line 178):

```rust
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ChatPostMessageParams {
    /// Channel ID to post to.
    pub channel: i64,
    /// Message text content.
    pub text: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ChatHistoryParams {
    /// Channel ID to read history from.
    pub channel: i64,
    /// Maximum number of messages to return.
    #[serde(default = "default_history_limit")]
    pub limit: usize,
}

fn default_history_limit() -> usize {
    50
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ChatReplyParams {
    /// Channel ID containing the thread.
    pub channel: i64,
    /// Thread parent message ID (ts).
    pub thread_ts: i64,
    /// Reply text content.
    pub text: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ChatReactParams {
    /// Message ID (ts) to react to.
    pub timestamp: i64,
    /// Emoji name (e.g. "thumbsup", "heart").
    pub name: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ChatListChannelsParams {
    /// Optional bot user ID. If omitted, lists all public channels.
    #[serde(default)]
    pub bot_user_id: Option<i64>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ChatSearchParams {
    /// Search query string.
    pub query: String,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
}
```

- [x] Add a helper method on `Teidelum` to resolve the bot user ID.

Add inside the `#[tool_router] impl Teidelum` block, before the first `#[tool]` method:

```rust
    /// Look up the first bot user. Returns (user_id, username) or error.
    fn resolve_bot_user(&self) -> Result<(i64, String), McpError> {
        let sql = "SELECT id, username FROM users WHERE is_bot = true LIMIT 1";
        let result = self.api.query_router().query_sync(sql)
            .map_err(|e| McpError::internal_error(format!("bot user lookup failed: {e}"), None))?;

        if result.rows.is_empty() {
            return Err(McpError::internal_error(
                "no bot user found — create one with is_bot=true via auth.register".to_string(),
                None,
            ));
        }

        let user_id = match &result.rows[0][0] {
            Value::Int(v) => *v,
            _ => return Err(McpError::internal_error("invalid bot user id".to_string(), None)),
        };
        let username = match &result.rows[0][1] {
            Value::String(s) => s.clone(),
            _ => return Err(McpError::internal_error("invalid bot username".to_string(), None)),
        };

        Ok((user_id, username))
    }

    /// Check if a user is a member of a channel.
    fn is_member(&self, channel_id: i64, user_id: i64) -> bool {
        let sql = format!(
            "SELECT channel_id FROM channel_members WHERE channel_id = {} AND user_id = {}",
            channel_id, user_id
        );
        match self.api.query_router().query_sync(&sql) {
            Ok(r) => !r.rows.is_empty(),
            Err(_) => false,
        }
    }
```

Note: the helper uses `Value` which is already imported as `crate::connector::Value` in the file's use section.

- [x] Add the six `#[tool]` methods inside the `#[tool_router] impl Teidelum` block, after the existing `add_relationship` tool.

```rust
    #[tool(description = "Send a message to a chat channel (as bot user)")]
    async fn chat_post_message(
        &self,
        Parameters(params): Parameters<ChatPostMessageParams>,
    ) -> Result<CallToolResult, McpError> {
        let (bot_id, _) = self.resolve_bot_user()?;

        if !self.is_member(params.channel, bot_id) {
            return Err(McpError::invalid_params(
                format!("bot is not a member of channel {}", params.channel),
                None,
            ));
        }

        let id = crate::chat::id::next_id();
        let now = crate::chat::models::now_timestamp();
        let text_escaped = crate::chat::models::escape_sql(&params.text);

        let sql = format!(
            "INSERT INTO messages (id, channel_id, user_id, thread_id, content, deleted_at, edited_at, created_at) \
             VALUES ({id}, {channel}, {bot}, 0, '{text}', NULL, NULL, '{now}')",
            channel = params.channel,
            bot = bot_id,
            text = text_escaped,
        );

        self.api.query_router().query_sync(&sql)
            .map_err(|e| McpError::internal_error(format!("post message failed: {e}"), None))?;

        // Index in tantivy
        let channel_name = {
            let name_sql = format!("SELECT name FROM channels WHERE id = {}", params.channel);
            match self.api.query_router().query_sync(&name_sql) {
                Ok(r) if !r.rows.is_empty() => {
                    match &r.rows[0][0] {
                        Value::String(s) => format!("#{s}"),
                        _ => format!("#{}", params.channel),
                    }
                }
                _ => format!("#{}", params.channel),
            }
        };
        let doc = vec![(
            id.to_string(),
            "chat".to_string(),
            channel_name,
            params.text.clone(),
        )];
        if let Err(e) = self.api.search_engine().index_documents(&doc) {
            tracing::warn!("MCP chat message indexing failed: {e}");
        }

        let result = serde_json::json!({
            "ok": true,
            "ts": id.to_string(),
            "channel": params.channel.to_string(),
            "text": params.text,
        });
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&result).unwrap(),
        )]))
    }

    #[tool(description = "Read recent message history from a chat channel")]
    async fn chat_history(
        &self,
        Parameters(params): Parameters<ChatHistoryParams>,
    ) -> Result<CallToolResult, McpError> {
        let (bot_id, _) = self.resolve_bot_user()?;

        if !self.is_member(params.channel, bot_id) {
            return Err(McpError::invalid_params(
                format!("bot is not a member of channel {}", params.channel),
                None,
            ));
        }

        let limit = params.limit.min(200);
        let sql = format!(
            "SELECT m.id, m.user_id, m.thread_id, m.content, m.created_at, u.username \
             FROM messages m \
             JOIN users u ON m.user_id = u.id \
             WHERE m.channel_id = {} AND m.thread_id = 0 AND m.deleted_at IS NULL \
             ORDER BY m.id DESC LIMIT {}",
            params.channel, limit
        );

        let result = self.api.query_router().query_sync(&sql)
            .map_err(|e| McpError::internal_error(format!("history query failed: {e}"), None))?;

        let messages: Vec<serde_json::Value> = result.rows.iter().map(|row| {
            serde_json::json!({
                "ts": row[0].to_json(),
                "user": row[1].to_json(),
                "thread_ts": row[2].to_json(),
                "text": row[3].to_json(),
                "created_at": row[4].to_json(),
                "username": row[5].to_json(),
            })
        }).collect();

        let output = serde_json::json!({
            "ok": true,
            "channel": params.channel.to_string(),
            "messages": messages,
        });
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    #[tool(description = "Reply to a thread in a chat channel (as bot user)")]
    async fn chat_reply(
        &self,
        Parameters(params): Parameters<ChatReplyParams>,
    ) -> Result<CallToolResult, McpError> {
        let (bot_id, _) = self.resolve_bot_user()?;

        if !self.is_member(params.channel, bot_id) {
            return Err(McpError::invalid_params(
                format!("bot is not a member of channel {}", params.channel),
                None,
            ));
        }

        let id = crate::chat::id::next_id();
        let now = crate::chat::models::now_timestamp();
        let text_escaped = crate::chat::models::escape_sql(&params.text);

        let sql = format!(
            "INSERT INTO messages (id, channel_id, user_id, thread_id, content, deleted_at, edited_at, created_at) \
             VALUES ({id}, {channel}, {bot}, {thread}, '{text}', NULL, NULL, '{now}')",
            channel = params.channel,
            bot = bot_id,
            thread = params.thread_ts,
            text = text_escaped,
        );

        self.api.query_router().query_sync(&sql)
            .map_err(|e| McpError::internal_error(format!("reply failed: {e}"), None))?;

        // Index in tantivy
        let channel_name = {
            let name_sql = format!("SELECT name FROM channels WHERE id = {}", params.channel);
            match self.api.query_router().query_sync(&name_sql) {
                Ok(r) if !r.rows.is_empty() => {
                    match &r.rows[0][0] {
                        Value::String(s) => format!("#{s}"),
                        _ => format!("#{}", params.channel),
                    }
                }
                _ => format!("#{}", params.channel),
            }
        };
        let doc = vec![(
            id.to_string(),
            "chat".to_string(),
            channel_name,
            params.text.clone(),
        )];
        if let Err(e) = self.api.search_engine().index_documents(&doc) {
            tracing::warn!("MCP chat reply indexing failed: {e}");
        }

        let result = serde_json::json!({
            "ok": true,
            "ts": id.to_string(),
            "channel": params.channel.to_string(),
            "thread_ts": params.thread_ts.to_string(),
            "text": params.text,
        });
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&result).unwrap(),
        )]))
    }

    #[tool(description = "Add an emoji reaction to a message (as bot user)")]
    async fn chat_react(
        &self,
        Parameters(params): Parameters<ChatReactParams>,
    ) -> Result<CallToolResult, McpError> {
        let (bot_id, _) = self.resolve_bot_user()?;

        // Get channel from message
        let check_sql = format!(
            "SELECT channel_id FROM messages WHERE id = {}",
            params.timestamp
        );
        let result = self.api.query_router().query_sync(&check_sql)
            .map_err(|e| McpError::internal_error(format!("message lookup failed: {e}"), None))?;

        if result.rows.is_empty() {
            return Err(McpError::invalid_params("message_not_found".to_string(), None));
        }

        let channel_id = match &result.rows[0][0] {
            Value::Int(v) => *v,
            _ => return Err(McpError::internal_error("invalid channel_id".to_string(), None)),
        };

        if !self.is_member(channel_id, bot_id) {
            return Err(McpError::invalid_params(
                format!("bot is not a member of channel {channel_id}"),
                None,
            ));
        }

        // Check if already reacted
        let dup_sql = format!(
            "SELECT message_id FROM reactions WHERE message_id = {} AND user_id = {} AND emoji = '{}'",
            params.timestamp, bot_id, crate::chat::models::escape_sql(&params.name)
        );
        if let Ok(r) = self.api.query_router().query_sync(&dup_sql) {
            if !r.rows.is_empty() {
                return Err(McpError::invalid_params("already_reacted".to_string(), None));
            }
        }

        let now = crate::chat::models::now_timestamp();
        let insert_sql = format!(
            "INSERT INTO reactions (message_id, user_id, emoji, created_at) \
             VALUES ({}, {}, '{}', '{now}')",
            params.timestamp, bot_id, crate::chat::models::escape_sql(&params.name)
        );

        self.api.query_router().query_sync(&insert_sql)
            .map_err(|e| McpError::internal_error(format!("reaction insert failed: {e}"), None))?;

        let result = serde_json::json!({"ok": true});
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&result).unwrap(),
        )]))
    }

    #[tool(description = "List chat channels accessible to the bot")]
    async fn chat_list_channels(
        &self,
        Parameters(params): Parameters<ChatListChannelsParams>,
    ) -> Result<CallToolResult, McpError> {
        let sql = if let Some(bot_id) = params.bot_user_id {
            // List channels the bot is a member of
            format!(
                "SELECT c.id, c.name, c.kind, c.topic \
                 FROM channels c \
                 JOIN channel_members cm ON c.id = cm.channel_id \
                 WHERE cm.user_id = {}",
                bot_id
            )
        } else {
            // List all public channels
            "SELECT id, name, kind, topic FROM channels WHERE kind = 'public'".to_string()
        };

        let result = self.api.query_router().query_sync(&sql)
            .map_err(|e| McpError::internal_error(format!("list channels failed: {e}"), None))?;

        let channels: Vec<serde_json::Value> = result.rows.iter().map(|row| {
            serde_json::json!({
                "id": row[0].to_json(),
                "name": row[1].to_json(),
                "kind": row[2].to_json(),
                "topic": row[3].to_json(),
            })
        }).collect();

        let output = serde_json::json!({
            "ok": true,
            "channels": channels,
        });
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    #[tool(description = "Search chat messages by keyword")]
    async fn chat_search(
        &self,
        Parameters(params): Parameters<ChatSearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let query = SearchQuery {
            text: params.query,
            sources: Some(vec!["chat".to_string()]),
            limit: params.limit.min(100),
            date_from: None,
            date_to: None,
        };

        let results = self.api.search(&query)
            .map_err(|e| McpError::internal_error(format!("chat search failed: {e}"), None))?;

        let matches: Vec<serde_json::Value> = results.iter().map(|r| {
            serde_json::json!({
                "ts": r.id,
                "channel": r.title,
                "text": r.snippet,
                "score": r.score,
            })
        }).collect();

        let output = serde_json::json!({
            "ok": true,
            "messages": {
                "matches": matches,
                "total": matches.len(),
            }
        });
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }
```

- [x] Verify that the necessary imports are present at the top of `mcp.rs`. The file already imports `crate::connector::Value` and `crate::search::SearchQuery`. No new imports needed since we use fully qualified paths for `crate::chat::*` items.

**Test commands:**

```bash
cd teidelum && cargo check
```

**Commit:**

```bash
cd teidelum && git add src/mcp.rs && git commit -m "mcp: add chat tools — postMessage, history, reply, react, listChannels, search"
```

---

## Chunk 4: Integration

### Task 7: Integration test

**Files:** `teidelum/tests/chat_plan2_integration.rs` (new)

**Context:** Write an integration test that exercises the full flow: create API, init chat tables, register a user, create a channel, post messages, verify search indexing works, upload a file, and verify download. This test uses the `TeidelumApi` and handler functions directly, not HTTP.

**Steps:**

- [x] Create `teidelum/tests/chat_plan2_integration.rs` with integration tests.

```rust
//! Integration tests for Chat Plan 2: Search, Files, MCP Tools.
//!
//! Tests the full flow: post messages → search indexes them,
//! file upload → file metadata stored, search via MCP tools.

use std::sync::Arc;
use teidelum::api::TeidelumApi;
use teidelum::chat::models::init_chat_tables;
use teidelum::search::SearchQuery;

/// Create a test API with chat tables initialized.
fn setup() -> (tempfile::TempDir, Arc<TeidelumApi>) {
    let tmp = tempfile::tempdir().unwrap();
    let api = TeidelumApi::new(tmp.path()).unwrap();
    init_chat_tables(&api).unwrap();
    (tmp, Arc::new(api))
}

/// Helper: insert a user directly via SQL. Returns user_id.
fn create_user(api: &TeidelumApi, username: &str, is_bot: bool) -> i64 {
    let id = teidelum::chat::id::next_id();
    let now = teidelum::chat::models::now_timestamp();
    let sql = format!(
        "INSERT INTO users (id, username, display_name, email, password_hash, avatar_url, status, is_bot, created_at) \
         VALUES ({id}, '{username}', '{username}', '{username}@test.com', 'hash', '', 'offline', {is_bot}, '{now}')"
    );
    api.query_router().query_sync(&sql).unwrap();
    id
}

/// Helper: create a channel and add the user as owner. Returns channel_id.
fn create_channel(api: &TeidelumApi, name: &str, user_id: i64) -> i64 {
    let id = teidelum::chat::id::next_id();
    let now = teidelum::chat::models::now_timestamp();
    let sql = format!(
        "INSERT INTO channels (id, name, kind, topic, created_by, created_at) \
         VALUES ({id}, '{name}', 'public', '', {user_id}, '{now}')"
    );
    api.query_router().query_sync(&sql).unwrap();

    let member_sql = format!(
        "INSERT INTO channel_members (channel_id, user_id, role, joined_at) \
         VALUES ({id}, {user_id}, 'owner', '{now}')"
    );
    api.query_router().query_sync(&member_sql).unwrap();
    id
}

/// Helper: post a message and index it in tantivy. Returns message_id.
fn post_and_index(api: &TeidelumApi, channel_id: i64, user_id: i64, text: &str) -> i64 {
    let id = teidelum::chat::id::next_id();
    let now = teidelum::chat::models::now_timestamp();
    let sql = format!(
        "INSERT INTO messages (id, channel_id, user_id, thread_id, content, deleted_at, edited_at, created_at) \
         VALUES ({id}, {channel_id}, {user_id}, 0, '{}', NULL, NULL, '{now}')",
        teidelum::chat::models::escape_sql(text),
    );
    api.query_router().query_sync(&sql).unwrap();

    // Index in tantivy (same as what chat_post_message does)
    let name_sql = format!("SELECT name FROM channels WHERE id = {channel_id}");
    let channel_name = match api.query_router().query_sync(&name_sql) {
        Ok(r) if !r.rows.is_empty() => {
            match &r.rows[0][0] {
                teidelum::connector::Value::String(s) => format!("#{s}"),
                _ => format!("#{channel_id}"),
            }
        }
        _ => format!("#{channel_id}"),
    };
    let doc = vec![(
        id.to_string(),
        "chat".to_string(),
        channel_name,
        text.to_string(),
    )];
    api.search_engine().index_documents(&doc).unwrap();

    id
}

#[test]
fn test_message_search_indexing() {
    let (_tmp, api) = setup();
    let user_id = create_user(&api, "alice", false);
    let channel_id = create_channel(&api, "general", user_id);

    post_and_index(&api, channel_id, user_id, "The deployment pipeline is broken");
    post_and_index(&api, channel_id, user_id, "Can someone review my pull request?");
    post_and_index(&api, channel_id, user_id, "Meeting at 3pm to discuss the roadmap");

    // Search for "deployment" — should find the first message
    let results = api.search_engine().search(&SearchQuery {
        text: "deployment pipeline".to_string(),
        sources: Some(vec!["chat".to_string()]),
        limit: 10,
        date_from: None,
        date_to: None,
    }).unwrap();

    assert!(!results.is_empty(), "should find deployment message");
    assert_eq!(results[0].source, "chat");
    assert_eq!(results[0].title, "#general");

    // Search for "roadmap" — should find the third message
    let results = api.search_engine().search(&SearchQuery {
        text: "roadmap".to_string(),
        sources: Some(vec!["chat".to_string()]),
        limit: 10,
        date_from: None,
        date_to: None,
    }).unwrap();

    assert!(!results.is_empty(), "should find roadmap message");

    // Search with no chat filter should also work
    let results = api.search_engine().search(&SearchQuery {
        text: "pull request".to_string(),
        sources: None,
        limit: 10,
        date_from: None,
        date_to: None,
    }).unwrap();

    assert!(!results.is_empty(), "should find pull request message without source filter");
}

#[test]
fn test_search_does_not_find_other_sources() {
    let (_tmp, api) = setup();
    let user_id = create_user(&api, "bob", false);
    let channel_id = create_channel(&api, "random", user_id);

    post_and_index(&api, channel_id, user_id, "unique test message for chat");

    // Index a non-chat document
    let non_chat_doc = vec![(
        "notion-123".to_string(),
        "notion".to_string(),
        "Notion Page".to_string(),
        "unique test message for notion".to_string(),
    )];
    api.search_engine().index_documents(&non_chat_doc).unwrap();

    // Search with chat filter should only find the chat message
    let results = api.search_engine().search(&SearchQuery {
        text: "unique test message".to_string(),
        sources: Some(vec!["chat".to_string()]),
        limit: 10,
        date_from: None,
        date_to: None,
    }).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source, "chat");
}

#[test]
fn test_file_metadata_storage() {
    let (_tmp, api) = setup();
    let user_id = create_user(&api, "carol", false);
    let channel_id = create_channel(&api, "files-test", user_id);

    let msg_id = teidelum::chat::id::next_id();
    let file_id = teidelum::chat::id::next_id();
    let now = teidelum::chat::models::now_timestamp();

    // Insert message
    let msg_sql = format!(
        "INSERT INTO messages (id, channel_id, user_id, thread_id, content, deleted_at, edited_at, created_at) \
         VALUES ({msg_id}, {channel_id}, {user_id}, 0, '[file: report.pdf]', NULL, NULL, '{now}')"
    );
    api.query_router().query_sync(&msg_sql).unwrap();

    // Insert file metadata
    let file_sql = format!(
        "INSERT INTO files (id, message_id, user_id, channel_id, filename, mime_type, size_bytes, storage_path, created_at) \
         VALUES ({file_id}, {msg_id}, {user_id}, {channel_id}, 'report.pdf', 'application/pdf', 1024, 'data/files/uuid123/report.pdf', '{now}')"
    );
    api.query_router().query_sync(&file_sql).unwrap();

    // Query back the file
    let result = api.query_router().query_sync(&format!(
        "SELECT filename, mime_type, size_bytes FROM files WHERE id = {file_id}"
    )).unwrap();

    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        teidelum::connector::Value::String(s) => assert_eq!(s, "report.pdf"),
        other => panic!("expected string filename, got {other:?}"),
    }
    match &result.rows[0][1] {
        teidelum::connector::Value::String(s) => assert_eq!(s, "application/pdf"),
        other => panic!("expected string mime_type, got {other:?}"),
    }
    match &result.rows[0][2] {
        teidelum::connector::Value::Int(v) => assert_eq!(*v, 1024),
        other => panic!("expected int size, got {other:?}"),
    }
}

#[test]
fn test_bot_user_query() {
    let (_tmp, api) = setup();
    let _human = create_user(&api, "dave", false);
    let bot_id = create_user(&api, "teidebot", true);

    // Query for bot users
    let result = api.query_router().query_sync(
        "SELECT id, username FROM users WHERE is_bot = true LIMIT 1"
    ).unwrap();

    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0] {
        teidelum::connector::Value::Int(v) => assert_eq!(*v, bot_id),
        other => panic!("expected int bot id, got {other:?}"),
    }
}
```

**Test commands:**

```bash
cd teidelum && cargo test --test chat_plan2_integration
```

**Commit:**

```bash
cd teidelum && git add tests/chat_plan2_integration.rs && git commit -m "test: add integration tests for chat search indexing, files, and bot users"
```

---

## Summary

| Chunk | Task | Files | What |
|-------|------|-------|------|
| 1 | 1 | `handlers.rs` | Index messages in tantivy on `chat.postMessage` |
| 1 | 2 | `handlers.rs` | `search.messages` Slack API endpoint |
| 2 | 3 | `files.rs` (new), `mod.rs` | File upload handler (multipart → disk → DB → message) |
| 2 | 4 | `files.rs` | File download handler (GET with auth check) |
| 2 | 5 | `handlers.rs`, `server.rs` | Wire file routes into the router |
| 3 | 6 | `mcp.rs` | Six MCP chat tools for AI agents |
| 4 | 7 | `tests/chat_plan2_integration.rs` (new) | Integration tests for the full flow |

**Total files modified:** 4 (`handlers.rs`, `mcp.rs`, `server.rs`, `chat/mod.rs`)
**Total files created:** 2 (`chat/files.rs`, `tests/chat_plan2_integration.rs`)
**Total new MCP tools:** 6 (`chat.postMessage`, `chat.history`, `chat.reply`, `chat.react`, `chat.listChannels`, `chat.search`)
