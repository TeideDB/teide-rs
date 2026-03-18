# Guide Articles Design

**Goal:** Create 6 narrative-driven guide articles following the DuckDB blog style — problem-first, progressive complexity, realistic datasets, "try it yourself" challenges.

**Location:** `docs/guides/` as standalone HTML pages with shared sidebar/styling.

**Audience:** Mix of analysts, data engineers, graph practitioners, ML engineers, and Rust developers.

---

## Shared Template

Every article follows this skeleton:

1. **Hero Section** — Title, one-sentence hook, "What you'll learn" bullets, "~10 minutes"
2. **The Problem** — Realistic scenario, traditional pain point, "what if..." teaser
3. **Setup** — Self-contained dataset (CREATE TABLE + INSERT), exploration query
4. **Progressive Feature Sections (3-5)** — Each: Context → Query → Result Table → Interpretation
5. **Putting It All Together** — Complex query combining everything
6. **What's Next** — Links to related articles + reference docs

Code pattern: Setup → Query → Formatted Result → Interpretation. Every example is copy-pasteable.

---

## Article 1: Getting Started — From CSV to Insights in 5 Minutes

**Persona:** Analyst who knows SQL, new to TeideDB.
**Narrative:** Startup data analyst, CSV export, leadership meeting in 30 minutes.
**Dataset:** `employees` (15 rows: id, name, dept, title, salary, hire_date DATE, active BOOL)

### Sections
1. Launch & Load — REPL, CREATE TABLE, INSERT with date literals, basic SELECT
2. Explore the Data — DISTINCT, MIN/MAX, WHERE filtering
3. Ask Business Questions — GROUP BY, ORDER BY, AVG/STDDEV/COUNT per dept
4. Rank and Compare — ROW_NUMBER OVER (PARTITION BY), LAG for salary gaps
5. Derive New Tables — CREATE TABLE AS SELECT, export workflow
6. Constant Expressions — SELECT 1+1, CAST('2025-01-15' AS DATE)

### Challenges
- Find departments where average salary exceeds 80k
- Rank employees by hire date (longest tenure first)

---

## Article 2: SQL Deep Dive — Joins, Windows, and Subqueries

**Persona:** Data engineer validating multi-table analytics.
**Narrative:** E-commerce reporting pipeline, star schema, executive dashboard.
**Dataset:**
- `customers` (20 rows: id, name, email, region, joined_date DATE)
- `products` (15 rows: id, name, category, price REAL)
- `orders` (25 rows: id, customer_id, product_id, quantity, order_date DATE, status)

### Sections
1. Multi-Table Setup — Create 3 tables, quick exploration
2. Joining Tables — INNER JOIN, three-way join, qualified columns
3. LEFT JOIN for Complete Pictures — "Customers who haven't ordered", FULL OUTER JOIN
4. Aggregation Across Joins — Revenue per category/region, HAVING, top-N
5. Subqueries & CTEs — Scalar subquery in WHERE, multi-step CTE report
6. Window Functions for Analytics — Running total, RANK, LAG/LEAD
7. Putting It All Together — CTE-based executive dashboard

### Challenges
- Customer with highest single-order value
- Product's share of total revenue via window function

---

## Article 3: Working with Time — Dates, Timestamps, and Temporal Queries

**Persona:** Ops engineer analyzing event logs.
**Narrative:** Server event log analysis, outage patterns, incident investigation.
**Dataset:**
- `events` (25 rows: id, server_name, event_type, severity, occurred_at TIMESTAMP)
- `deployments` (8 rows: id, service_name, deployed_at TIMESTAMP, version)

### Sections
1. Creating Temporal Data — String literal INSERT, DATE/TIME/TIMESTAMP formats, CAST
2. Extracting Components — EXTRACT(YEAR/MONTH/DAY/HOUR/DOW), day-of-week error analysis
3. Truncating for Bucketing — DATE_TRUNC('hour'/'day'), hourly event count
4. Date Arithmetic — DATE_DIFF, response time calculations
5. Filtering by Time Ranges — BETWEEN timestamps, CURRENT_TIMESTAMP
6. Correlating Temporal Tables — JOIN events with deployments, post-deploy error spikes
7. Putting It All Together — Incident report: bucket, flag spikes, rank servers

### Challenges
- Longest gap between consecutive errors on any server
- Hour of day with highest error rate

---

## Article 4: Graph Queries with SQL/PGQ — Finding Patterns in Connected Data

**Persona:** Fraud analyst, network engineer. Flagship narrative article.
**Narrative:** Fintech fraud detection — circular flows, smurfing, hidden connections.
**Dataset:**
- `persons` (10 rows: id, name, risk_score REAL, country)
- `accounts` (15 rows: id, owner_id, account_type, balance REAL, opened_date DATE)
- `transfers` (25 rows: id, src_account, dst_account, amount REAL, transfer_date DATE, memo)

Planted patterns: one circular flow, one smurfing cluster, one legitimate trader.

### Sections
1. Why Graphs? — Show painful self-join/CTE approach, establish pain point
2. Creating a Property Graph — CREATE PROPERTY GRAPH, KEY, REFERENCES, labels
3. First MATCH: Direct Transfers — 1-hop pattern, COLUMNS, WHERE on amount
4. Following the Money: Multi-Hop — Variable-length {1,3}, ANY SHORTEST, path accessors
5. Detecting Circular Flows — Path back to origin {2,5}, CHEAPEST with COST
6. Graph Algorithms at Scale — PAGERANK, COMMUNITY, CONNECTED_COMPONENT, CLUSTERING_COEFFICIENT
7. Composing with SQL — GROUP BY on algorithm output, JOINs, CTEs wrapping GRAPH_TABLE
8. Putting It All Together — Full fraud investigation pipeline

### Challenges
- All persons within 2 transfer hops of 'Alice'
- Community with highest average risk_score

---

## Article 5: Vector Search — Embeddings, Similarity, and Nearest Neighbors

**Persona:** ML practitioner, search engineer.
**Narrative:** Developer docs knowledge base, "related articles" feature.
**Dataset:**
- `articles` (15 rows: id, title, category, published_date DATE)
- `embedding` column (8-dim F32 vectors, added via Rust API)

### Sections
1. What Are Embeddings? — Text→vector, cosine similarity intuition
2. Setting Up the Data — Rust API for embedding creation, then pure SQL
3. Cosine Similarity — COSINE_SIMILARITY with query vector, top-5
4. Euclidean Distance — Same query, when to use which
5. KNN: The Auto-Optimized Pattern — ORDER BY sim LIMIT k auto-detection
6. Building an HNSW Index — CREATE VECTOR INDEX, M/ef_construction explained
7. Limitations & Workarounds — DML restrictions, index lifecycle, honest constraints

### Challenges
- 3 articles most distant from a query vector
- Compare cosine vs euclidean rankings

---

## Article 6: Embedding TeideDB in Rust Applications

**Persona:** Rust developer building services with TeideDB.
**Narrative:** IoT sensor microservice — ingest, store, query, serve via PgWire.
**Dataset:** Built programmatically in Rust (sensor readings).

### Sections
1. Adding the Dependency — Cargo.toml, feature flags, build notes
2. Your First Table — Context, Table::from_vecs, Graph API, lazy DAG
3. Using SQL Instead — Session, execute(), ExecResult, reading results
4. Working with Dates and Types — Temporal inserts via SQL, type system
5. Adding Embeddings — create_embedding_column, HNSW from Rust, KNN queries
6. The PgWire Server — Programmatic start, thread model, psql/DBeaver
7. Critical Constraints — !Send + !Sync, ENGINE_LOCK, multi-session patterns

### Challenges
- Rolling average over last 10 readings
- PgWire endpoint with CSV pre-load

---

## Implementation Notes

- All articles share the same CSS/sidebar as existing docs
- New "Guides" sidebar section links to all 6 articles
- Each article is self-contained — can be read independently
- Code examples are copy-pasteable into the REPL (articles 1-5) or a Rust file (article 6)
- Result tables are shown in `<div class="output">` blocks matching existing docs
- "Try it yourself" challenges use `<div class="note">` styling
- Articles cross-reference each other and link to reference docs for functions used
