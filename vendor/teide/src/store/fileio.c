/*
 *   Copyright (c) 2024-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "fileio.h"

#ifdef _WIN32

#include <errno.h>

/* ===== Windows implementation ===== */

/* Translate GetLastError() into errno so callers can use errno portably. */
static void win_set_errno(void) {
    DWORD e = GetLastError();
    switch (e) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:    errno = ENOENT;  break;
    case ERROR_ACCESS_DENIED:     errno = EACCES;  break;
    case ERROR_WRITE_PROTECT:     errno = EROFS;   break;
    case ERROR_TOO_MANY_OPEN_FILES: errno = EMFILE; break;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:    errno = EEXIST;  break;
    default:                      errno = EIO;     break;
    }
}

td_fd_t td_file_open(const char* path, int flags) {
    if (!path) return TD_FD_INVALID;

    DWORD access = 0;
    DWORD creation = OPEN_EXISTING;

    if (flags & TD_OPEN_READ)  access |= GENERIC_READ;
    if (flags & TD_OPEN_WRITE) access |= GENERIC_WRITE;
    if (flags & TD_OPEN_CREATE) creation = OPEN_ALWAYS;

    HANDLE h = CreateFileA(path, access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           NULL, creation, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) win_set_errno();
    return h;
}

void td_file_close(td_fd_t fd) {
    if (fd != TD_FD_INVALID) CloseHandle(fd);
}

td_err_t td_file_lock_ex(td_fd_t fd) {
    if (fd == TD_FD_INVALID) return TD_ERR_IO;
    OVERLAPPED ov = {0};
    if (!LockFileEx(fd, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &ov))
        return TD_ERR_IO;
    return TD_OK;
}

td_err_t td_file_lock_sh(td_fd_t fd) {
    if (fd == TD_FD_INVALID) return TD_ERR_IO;
    OVERLAPPED ov = {0};
    if (!LockFileEx(fd, 0, 0, MAXDWORD, MAXDWORD, &ov))
        return TD_ERR_IO;
    return TD_OK;
}

td_err_t td_file_unlock(td_fd_t fd) {
    if (fd == TD_FD_INVALID) return TD_OK;
    OVERLAPPED ov = {0};
    if (!UnlockFileEx(fd, 0, MAXDWORD, MAXDWORD, &ov))
        return TD_ERR_IO;
    return TD_OK;
}

td_err_t td_file_sync(td_fd_t fd) {
    if (fd == TD_FD_INVALID) return TD_ERR_IO;
    if (!FlushFileBuffers(fd)) return TD_ERR_IO;
    return TD_OK;
}

td_err_t td_file_sync_dir(const char* path) {
    /* Windows: rename durability is handled by MOVEFILE_WRITE_THROUGH in
     * td_file_rename; no separate directory fsync needed. */
    (void)path;
    return TD_OK;
}

td_err_t td_file_rename(const char* old_path, const char* new_path) {
    if (!old_path || !new_path) return TD_ERR_IO;
    /* MOVEFILE_WRITE_THROUGH flushes the rename to disk before returning,
     * providing crash-safe durability equivalent to POSIX fsync-after-rename. */
    if (!MoveFileExA(old_path, new_path,
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return TD_ERR_IO;
    return TD_OK;
}

#else

/* ===== POSIX implementation ===== */

#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

td_fd_t td_file_open(const char* path, int flags) {
    if (!path) return TD_FD_INVALID;

    int oflags = 0;
    if ((flags & TD_OPEN_READ) && (flags & TD_OPEN_WRITE))
        oflags = O_RDWR;
    else if (flags & TD_OPEN_WRITE)
        oflags = O_WRONLY;
    else
        oflags = O_RDONLY;

    if (flags & TD_OPEN_CREATE) oflags |= O_CREAT;

    return open(path, oflags, 0644);
}

void td_file_close(td_fd_t fd) {
    if (fd != TD_FD_INVALID) close(fd);
}

td_err_t td_file_lock_ex(td_fd_t fd) {
    if (fd == TD_FD_INVALID) return TD_ERR_IO;
    if (flock(fd, LOCK_EX) != 0) return TD_ERR_IO;
    return TD_OK;
}

td_err_t td_file_lock_sh(td_fd_t fd) {
    if (fd == TD_FD_INVALID) return TD_ERR_IO;
    if (flock(fd, LOCK_SH) != 0) return TD_ERR_IO;
    return TD_OK;
}

td_err_t td_file_unlock(td_fd_t fd) {
    if (fd == TD_FD_INVALID) return TD_OK;
    if (flock(fd, LOCK_UN) != 0) return TD_ERR_IO;
    return TD_OK;
}

td_err_t td_file_sync(td_fd_t fd) {
    if (fd == TD_FD_INVALID) return TD_ERR_IO;
    if (fsync(fd) != 0) return TD_ERR_IO;
    return TD_OK;
}

td_err_t td_file_sync_dir(const char* path) {
    if (!path) return TD_ERR_IO;
    /* Extract parent directory from path */
    char dir[1024];
    size_t len = strlen(path);
    if (len >= sizeof(dir)) return TD_ERR_IO;
    memcpy(dir, path, len + 1);
    /* Find last '/' */
    char* slash = strrchr(dir, '/');
    if (slash) {
        if (slash == dir)
            dir[1] = '\0';  /* root directory */
        else
            *slash = '\0';
    } else {
        dir[0] = '.'; dir[1] = '\0';  /* current directory */
    }
    int fd = open(dir, O_RDONLY);
    if (fd < 0) return TD_ERR_IO;
    int rc = fsync(fd);
    close(fd);
    return (rc == 0) ? TD_OK : TD_ERR_IO;
}

td_err_t td_file_rename(const char* old_path, const char* new_path) {
    if (!old_path || !new_path) return TD_ERR_IO;
    if (rename(old_path, new_path) != 0) return TD_ERR_IO;
    return TD_OK;
}

#endif
