/*
 * Failure-only NVRTC diagnostics. No compile options or program contents change.
 *
 * Build on the target Linux host:
 *   cc -std=c11 -shared -fPIC -O2 -Wall -Wextra nvrtc_trace.c -ldl -o nvrtc_trace.so
 * Run one bounded probe with LD_PRELOAD=/absolute/path/nvrtc_trace.so.
 *
 * cuDNN may resolve NVRTC through dlsym on a private dlopen handle. On glibc,
 * the narrowly scoped dlsym interceptor below also covers that path. Other
 * dlsym requests return the original result. If multiple NVRTC providers are
 * discovered, additional providers are left untouched to preserve dispatch.
 *
 * Each failed compile emits at most 64 KiB. Compiler source dumping is absent.
 * A compiler log larger than 8 MiB is omitted to bound diagnostic allocation.
 * Successful compiles emit no output. The original nvrtcResult is returned.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *nvrtc_program;
typedef int (*compile_fn)(nvrtc_program, int, const char *const *);
typedef int (*log_size_fn)(nvrtc_program, size_t *);
typedef int (*log_fn)(nvrtc_program, char *);
typedef int (*version_fn)(int *, int *);
typedef const char *(*error_string_fn)(int);
typedef void *(*dlsym_fn)(void *, const char *);

enum { TRACE_CAP = 65536, LOG_ALLOC_CAP = 8 * 1024 * 1024 };
struct compile_snapshot {
    int count;
    int captured;
    char options[128][257];
};
static _Atomic(uintptr_t) compiler_address;
static _Atomic(uintptr_t) provider_handle;
static _Thread_local int in_trace;

int nvrtcCompileProgram(nvrtc_program program, int count, const char *const *options);

static dlsym_fn real_dlsym(void) {
#if defined(__GLIBC__)
    /* These are the original symbol versions for aarch64 and x86_64 glibc. */
#if defined(__aarch64__)
    return (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
#elif defined(__x86_64__)
    return (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
#else
    return (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
#endif
#else
    return dlsym;
#endif
}

static void *register_compiler(void *address, void *handle) {
    uintptr_t expected = 0;
    if (address == NULL || address == (void *)nvrtcCompileProgram)
        return address;
    if (atomic_compare_exchange_strong(&compiler_address, &expected, (uintptr_t)address)) {
        atomic_store(&provider_handle, (uintptr_t)handle);
        return (void *)nvrtcCompileProgram;
    }
    return expected == (uintptr_t)address ? (void *)nvrtcCompileProgram : address;
}

#if defined(__GLIBC__)
void *dlsym(void *handle, const char *name) {
    dlsym_fn lookup = real_dlsym();
    if (lookup == NULL)
        return NULL;
    void *address = lookup(handle, name);
    if (strcmp(name, "nvrtcCompileProgram") == 0)
        return register_compiler(address, handle);
    return address;
}
#endif

static void append(char *buffer, size_t *used, const char *format, ...) {
    if (*used >= TRACE_CAP - 1)
        return;
    va_list args;
    va_start(args, format);
    int n = vsnprintf(buffer + *used, TRACE_CAP - *used, format, args);
    va_end(args);
    if (n > 0) {
        size_t remaining = TRACE_CAP - 1 - *used;
        *used += (size_t)n < remaining ? (size_t)n : remaining;
    }
}

static void snapshot_options(struct compile_snapshot *snapshot, int count,
                             const char *const *options) {
    snapshot->count = count;
    snapshot->captured = 0;
    for (int i = 0; options != NULL && i < count && i < 128; ++i) {
        char *destination = snapshot->options[i];
        if (options[i] == NULL) {
            strcpy(destination, "<null>");
        } else {
            size_t j = 0;
            for (; j < 256 && options[i][j] != '\0'; ++j) {
                unsigned char c = (unsigned char)options[i][j];
                destination[j] = c >= 32 && c <= 126 ? (char)c : '?';
            }
            destination[j] = '\0';
        }
        snapshot->captured++;
    }
}

static void emit_failure(nvrtc_program program,
                         const struct compile_snapshot *snapshot, int result) {
    dlsym_fn lookup = real_dlsym();
    if (lookup == NULL)
        return;
    void *handle = (void *)atomic_load(&provider_handle);
    version_fn version = (version_fn)lookup(handle, "nvrtcVersion");
    log_size_fn get_size = (log_size_fn)lookup(handle, "nvrtcGetProgramLogSize");
    log_fn get_log = (log_fn)lookup(handle, "nvrtcGetProgramLog");
    error_string_fn error_string = (error_string_fn)lookup(handle, "nvrtcGetErrorString");
    char *buffer = malloc(TRACE_CAP);
    if (buffer == NULL)
        return;
    size_t used = 0;
    int major = -1, minor = -1;
    if (version != NULL)
        version(&major, &minor);
    append(buffer, &used, "[nvrtc-trace] failed nvrtcCompileProgram result=%d (%s) version=%d.%d options=%d\n",
           result, error_string != NULL ? error_string(result) : "unknown", major, minor, snapshot->count);
    for (int i = 0; i < snapshot->captured; ++i)
        append(buffer, &used, "[nvrtc-trace] option[%d]=%s\n", i, snapshot->options[i]);
    if (snapshot->count > 128)
        append(buffer, &used, "[nvrtc-trace] remaining options omitted\n");
    size_t log_bytes = 0;
    int size_status = get_size != NULL ? get_size(program, &log_bytes) : -1;
    append(buffer, &used, "[nvrtc-trace] program_log_bytes=%zu get_size_status=%d\n", log_bytes, size_status);
    if (size_status == 0 && log_bytes > 0 && log_bytes <= LOG_ALLOC_CAP && get_log != NULL) {
        char *log = malloc(log_bytes);
        if (log != NULL) {
            int status = get_log(program, log);
            if (status == 0) {
                log[log_bytes - 1] = '\0';
                append(buffer, &used, "[nvrtc-trace] program log (bounded):\n%s\n", log);
            } else {
                append(buffer, &used, "[nvrtc-trace] get_log_status=%d\n", status);
            }
            free(log);
        }
    } else if (log_bytes > LOG_ALLOC_CAP) {
        append(buffer, &used, "[nvrtc-trace] program log exceeds 8 MiB; omitted\n");
    }
    if (used == TRACE_CAP - 1) {
        const char marker[] = "\n[nvrtc-trace] truncated at 64 KiB\n";
        memcpy(buffer + used - (sizeof(marker) - 1), marker, sizeof(marker) - 1);
    }
    flockfile(stderr);
    fwrite(buffer, 1, used, stderr);
    fflush(stderr);
    funlockfile(stderr);
    free(buffer);
}

int nvrtcCompileProgram(nvrtc_program program, int count, const char *const *options) {
    compile_fn compile = (compile_fn)atomic_load(&compiler_address);
    if (compile == NULL) {
        dlsym_fn lookup = real_dlsym();
        if (lookup != NULL) {
            void *address = lookup(RTLD_NEXT, "nvrtcCompileProgram");
            register_compiler(address, RTLD_NEXT);
            compile = (compile_fn)address;
        }
    }
    if (compile == NULL || compile == nvrtcCompileProgram) {
        /* This cannot occur for a correctly loaded NVRTC provider. */
        fputs("[nvrtc-trace] unable to resolve original nvrtcCompileProgram\n", stderr);
        abort();
    }
    /* Some providers may release temporary option storage before returning. */
    struct compile_snapshot snapshot;
    snapshot_options(&snapshot, count, options);
    int result = compile(program, count, options);
    if (result != 0 && !in_trace) {
        in_trace = 1;
        emit_failure(program, &snapshot, result);
        in_trace = 0;
    }
    return result;
}
