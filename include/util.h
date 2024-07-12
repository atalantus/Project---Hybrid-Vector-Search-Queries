#pragma once

#include <cstdint>
#include <immintrin.h>

#if ENABLE_PERF_DBG
#define PERF_DBG(...) __VA_ARGS__;
#else
#define PERF_DBG(...)
#endif

uint64_t rdtsc()
{
    return __rdtsc();
}

#ifdef MEM_TRACK

uint64_t total_bytes_alloc = 0;

void* operator new(size_t size) {
    total_bytes_alloc += size;
    return malloc(size);
}

void* operator new[](size_t size) {
    total_bytes_alloc += size;
    return malloc(size);
}

void operator delete(void* memory, size_t size) {
    total_bytes_alloc -= size;
    free(memory);
}

void operator delete[](void* memory, size_t size) {
    total_bytes_alloc -= size;
    free(memory);
}

#endif