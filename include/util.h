#pragma once

#include <cstdint>
#include <immintrin.h>

/*
 * If enabled __rdtsc() instruction will be used measure performance via cycles.
 * Otherwise, chrono::high_resolution_clock will be used to measure performance in
 * nanoseconds.
 *
 * Note that __rdtsc() should be much more precise due to the way I placed my benchmark
 * counters throughout the code.
 */
#define RDTSC 1

#if ENABLE_PERF_DBG
#define PERF_DBG(...) __VA_ARGS__;
#else
#define PERF_DBG(...)
#endif

auto get_ts()
{
#if RDTSC
    return __rdtsc();
#else
    return std::chrono::high_resolution_clock::now();
#endif
}

template<typename T>
auto ts_dur(T& ts_start, T& ts_end)
{
#if RDTSC
    return ts_end - ts_start;
#else
    auto c = std::chrono::duration_cast<std::chrono::nanoseconds>(ts_end - ts_start).count();
    // ts_end becomes the next start point
    ts_end = std::chrono::high_resolution_clock::now();
    return c;
#endif
}

template<typename T>
auto ts_dur_now(T& ts_start)
{
    auto now = get_ts();

#if RDTSC
    return now - ts_start;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now - ts_start).count();
#endif
}


template<typename It, typename T, typename Compare = std::less<>>
auto lower_bound_branchless(It low, It last, const T& val, Compare lt = {})
{
    auto n = std::distance(low, last);

    while (auto half = n / 2)
    {
        auto middle = low;
        std::advance(middle, half);
        low = lt(*middle, val) ? middle : low;
        n -= half;
    }
    if (lt(*low, val))
        ++low;
    return low;
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