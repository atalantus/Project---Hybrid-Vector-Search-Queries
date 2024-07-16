#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <cassert>

#include "optimized_impl.h"
#include "util.h"

template<class T>
class ThreadPool
{
    enum class ThreadState : uint8_t
    {
        IDLE, TASK, DONE
    };

    uint32_t thread_n;
    std::vector<std::thread> threads;

    ThreadState state;
    std::mutex state_mutex;
    std::condition_variable state_cv;

    uint32_t num_finished_workers;
    std::mutex finished_mutex;
    std::condition_variable finished_cv;

    uint32_t work_range_size = 0;
    std::function<void(uint32_t start, uint32_t end, T& knn)> task_fn;
    std::vector<T>& knns;

public:
    explicit ThreadPool(uint32_t thread_n, std::vector<T>& knns) : thread_n{thread_n}, knns{knns},
                                                                   state{ThreadState::IDLE}, num_finished_workers{0}
    {
        assert(knns.size() == thread_n);

        if (thread_n > 1)
        {
            threads.reserve(thread_n);
            for (uint32_t i = 0; i < thread_n; ++i)
            {
                threads.emplace_back(&ThreadPool::worker, this, i);
            }
        }
    }

    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            state = ThreadState::DONE;
        }
        state_cv.notify_all();
        for (auto& t: threads)
        {
            t.join();
        }
    }

    void parallel_for(uint32_t size, std::function<void(uint32_t, uint32_t, T&)> task)
    {
        if (thread_n <= 1)
        {
            task(0, size, knns[0]);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            work_range_size = size;
            task_fn = std::move(task);
            state = ThreadState::TASK;
        }
        state_cv.notify_all();

        {
            std::unique_lock<std::mutex> lock(finished_mutex);
            finished_cv.wait(lock, [this]
            { return num_finished_workers == thread_n; });
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            state = ThreadState::IDLE;
        }
        state_cv.notify_all();

        {
            std::unique_lock<std::mutex> lock(finished_mutex);
            finished_cv.wait(lock, [this]
            { return num_finished_workers == 0; });
        }
    }

private:
    void worker(unsigned tid)
    {
        while (true)
        {
            {
                std::unique_lock<std::mutex> lock(state_mutex);
                state_cv.wait(lock, [this]
                { return state != ThreadState::IDLE; });

                if (state == ThreadState::DONE)
                {
                    return;
                }

                if (state == ThreadState::TASK)
                {
                    uint32_t worker_size = work_range_size / thread_n;
                    uint32_t worker_start = tid * worker_size;
                    uint32_t worker_end = (tid == thread_n - 1) ? work_range_size : (worker_start + worker_size);

                    lock.unlock();
                    task_fn(worker_start, worker_end, knns[tid]);
                    lock.lock();

                    {
                        std::lock_guard<std::mutex> finished_lock(finished_mutex);
                        ++num_finished_workers;
                    }
                    finished_cv.notify_one();

                    state_cv.wait(lock, [this]
                    { return state == ThreadState::IDLE; });

                    {
                        std::lock_guard<std::mutex> finished_lock(finished_mutex);
                        --num_finished_workers;
                    }
                    finished_cv.notify_one();
                }
            }
        }
    }
};