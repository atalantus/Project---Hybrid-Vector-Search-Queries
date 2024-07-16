#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "optimized_impl.h"
#include "util.h"

template<class T>
class ThreadPool
{
    enum ThreadState : uint8_t
    {
        IDLE, TASK, DONE
    };

    uint32_t thread_n;
    std::vector<std::thread> threads;
    std::atomic<ThreadState> state;
    std::mutex scheduler_mutex;
    std::atomic<uint8_t> num_finished_workers;
    std::condition_variable task_cv;
    uint32_t work_range_size = 0;
    std::function<void(uint32_t start, uint32_t end, Knn& knn)> task_fn;
    std::vector<T>& knns;

public:
    explicit ThreadPool(uint32_t thread_n, std::vector<T>& knns) : thread_n{thread_n}, knns{knns}
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
        state.store(DONE);
        state.notify_all();
        for (auto& t: threads)
        {
            t.join();
        }
    }

    void parallel_for(uint32_t size, std::function<void(uint32_t, uint32_t, Knn&)> task)
    {
        if (thread_n <= 1)
        {
            task(0, size, knns[0]);
            return;
        }

        scheduler_mutex.lock();

        // prepare task for workers
        num_finished_workers = 0;
        work_range_size = size;
        task_fn = task;

        // wake up workers
        state.store(TASK);
        state.notify_all();

        // wait for workers to finish
        while (auto nfw = num_finished_workers < thread_n)
        {
            num_finished_workers.wait(nfw);
        }

        // reset task
        state.store(IDLE);
        state.notify_all();

        // wait for workers to reset
        while (auto nfw = num_finished_workers > 0)
        {
            num_finished_workers.wait(nfw);
        }

        scheduler_mutex.unlock();
    }

private:
    void worker(unsigned tid)
    {
        while (true)
        {
            state.wait(IDLE);

            switch (state.load())
            {
                case IDLE:
                    break;
                case DONE:
                    return;
                case TASK:
                {
                    uint32_t worker_size = work_range_size / thread_n;
                    uint32_t worker_start = tid * worker_size;
                    uint32_t worker_end = tid == thread_n - 1 ? work_range_size : worker_start + worker_size;

                    // execute task
                    task_fn(worker_start, worker_end, knns[tid]);

                    // notify finish
                    num_finished_workers++;
                    num_finished_workers.notify_one();

                    // wait for task to finish
                    while (state == TASK)
                    {
                        state.wait(TASK);
                    }

                    // notify reset
                    num_finished_workers--;
                    num_finished_workers.notify_one();

                    break;
                }
            }
        }
    }
};