#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "optimized_impl.h"
#include "util.h"

struct VectorQueryRange
{
    uint32_t start;
    uint32_t end;

    VectorQueryRange() : start{0}, end{0}
    {}

    VectorQueryRange(uint32_t start, uint32_t end) : start{start}, end{end}
    {}
};

template<class T>
class ThreadScheduler
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
    VectorQueryRange task_range;
    uint64_t worker_size = 0;
    std::function<void(VectorQueryRange, Knn& knn)> task_fn;
    std::vector<T>& knns;

public:
    explicit ThreadScheduler(uint32_t thread_n, std::vector<T>& knns) : thread_n{thread_n}, knns{knns}
    {
        assert(knns.size() == thread_n);

        if (thread_n > 1)
        {
            threads.reserve(thread_n);
            for (uint32_t i = 0; i < thread_n; ++i)
            {
                threads.emplace_back(&ThreadScheduler::worker, this, i);
            }
        }
    }

    ~ThreadScheduler()
    {
        state.store(DONE);
        state.notify_all();
        for (auto& t: threads)
        {
            t.join();
        }
    }

    template<typename F>
    void parallelFor(VectorQueryRange range, F task)
    {
        if (thread_n <= 1) {
            task(range, knns[0]);
            return;
        }

        scheduler_mutex.lock();

        // prepare task for workers
        num_finished_workers = 0;
        task_range = range;
        task_fn = task;
        worker_size = (range.end - range.start) / thread_n;

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
                    uint64_t worker_begin = task_range.start + (tid * worker_size);
                    VectorQueryRange worker_range(worker_begin,
                                                  tid == thread_n - 1
                                                  ? task_range.end
                                                  : worker_begin + worker_size);

                    // execute task
                    task_fn(worker_range, knns[tid]);

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