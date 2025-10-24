#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

// CUDA event-based timer for kernel-only timing
class CudaTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool created;

public:
    CudaTimer() : created(false) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        created = true;
    }

    ~CudaTimer() {
        if (created) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }
    }

    // Start timing
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_event, stream);
    }

    // Stop timing and return elapsed milliseconds
    float stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);

        return milliseconds;
    }
};

// Benchmark result structure
struct BenchResult {
    std::string config_name;
    int64_t elements;
    int bitwidth;
    int num_partitions;
    float kernel_time_ms;
    double throughput_gbps;
    int warmup_iters;
    int timed_iters;
};

// Benchmark runner with warmup
template<typename KernelFunc, typename... Args>
BenchResult runBenchmark(
    const std::string& name,
    int64_t elements,
    int bitwidth,
    int num_partitions,
    int warmup_iters,
    int timed_iters,
    KernelFunc kernel_func,
    Args&&... args)
{
    CudaTimer timer;

    // Warmup runs
    for (int i = 0; i < warmup_iters; ++i) {
        kernel_func(std::forward<Args>(args)...);
    }
    cudaDeviceSynchronize();

    // Timed runs
    std::vector<float> times;
    times.reserve(timed_iters);

    for (int i = 0; i < timed_iters; ++i) {
        timer.start();
        kernel_func(std::forward<Args>(args)...);
        float elapsed = timer.stop();
        times.push_back(elapsed);
    }

    // Compute median time
    std::sort(times.begin(), times.end());
    float median_time = times[timed_iters / 2];

    // Calculate throughput (assuming 4-byte elements)
    const double bytes = static_cast<double>(elements) * sizeof(int32_t);
    const double gbps = (bytes / (median_time * 1e-3)) / 1e9;

    BenchResult result;
    result.config_name = name;
    result.elements = elements;
    result.bitwidth = bitwidth;
    result.num_partitions = num_partitions;
    result.kernel_time_ms = median_time;
    result.throughput_gbps = gbps;
    result.warmup_iters = warmup_iters;
    result.timed_iters = timed_iters;

    return result;
}

// Print benchmark result
void printBenchResult(const BenchResult& result) {
    std::cout << "=== Benchmark Result ===" << std::endl;
    std::cout << "Config: " << result.config_name << std::endl;
    std::cout << "Elements: " << result.elements << std::endl;
    std::cout << "Bitwidth: " << result.bitwidth << std::endl;
    std::cout << "Partitions: " << result.num_partitions << std::endl;
    std::cout << "Kernel Time: " << result.kernel_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << result.throughput_gbps << " GB/s" << std::endl;
    std::cout << "Warmup/Timed Iters: " << result.warmup_iters << "/" << result.timed_iters << std::endl;
    std::cout << std::endl;
}

// Write result to CSV
void writeBenchResultToCSV(const std::string& filename, const BenchResult& result, bool write_header = false) {
    FILE* fp = fopen(filename.c_str(), write_header ? "w" : "a");
    if (!fp) {
        std::cerr << "Failed to open CSV file: " << filename << std::endl;
        return;
    }

    if (write_header) {
        fprintf(fp, "datetime,config,elements,bitwidth,partitions,kernel_ms,gbps,warmup,timed\n");
    }

    // Get current time
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now_c));

    fprintf(fp, "%s,%s,%lld,%d,%d,%.6f,%.3f,%d,%d\n",
            time_str,
            result.config_name.c_str(),
            (long long)result.elements,
            result.bitwidth,
            result.num_partitions,
            result.kernel_time_ms,
            result.throughput_gbps,
            result.warmup_iters,
            result.timed_iters);

    fclose(fp);
}

// CPU timer for reference
class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;  // Return milliseconds
    }
};

// CUDA error checking helper
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)
