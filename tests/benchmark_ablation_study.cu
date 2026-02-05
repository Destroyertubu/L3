/**
 * Decompression Ablation Study Benchmark
 *
 * Tests decompression kernels from NAIVE to HIGHLY OPTIMIZED:
 * - Horizontal (L3): H-L0 to H-L4
 * - Vertical (Vertical): V-L0 to V-L2
 *
 * Measures the contribution of each optimization technique.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cmath>

// L3 headers
#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "L3_codec.hpp"
#include "L3_opt.h"
#include "sosd_loader.h"

// Partitioner
#include "../src/kernels/compression/encoder_cost_optimal_gpu_merge_v2.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

// ============================================================================
// Configuration
// ============================================================================

const int WARMUP_RUNS = 3;
const int TIMED_RUNS = 5;
const int PARTITION_SIZE = 4096;
const std::string DATA_DIR = "data/sosd/";

// ============================================================================
// Kernel Enum and Names
// ============================================================================

enum class KernelLevel {
    // Horizontal (L3) Layout
    H_L0_SIMPLE,           // launchDecompressSimple
    H_L1_OPTIMIZED,        // launchDecompressOptimized
    H_L2_SPECIALIZED,      // launchDecompressSpecialized (template-specialized)
    H_L3_WARP_OPT,         // launchDecompressWarpOpt (Double-buffer+cp.async)

    // Vertical (Vertical) Layout
    // NOTE: In v3.0, all DecompressMode values route to the same kernel
    // (decompressInterleavedAllPartitions). We keep only one entry to
    // accurately represent the implementation.
    V_OPTIMIZED            // decompressAll (unified kernel in v3.0)
};

const char* getKernelName(KernelLevel k) {
    switch(k) {
        case KernelLevel::H_L0_SIMPLE: return "H-L0-Simple";
        case KernelLevel::H_L1_OPTIMIZED: return "H-L1-Optimized";
        case KernelLevel::H_L2_SPECIALIZED: return "H-L2-Specialized";
        case KernelLevel::H_L3_WARP_OPT: return "H-L3-WarpOpt";
        case KernelLevel::V_OPTIMIZED: return "V-Optimized";
        default: return "Unknown";
    }
}

const char* getOptimizations(KernelLevel k) {
    switch(k) {
        case KernelLevel::H_L0_SIMPLE: return "Naive grid-stride";
        case KernelLevel::H_L1_OPTIMIZED: return "+SharedMem +WarpCoop";
        case KernelLevel::H_L2_SPECIALIZED: return "+TemplateSpecialized";
        case KernelLevel::H_L3_WARP_OPT: return "+DoubleBuffer +FunnelShift +cp.async";
        case KernelLevel::V_OPTIMIZED: return "MiniVector(256) +Branchless +Interleaved";
        default: return "";
    }
}

bool isHorizontal(KernelLevel k) {
    return k == KernelLevel::H_L0_SIMPLE ||
           k == KernelLevel::H_L1_OPTIMIZED ||
           k == KernelLevel::H_L2_SPECIALIZED ||
           k == KernelLevel::H_L3_WARP_OPT;
}

// ============================================================================
// Result Structure
// ============================================================================

struct AblationResult {
    std::string dataset;
    std::string kernel_name;
    std::string layout;
    std::string optimizations;
    int level;
    double kernel_ms;
    double throughput_gbps;
    double speedup_vs_baseline;
    double speedup_vs_previous;
    size_t num_elements;
    bool correct;
};

// ============================================================================
// Partitioner Helper
// ============================================================================

template<typename T>
std::vector<PartitionInfo> generatePartitions(const std::vector<T>& data, int partition_size) {
    CostOptimalConfig config;
    config.target_partition_size = partition_size;
    config.analysis_block_size = partition_size / 2;
    config.min_partition_size = 256;
    config.max_partition_size = partition_size * 2;
    config.breakpoint_threshold = 2;
    config.merge_benefit_threshold = 0.05f;
    config.max_merge_rounds = 4;
    config.enable_merging = true;
    config.enable_polynomial_models = true;
    config.polynomial_min_size = 10;
    config.cubic_min_size = 20;
    config.polynomial_cost_threshold = 0.95f;

    GPUCostOptimalPartitionerV2<T> partitioner(data, config, 0);
    return partitioner.partition();
}

// ============================================================================
// Timing Helper
// ============================================================================

double getMedian(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    size_t n = times.size();
    if (n % 2 == 0) {
        return (times[n/2 - 1] + times[n/2]) / 2.0;
    }
    return times[n/2];
}

// ============================================================================
// Benchmark Kernel Dispatch
// ============================================================================

template<typename T>
double benchmarkHorizontalKernel(
    KernelLevel kernel,
    const CompressedDataL3<T>* h_compressed,
    T* d_output,
    size_t n,
    cudaStream_t stream)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        switch(kernel) {
            case KernelLevel::H_L0_SIMPLE:
                launchDecompressSimple(h_compressed, d_output, stream);
                break;
            case KernelLevel::H_L1_OPTIMIZED:
                launchDecompressOptimized(h_compressed, d_output, stream);
                break;
            case KernelLevel::H_L2_SPECIALIZED:
                launchDecompressSpecialized(h_compressed, d_output, 0, stream);
                break;
            case KernelLevel::H_L3_WARP_OPT:
                launchDecompressWarpOpt(h_compressed, d_output, stream);
                break;
            default:
                break;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Timed runs
    std::vector<double> times;
    for (int i = 0; i < TIMED_RUNS; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));

        switch(kernel) {
            case KernelLevel::H_L0_SIMPLE:
                launchDecompressSimple(h_compressed, d_output, stream);
                break;
            case KernelLevel::H_L1_OPTIMIZED:
                launchDecompressOptimized(h_compressed, d_output, stream);
                break;
            case KernelLevel::H_L2_SPECIALIZED:
                launchDecompressSpecialized(h_compressed, d_output, 0, stream);
                break;
            case KernelLevel::H_L3_WARP_OPT:
                launchDecompressWarpOpt(h_compressed, d_output, stream);
                break;
            default:
                break;
        }

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return getMedian(times);
}

template<typename T>
double benchmarkVerticalKernel(
    KernelLevel kernel,
    const CompressedDataVertical<T>& fl_compressed,
    T* d_output,
    size_t n,
    cudaStream_t stream)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // NOTE: In v3.0, all DecompressMode values route to the same kernel
    // (decompressInterleavedAllPartitions). We use INTERLEAVED as the canonical mode.

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        Vertical_decoder::decompressAll(fl_compressed, d_output, DecompressMode::INTERLEAVED, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Timed runs
    std::vector<double> times;
    for (int i = 0; i < TIMED_RUNS; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));

        Vertical_decoder::decompressAll(fl_compressed, d_output, DecompressMode::INTERLEAVED, stream);

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return getMedian(times);
}

// ============================================================================
// Verify Correctness
// ============================================================================

template<typename T>
bool verifyOutput(const T* d_output, const std::vector<T>& expected, size_t n) {
    std::vector<T> output(n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; i++) {
        if (output[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Run Ablation for Single Dataset
// ============================================================================

template<typename T>
std::vector<AblationResult> runAblation(
    const std::vector<T>& data,
    const std::string& dataset_name)
{
    std::vector<AblationResult> results;
    size_t n = data.size();
    size_t data_bytes = n * sizeof(T);

    std::cout << "\nDataset: " << dataset_name << " (" << n << " elements, "
              << (sizeof(T) == 8 ? "uint64" : "uint32") << ")" << std::endl;

    // Generate partitions
    auto partitions = generatePartitions(data, PARTITION_SIZE);
    std::cout << "  Partitions: " << partitions.size() << std::endl;

    // Compress with Horizontal (L3) encoder
    auto* h_compressed = compressDataWithPartitions(data, partitions, nullptr);
    if (!h_compressed) {
        std::cerr << "  Horizontal compression failed!" << std::endl;
        return results;
    }

    // Compress with Vertical (Vertical) encoder
    VerticalConfig fl_config;
    auto fl_compressed = Vertical_encoder::encodeVertical<T>(data, partitions, fl_config, 0);

    // Allocate output
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));

    // Test Horizontal kernels
    std::cout << "  [Horizontal Layout]" << std::endl;
    std::vector<KernelLevel> h_kernels = {
        KernelLevel::H_L0_SIMPLE,
        KernelLevel::H_L1_OPTIMIZED,
        KernelLevel::H_L2_SPECIALIZED,
        KernelLevel::H_L3_WARP_OPT
    };

    double h_baseline_gbps = 0;
    double h_prev_gbps = 0;

    for (auto kernel : h_kernels) {
        double ms = benchmarkHorizontalKernel(kernel, h_compressed, d_output, n, 0);
        double gbps = (data_bytes / 1e9) / (ms / 1e3);
        bool correct = verifyOutput(d_output, data, n);

        AblationResult r;
        r.dataset = dataset_name;
        r.kernel_name = getKernelName(kernel);
        r.layout = "Horizontal";
        r.optimizations = getOptimizations(kernel);
        r.level = static_cast<int>(kernel);
        r.kernel_ms = ms;
        r.throughput_gbps = gbps;
        r.num_elements = n;
        r.correct = correct;

        if (kernel == KernelLevel::H_L0_SIMPLE) {
            h_baseline_gbps = gbps;
            r.speedup_vs_baseline = 1.0;
            r.speedup_vs_previous = 1.0;
        } else {
            r.speedup_vs_baseline = gbps / h_baseline_gbps;
            r.speedup_vs_previous = (h_prev_gbps > 0) ? gbps / h_prev_gbps : 1.0;
        }
        h_prev_gbps = gbps;

        std::cout << "  " << std::left << std::setw(16) << r.kernel_name
                  << std::right << std::setw(8) << std::fixed << std::setprecision(1) << gbps << " GB/s"
                  << " (" << std::setprecision(2) << r.speedup_vs_baseline << "x vs baseline)"
                  << (correct ? " PASS" : " FAIL") << std::endl;

        results.push_back(r);
    }

    // Test Vertical kernel (single unified kernel in v3.0)
    std::cout << "  [Vertical Layout]" << std::endl;

    // NOTE: In v3.0, all DecompressMode values route to the same kernel
    // (decompressInterleavedAllPartitions). We test only V_OPTIMIZED.
    KernelLevel v_kernel = KernelLevel::V_OPTIMIZED;
    double ms = benchmarkVerticalKernel(v_kernel, fl_compressed, d_output, n, 0);
    double gbps = (data_bytes / 1e9) / (ms / 1e3);
    bool correct = verifyOutput(d_output, data, n);

    AblationResult v_result;
    v_result.dataset = dataset_name;
    v_result.kernel_name = getKernelName(v_kernel);
    v_result.layout = "Vertical";
    v_result.optimizations = getOptimizations(v_kernel);
    v_result.level = 0;
    v_result.kernel_ms = ms;
    v_result.throughput_gbps = gbps;
    v_result.num_elements = n;
    v_result.correct = correct;
    v_result.speedup_vs_baseline = 1.0;  // Only one level, it's its own baseline
    v_result.speedup_vs_previous = 1.0;

    std::cout << "  " << std::left << std::setw(16) << v_result.kernel_name
              << std::right << std::setw(8) << std::fixed << std::setprecision(1) << gbps << " GB/s"
              << " (unified v3.0 kernel)"
              << (correct ? " PASS" : " FAIL") << std::endl;

    results.push_back(v_result);

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    freeCompressedData(h_compressed);
    Vertical_encoder::freeCompressedData(fl_compressed);

    return results;
}

// ============================================================================
// Generate Report
// ============================================================================

void generateReport(const std::vector<AblationResult>& all_results, const std::string& output_path) {
    std::ofstream file(output_path);

    file << "# Decompression Ablation Study Report\n\n";

    // Get GPU name
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    file << "**GPU**: " << prop.name << "\n";
    file << "**Partition Size**: " << PARTITION_SIZE << "\n";
    file << "**Warmup Runs**: " << WARMUP_RUNS << ", **Timed Runs**: " << TIMED_RUNS << " (median)\n\n";

    // Collect unique datasets
    std::vector<std::string> datasets;
    for (const auto& r : all_results) {
        if (std::find(datasets.begin(), datasets.end(), r.dataset) == datasets.end()) {
            datasets.push_back(r.dataset);
        }
    }

    // Section 1: Horizontal Layout Progression
    file << "## 1. Horizontal (L3) Layout Performance Progression\n\n";
    file << "| Level | Kernel | Optimizations |";
    for (const auto& ds : datasets) file << " " << ds << " |";
    file << " Avg |\n";
    file << "|-------|--------|---------------|";
    for (size_t i = 0; i < datasets.size(); i++) file << "------|";
    file << "-----|\n";

    std::vector<KernelLevel> h_kernels = {
        KernelLevel::H_L0_SIMPLE,
        KernelLevel::H_L1_OPTIMIZED,
        KernelLevel::H_L2_SPECIALIZED,
        KernelLevel::H_L3_WARP_OPT
    };

    for (auto kernel : h_kernels) {
        int level_num = (kernel == KernelLevel::H_L0_SIMPLE) ? 0 :
                        (kernel == KernelLevel::H_L1_OPTIMIZED) ? 1 :
                        (kernel == KernelLevel::H_L2_SPECIALIZED) ? 2 : 3;
        file << "| L" << level_num
             << " | " << getKernelName(kernel)
             << " | " << getOptimizations(kernel) << " |";

        double sum = 0;
        int count = 0;
        for (const auto& ds : datasets) {
            for (const auto& r : all_results) {
                if (r.dataset == ds && r.kernel_name == getKernelName(kernel)) {
                    file << " " << std::fixed << std::setprecision(1) << r.throughput_gbps
                         << " (" << std::setprecision(2) << r.speedup_vs_baseline << "x) |";
                    sum += r.throughput_gbps;
                    count++;
                    break;
                }
            }
        }
        file << " " << std::fixed << std::setprecision(1) << (count > 0 ? sum/count : 0) << " |\n";
    }

    // Section 2: Vertical Layout (single unified kernel in v3.0)
    file << "\n## 2. Vertical (Vertical) Layout Performance\n\n";
    file << "**Note**: In v3.0, all DecompressMode values route to the same kernel\n";
    file << "(decompressInterleavedAllPartitions). There is only one optimization level.\n\n";
    file << "| Kernel | Optimizations |";
    for (const auto& ds : datasets) file << " " << ds << " |";
    file << " Avg |\n";
    file << "|--------|---------------|";
    for (size_t i = 0; i < datasets.size(); i++) file << "------|";
    file << "-----|\n";

    // V_OPTIMIZED (single level)
    file << "| V-Optimized | MiniVector(256) +Branchless +Interleaved |";
    double v_sum = 0;
    int v_count = 0;
    for (const auto& ds : datasets) {
        for (const auto& r : all_results) {
            if (r.dataset == ds && r.kernel_name == "V-Optimized") {
                file << " " << std::fixed << std::setprecision(1) << r.throughput_gbps << " |";
                v_sum += r.throughput_gbps;
                v_count++;
                break;
            }
        }
    }
    file << " " << std::fixed << std::setprecision(1) << (v_count > 0 ? v_sum/v_count : 0) << " |\n";

    // Section 3: Optimization Contribution
    file << "\n## 3. Optimization Contribution Analysis\n\n";
    file << "### Horizontal Layout\n\n";
    file << "| Transition | Technique Added | Avg Speedup |\n";
    file << "|------------|-----------------|-------------|\n";

    // Calculate average speedup for H-L0→L1
    double h_l0_to_l1_sum = 0;
    int h_l0_to_l1_count = 0;
    for (const auto& ds : datasets) {
        double l0_gbps = 0, l1_gbps = 0;
        for (const auto& r : all_results) {
            if (r.dataset == ds) {
                if (r.kernel_name == "H-L0-Simple") l0_gbps = r.throughput_gbps;
                if (r.kernel_name == "H-L1-Optimized") l1_gbps = r.throughput_gbps;
            }
        }
        if (l0_gbps > 0 && l1_gbps > 0) {
            h_l0_to_l1_sum += l1_gbps / l0_gbps;
            h_l0_to_l1_count++;
        }
    }
    file << "| L0 -> L1 | +SharedMem +WarpCoop | "
         << std::fixed << std::setprecision(2) << (h_l0_to_l1_count > 0 ? h_l0_to_l1_sum/h_l0_to_l1_count : 0) << "x |\n";

    // Calculate average speedup for H-L1→L2
    double h_l1_to_l2_sum = 0;
    int h_l1_to_l2_count = 0;
    for (const auto& ds : datasets) {
        double l1_gbps = 0, l2_gbps = 0;
        for (const auto& r : all_results) {
            if (r.dataset == ds) {
                if (r.kernel_name == "H-L1-Optimized") l1_gbps = r.throughput_gbps;
                if (r.kernel_name == "H-L2-Specialized") l2_gbps = r.throughput_gbps;
            }
        }
        if (l1_gbps > 0 && l2_gbps > 0) {
            h_l1_to_l2_sum += l2_gbps / l1_gbps;
            h_l1_to_l2_count++;
        }
    }
    file << "| L1 -> L2 | +TemplateSpecialized | "
         << std::fixed << std::setprecision(2) << (h_l1_to_l2_count > 0 ? h_l1_to_l2_sum/h_l1_to_l2_count : 0) << "x |\n";

    // Calculate average speedup for H-L2→L3
    double h_l2_to_l3_sum = 0;
    int h_l2_to_l3_count = 0;
    for (const auto& ds : datasets) {
        double l2_gbps = 0, l3_gbps = 0;
        for (const auto& r : all_results) {
            if (r.dataset == ds) {
                if (r.kernel_name == "H-L2-Specialized") l2_gbps = r.throughput_gbps;
                if (r.kernel_name == "H-L3-WarpOpt") l3_gbps = r.throughput_gbps;
            }
        }
        if (l2_gbps > 0 && l3_gbps > 0) {
            h_l2_to_l3_sum += l3_gbps / l2_gbps;
            h_l2_to_l3_count++;
        }
    }
    file << "| L2 -> L3 | +DoubleBuffer +FunnelShift +cp.async | "
         << std::fixed << std::setprecision(2) << (h_l2_to_l3_count > 0 ? h_l2_to_l3_sum/h_l2_to_l3_count : 0) << "x |\n";

    file << "\n### Vertical Layout\n\n";
    file << "**Note**: In v3.0, Vertical uses a single unified kernel (`decompressInterleavedAllPartitions`)\n";
    file << "for all decompression modes. The kernel combines:\n";
    file << "- MiniVector (256) lane-parallel processing\n";
    file << "- Branchless bit extraction\n";
    file << "- Interleaved memory access pattern\n\n";
    file << "No ablation progression is available since all modes route to the same optimized kernel.\n";

    // Section 4: Layout Comparison
    file << "\n## 4. Layout Comparison (Peak Performance)\n\n";
    file << "| Dataset | H-Peak (GB/s) | V-Peak (GB/s) | Ratio | Winner |\n";
    file << "|---------|---------------|---------------|-------|--------|\n";

    for (const auto& ds : datasets) {
        double h_peak = 0, v_peak = 0;
        for (const auto& r : all_results) {
            if (r.dataset == ds) {
                if (r.layout == "Horizontal" && r.throughput_gbps > h_peak) h_peak = r.throughput_gbps;
                if (r.layout == "Vertical" && r.throughput_gbps > v_peak) v_peak = r.throughput_gbps;
            }
        }
        std::string winner = (h_peak > v_peak) ? "Horizontal" : "Vertical";
        double ratio = (v_peak > 0) ? h_peak / v_peak : 0;
        file << "| " << ds << " | " << std::fixed << std::setprecision(1) << h_peak
             << " | " << v_peak << " | " << std::setprecision(2) << ratio << "x | " << winner << " |\n";
    }

    file.close();
    std::cout << "\nReport saved to: " << output_path << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "DECOMPRESSION ABLATION STUDY" << std::endl;
    std::cout << "========================================" << std::endl;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Partition Size: " << PARTITION_SIZE << std::endl;
    std::cout << "Warmup: " << WARMUP_RUNS << ", Timed: " << TIMED_RUNS << " (median)" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<AblationResult> all_results;

    // Dataset 1: fb (uint64)
    {
        std::string filename = DATA_DIR + "6-fb_200M_uint64.bin";
        std::vector<uint64_t> data;
        if (SOSDLoader::loadDataset(filename, data)) {
            auto results = runAblation(data, "fb");
            all_results.insert(all_results.end(), results.begin(), results.end());
        } else {
            std::cerr << "Failed to load: " << filename << std::endl;
        }
    }

    // Dataset 2: wiki (uint64)
    {
        std::string filename = DATA_DIR + "7-wiki_200M_uint64.bin";
        std::vector<uint64_t> data;
        if (SOSDLoader::loadDataset(filename, data)) {
            auto results = runAblation(data, "wiki");
            all_results.insert(all_results.end(), results.begin(), results.end());
        } else {
            std::cerr << "Failed to load: " << filename << std::endl;
        }
    }

    // Dataset 3: books (uint32)
    {
        std::string filename = DATA_DIR + "5-books_200M_uint32.bin";
        std::vector<uint32_t> data;
        if (SOSDLoader::loadDataset(filename, data)) {
            auto results = runAblation(data, "books");
            all_results.insert(all_results.end(), results.begin(), results.end());
        } else {
            std::cerr << "Failed to load: " << filename << std::endl;
        }
    }

    // Generate report
    std::string report_path = "papers/responses/R2/O3/ablation_report.md";
    generateReport(all_results, report_path);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Ablation Study Complete!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
