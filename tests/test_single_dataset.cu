/**
 * L3 Single Dataset Comprehensive Test
 *
 * Tests a single dataset with all partitioning strategies and decompression modes.
 * Generates detailed report.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <sstream>

#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"
#include "../src/kernels/compression/cost_optimal_partitioner.cuh"

using namespace std;

// Partitioning strategies
enum class PartitionStrategy {
    FIXED_128,
    FIXED_256,
    FIXED_512,
    FIXED_1024,
    FIXED_2048,
    FIXED_4096,
    FIXED_8192,
    VARIANCE_ADAPTIVE,
    COST_OPTIMAL_BALANCED
};

const char* strategyName(PartitionStrategy s) {
    switch (s) {
        case PartitionStrategy::FIXED_128: return "FIXED_128";
        case PartitionStrategy::FIXED_256: return "FIXED_256";
        case PartitionStrategy::FIXED_512: return "FIXED_512";
        case PartitionStrategy::FIXED_1024: return "FIXED_1024";
        case PartitionStrategy::FIXED_2048: return "FIXED_2048";
        case PartitionStrategy::FIXED_4096: return "FIXED_4096";
        case PartitionStrategy::FIXED_8192: return "FIXED_8192";
        case PartitionStrategy::VARIANCE_ADAPTIVE: return "VARIANCE_ADAPTIVE";
        case PartitionStrategy::COST_OPTIMAL_BALANCED: return "COST_OPTIMAL_BALANCED";
        default: return "UNKNOWN";
    }
}

const char* modeName(DecompressMode m) {
    switch (m) {
        case DecompressMode::AUTO: return "AUTO";
        case DecompressMode::SEQUENTIAL: return "SEQUENTIAL";
        case DecompressMode::INTERLEAVED: return "INTERLEAVED";
        case DecompressMode::BRANCHLESS: return "BRANCHLESS";
        default: return "UNKNOWN";
    }
}

template<typename T>
vector<PartitionInfo> createPartitions(const vector<T>& data, PartitionStrategy strategy) {
    vector<PartitionInfo> partitions;
    int n = data.size();

    int fixed_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    int strategy_idx = static_cast<int>(strategy);

    if (strategy_idx <= 6) {  // Fixed size strategies
        int ps = fixed_sizes[strategy_idx];
        int num_parts = (n + ps - 1) / ps;
        for (int p = 0; p < num_parts; p++) {
            PartitionInfo info;
            info.start_idx = p * ps;
            info.end_idx = min((p + 1) * ps, n);
            partitions.push_back(info);
        }
    } else if (strategy == PartitionStrategy::VARIANCE_ADAPTIVE) {
        // Variance-based adaptive partitioning
        int min_size = 512, max_size = 8192;
        double var_threshold = 1e20;

        int start = 0;
        while (start < n) {
            int best_end = min(start + min_size, n);

            // Try to extend partition
            for (int end = start + min_size; end <= min(start + max_size, n); end += 256) {
                double sum = 0, sum_sq = 0;
                for (int i = start; i < end; i++) {
                    double v = static_cast<double>(data[i]);
                    sum += v;
                    sum_sq += v * v;
                }
                double mean = sum / (end - start);
                double var = sum_sq / (end - start) - mean * mean;

                if (var < var_threshold) {
                    best_end = end;
                } else {
                    break;
                }
            }

            PartitionInfo info;
            info.start_idx = start;
            info.end_idx = best_end;
            partitions.push_back(info);
            start = best_end;
        }
    } else {  // COST_OPTIMAL_BALANCED
        T* d_data;
        cudaMalloc(&d_data, n * sizeof(T));
        cudaMemcpy(d_data, data.data(), n * sizeof(T), cudaMemcpyHostToDevice);

        partitions = cost_optimal_partitioner::partitionCostOptimal<T>(d_data, n);

        cudaFree(d_data);
    }

    return partitions;
}

struct TestResult {
    string strategy;
    string mode;
    int num_partitions;
    double avg_partition_size;
    double avg_delta_bits;
    double compression_ratio;
    double compress_time_ms;
    double decompress_time_ms;
    double decompress_throughput_gbps;
    double random_access_time_ms;
    bool decompress_correct;
    bool random_access_correct;
    int num_errors;
    // Model distribution
    double constant_pct;
    double linear_pct;
    double poly2_pct;
    double poly3_pct;
    double for_bp_pct;
};

template<typename T>
TestResult runTest(const vector<T>& data, PartitionStrategy strategy, DecompressMode mode) {
    TestResult result;
    result.strategy = strategyName(strategy);
    result.mode = modeName(mode);

    int n = data.size();
    double data_size_mb = n * sizeof(T) / (1024.0 * 1024.0);

    // Create partitions
    auto partitions = createPartitions(data, strategy);
    result.num_partitions = partitions.size();
    result.avg_partition_size = static_cast<double>(n) / partitions.size();

    // Configure encoder
    VerticalConfig config;
    config.enable_interleaved = true;
    config.enable_adaptive_selection = true;
    config.interleaved_threshold = 512;

    // Compress
    auto t1 = chrono::high_resolution_clock::now();
    auto compressed = Vertical_encoder::encodeVertical<T>(data, partitions, config);
    cudaDeviceSynchronize();
    auto t2 = chrono::high_resolution_clock::now();
    result.compress_time_ms = chrono::duration<double, milli>(t2 - t1).count();

    // Calculate compression ratio
    double compressed_size_mb = compressed.sequential_delta_words * 4.0 / (1024.0 * 1024.0);
    compressed_size_mb += partitions.size() * (4 + 4 + 4 + 32 + 4 + 8) / (1024.0 * 1024.0);  // metadata
    result.compression_ratio = data_size_mb / compressed_size_mb;

    // Calculate model distribution and avg delta bits
    result.constant_pct = result.linear_pct = result.poly2_pct = result.poly3_pct = result.for_bp_pct = 0;
    double total_bits = 0;
    for (const auto& p : partitions) {
        total_bits += p.delta_bits;
        switch (p.model_type) {
            case MODEL_CONSTANT: result.constant_pct++; break;
            case MODEL_LINEAR: result.linear_pct++; break;
            case MODEL_POLYNOMIAL2: result.poly2_pct++; break;
            case MODEL_POLYNOMIAL3: result.poly3_pct++; break;
            case MODEL_FOR_BITPACK: result.for_bp_pct++; break;
        }
    }
    result.avg_delta_bits = total_bits / partitions.size();
    result.constant_pct = 100.0 * result.constant_pct / partitions.size();
    result.linear_pct = 100.0 * result.linear_pct / partitions.size();
    result.poly2_pct = 100.0 * result.poly2_pct / partitions.size();
    result.poly3_pct = 100.0 * result.poly3_pct / partitions.size();
    result.for_bp_pct = 100.0 * result.for_bp_pct / partitions.size();

    // Decompress
    T* d_output;
    cudaMalloc(&d_output, n * sizeof(T));

    // Warmup
    Vertical_decoder::decompressAll<T>(compressed, d_output, mode);
    cudaDeviceSynchronize();

    // Timed run
    t1 = chrono::high_resolution_clock::now();
    Vertical_decoder::decompressAll<T>(compressed, d_output, mode);
    cudaDeviceSynchronize();
    t2 = chrono::high_resolution_clock::now();
    result.decompress_time_ms = chrono::duration<double, milli>(t2 - t1).count();
    result.decompress_throughput_gbps = data_size_mb / 1024.0 / (result.decompress_time_ms / 1000.0);

    // Verify
    vector<T> output(n);
    cudaMemcpy(output.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    result.decompress_correct = true;
    result.num_errors = 0;
    for (int i = 0; i < n; i++) {
        if (output[i] != data[i]) {
            result.decompress_correct = false;
            result.num_errors++;
            if (result.num_errors <= 5) {
                cerr << "  Mismatch at " << i << ": expected " << data[i] << ", got " << output[i] << endl;
            }
        }
    }

    // Random access test
    const int NUM_QUERIES = 10000;
    vector<int> queries(NUM_QUERIES);
    srand(42);
    for (int i = 0; i < NUM_QUERIES; i++) {
        queries[i] = rand() % n;
    }

    int* d_queries;
    T* d_ra_output;
    cudaMalloc(&d_queries, NUM_QUERIES * sizeof(int));
    cudaMalloc(&d_ra_output, NUM_QUERIES * sizeof(T));
    cudaMemcpy(d_queries, queries.data(), NUM_QUERIES * sizeof(int), cudaMemcpyHostToDevice);

    t1 = chrono::high_resolution_clock::now();
    Vertical_decoder::decompressIndices<T>(compressed, d_queries, NUM_QUERIES, d_ra_output);
    cudaDeviceSynchronize();
    t2 = chrono::high_resolution_clock::now();
    result.random_access_time_ms = chrono::duration<double, milli>(t2 - t1).count();

    vector<T> ra_output(NUM_QUERIES);
    cudaMemcpy(ra_output.data(), d_ra_output, NUM_QUERIES * sizeof(T), cudaMemcpyDeviceToHost);

    result.random_access_correct = true;
    for (int i = 0; i < NUM_QUERIES; i++) {
        if (ra_output[i] != data[queries[i]]) {
            result.random_access_correct = false;
            break;
        }
    }

    // Cleanup
    cudaFree(d_output);
    cudaFree(d_queries);
    cudaFree(d_ra_output);
    Vertical_encoder::freeCompressedData(compressed);

    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <data_file> <output_dir>" << endl;
        return 1;
    }

    string data_file = argv[1];
    string output_dir = argv[2];

    // Determine data type from filename
    bool is_uint64 = data_file.find("uint64") != string::npos ||
                     data_file.find("int64") != string::npos;
    bool is_int32 = data_file.find("int32") != string::npos;
    bool is_uint32 = data_file.find("uint32") != string::npos && !is_uint64;

    // Default to uint64 if not specified
    if (!is_uint64 && !is_int32 && !is_uint32) {
        is_uint64 = true;
    }

    cout << "========================================" << endl;
    cout << "  L3 Single Dataset Comprehensive Test" << endl;
    cout << "========================================" << endl;
    cout << "Data file: " << data_file << endl;
    cout << "Output dir: " << output_dir << endl;
    cout << "Data type: " << (is_uint64 ? "uint64" : (is_int32 ? "int32" : "uint32")) << endl;

    // Read data
    ifstream file(data_file, ios::binary);
    if (!file) {
        cerr << "Cannot open file: " << data_file << endl;
        return 1;
    }

    file.seekg(0, ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);

    vector<TestResult> results;

    if (is_uint64) {
        size_t n = file_size / sizeof(uint64_t);
        vector<uint64_t> data(n);
        file.read(reinterpret_cast<char*>(data.data()), n * sizeof(uint64_t));

        cout << "Elements: " << n << endl;
        cout << "Size: " << (n * sizeof(uint64_t) / (1024.0 * 1024.0)) << " MB" << endl;
        cout << "========================================" << endl;

        // Test all combinations
        vector<PartitionStrategy> strategies = {
            PartitionStrategy::FIXED_128,
            PartitionStrategy::FIXED_256,
            PartitionStrategy::FIXED_512,
            PartitionStrategy::FIXED_1024,
            PartitionStrategy::FIXED_2048,
            PartitionStrategy::FIXED_4096,
            PartitionStrategy::FIXED_8192,
            PartitionStrategy::VARIANCE_ADAPTIVE,
            PartitionStrategy::COST_OPTIMAL_BALANCED
        };

        vector<DecompressMode> modes = {
            DecompressMode::AUTO,
            DecompressMode::SEQUENTIAL,
            DecompressMode::INTERLEAVED,
            DecompressMode::BRANCHLESS
        };

        int total_tests = strategies.size() * modes.size();
        int test_num = 0;

        for (auto strategy : strategies) {
            for (auto mode : modes) {
                test_num++;
                cout << "[" << test_num << "/" << total_tests << "] "
                     << strategyName(strategy) << " + " << modeName(mode) << "... " << flush;

                auto result = runTest<uint64_t>(data, strategy, mode);
                results.push_back(result);

                if (result.decompress_correct && result.random_access_correct) {
                    cout << "OK (ratio=" << fixed << setprecision(2) << result.compression_ratio
                         << "x, " << result.decompress_throughput_gbps << " GB/s)" << endl;
                } else {
                    cout << "FAIL (" << result.num_errors << " errors)" << endl;
                }
            }
        }
    }

    // Generate report
    cout << endl << "Generating report..." << endl;

    // Summary report
    ofstream summary(output_dir + "/summary_report.md");
    summary << "# L3 Single Dataset Test Report" << endl << endl;
    summary << "**Dataset**: " << data_file << endl;
    summary << "**Generated**: " << __DATE__ << " " << __TIME__ << endl << endl;

    summary << "## Overall Statistics" << endl << endl;

    int total = results.size();
    int decomp_correct = 0, ra_correct = 0;
    double avg_ratio = 0, avg_throughput = 0;

    for (const auto& r : results) {
        if (r.decompress_correct) decomp_correct++;
        if (r.random_access_correct) ra_correct++;
        avg_ratio += r.compression_ratio;
        avg_throughput += r.decompress_throughput_gbps;
    }
    avg_ratio /= total;
    avg_throughput /= total;

    summary << "| Metric | Value |" << endl;
    summary << "|--------|-------|" << endl;
    summary << "| Total Tests | " << total << " |" << endl;
    summary << "| Decompression Correct | " << decomp_correct << " / " << total << " |" << endl;
    summary << "| Random Access Correct | " << ra_correct << " / " << total << " |" << endl;
    summary << "| Avg Compression Ratio | " << fixed << setprecision(2) << avg_ratio << "x |" << endl;
    summary << "| Avg Decompress Throughput | " << avg_throughput << " GB/s |" << endl;
    summary << endl;

    // Detailed CSV
    ofstream csv(output_dir + "/detailed_results.csv");
    csv << "strategy,mode,num_partitions,avg_partition_size,avg_delta_bits,compression_ratio,"
        << "compress_time_ms,decompress_time_ms,decompress_throughput_gbps,random_access_time_ms,"
        << "decompress_correct,random_access_correct,constant_pct,linear_pct,poly2_pct,poly3_pct,for_bp_pct" << endl;

    for (const auto& r : results) {
        csv << r.strategy << "," << r.mode << "," << r.num_partitions << ","
            << r.avg_partition_size << "," << r.avg_delta_bits << "," << r.compression_ratio << ","
            << r.compress_time_ms << "," << r.decompress_time_ms << "," << r.decompress_throughput_gbps << ","
            << r.random_access_time_ms << "," << (r.decompress_correct ? "true" : "false") << ","
            << (r.random_access_correct ? "true" : "false") << ","
            << r.constant_pct << "," << r.linear_pct << "," << r.poly2_pct << ","
            << r.poly3_pct << "," << r.for_bp_pct << endl;
    }

    // Best configuration by compression ratio
    summary << "## Best Configurations" << endl << endl;
    summary << "### By Compression Ratio" << endl << endl;
    summary << "| Rank | Strategy | Mode | Ratio | Throughput |" << endl;
    summary << "|------|----------|------|-------|------------|" << endl;

    vector<int> indices(results.size());
    for (int i = 0; i < results.size(); i++) indices[i] = i;
    sort(indices.begin(), indices.end(), [&](int a, int b) {
        return results[a].compression_ratio > results[b].compression_ratio;
    });

    for (int i = 0; i < min(10, (int)results.size()); i++) {
        const auto& r = results[indices[i]];
        if (r.decompress_correct) {
            summary << "| " << (i+1) << " | " << r.strategy << " | " << r.mode
                    << " | " << r.compression_ratio << "x | " << r.decompress_throughput_gbps << " GB/s |" << endl;
        }
    }

    summary << endl << "### By Throughput" << endl << endl;
    summary << "| Rank | Strategy | Mode | Throughput | Ratio |" << endl;
    summary << "|------|----------|------|------------|-------|" << endl;

    sort(indices.begin(), indices.end(), [&](int a, int b) {
        return results[a].decompress_throughput_gbps > results[b].decompress_throughput_gbps;
    });

    for (int i = 0; i < min(10, (int)results.size()); i++) {
        const auto& r = results[indices[i]];
        if (r.decompress_correct) {
            summary << "| " << (i+1) << " | " << r.strategy << " | " << r.mode
                    << " | " << r.decompress_throughput_gbps << " GB/s | " << r.compression_ratio << "x |" << endl;
        }
    }

    // Strategy comparison
    summary << endl << "## Strategy Comparison (AUTO mode)" << endl << endl;
    summary << "| Strategy | Ratio | Throughput | Avg Delta Bits | Model Distribution |" << endl;
    summary << "|----------|-------|------------|----------------|-------------------|" << endl;

    for (const auto& r : results) {
        if (r.mode == "AUTO") {
            stringstream model_dist;
            if (r.linear_pct > 0) model_dist << "L:" << fixed << setprecision(0) << r.linear_pct << "% ";
            if (r.poly2_pct > 0) model_dist << "P2:" << r.poly2_pct << "% ";
            if (r.poly3_pct > 0) model_dist << "P3:" << r.poly3_pct << "% ";
            if (r.for_bp_pct > 0) model_dist << "FOR:" << r.for_bp_pct << "% ";

            summary << "| " << r.strategy << " | " << r.compression_ratio << "x | "
                    << r.decompress_throughput_gbps << " GB/s | " << r.avg_delta_bits
                    << " | " << model_dist.str() << " |" << endl;
        }
    }

    summary.close();
    csv.close();

    cout << "Report saved to: " << output_dir << endl;
    cout << "========================================" << endl;

    return 0;
}
