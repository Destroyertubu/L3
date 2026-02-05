#ifndef GTSST_BENCH_HPP
#define GTSST_BENCH_HPP
#include <chrono>
#include <gtsst/gtsst.cuh>
#include <string>
#include <vector>

namespace gtsst::bench {
    struct CompressionThroughputs {
        double table_generation_throughput;
        double precomputation_throughput;
        double encoding_throughput;
        double postprocessing_throughput;
    };

    struct CompressionStats {
        std::chrono::microseconds compression_duration;
        double compression_throughput;
        CompressionStatistics internal_stats;
        size_t compress_len{};
        size_t original_len{};

        std::chrono::microseconds decompression_duration;
        double decompression_throughput;
        size_t decompress_len{};

        float ratio;
    };

    struct AggregatedCompressionStats {
        std::vector<CompressionStats> stats;

        // Same as in CompressionStats, but then mean
        std::chrono::microseconds compression_duration;
        std::chrono::microseconds decompression_duration;
        double compression_throughput;
        double decompression_throughput;

        float ratio;

        CompressionStatistics internal_stats;
        CompressionThroughputs internal_throughputs;
    };

    size_t read_file_size(const char* filename);
    uint8_t* read_file(const char* filename, size_t data_to_read, size_t buffer_size, bool silent = true);
    void write_file(const char* filename, const char* data, size_t len);


    CompressionStats compress_single(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                     CompressionConfiguration& compression_configuration,
                                     CompressionManager& compression_manager);

    AggregatedCompressionStats compress_repeat(const uint8_t* src, uint8_t* dst,
                                               uint8_t* tmp, CompressionConfiguration& compression_configuration,
                                               CompressionManager& compression_manager, int iterations = 1);

    bool decompress_single(CompressionStats& stats, const uint8_t* src, uint8_t* dst, const uint8_t* original_data,
                           DecompressionConfiguration& decompression_configuration,
                           CompressionManager& compression_manager, bool strict_checking, int iterations = 1);

    bool decompress_all(AggregatedCompressionStats& aggregated_stats, const uint8_t* src, uint8_t* tmp,
                        const uint8_t* original_data, DecompressionConfiguration& decompression_configuration,
                        CompressionManager& compression_manager, bool strict_checking, int iterations = 1);

    bool full_cycle(const char* filename, int compression_iterations, int decompression_iterations,
                    CompressionManager& compression_manager, bool print_csv, bool strict_checking);

    bool full_cycle_directory(const std::vector<std::string>& directories, bool use_dir, int compression_iterations,
                              int decompression_iterations, CompressionManager& compression_manager, bool print_csv,
                              bool strict_checking);

} // namespace gtsst::bench

#endif // GTSST_BENCH_HPP
