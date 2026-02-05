#include <bench/gtsst-bench.cuh>
#include <compressors/compactionv5t/compaction-compressor.cuh>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    const bool use_override = argc >= 2;

    // Set directories to use
    std::vector<std::string> directories = {
        "../bench-data/",
    };

    if (use_override) {
        directories.clear();

        for (int i = 1; i < argc; i++) {
            directories.emplace_back(argv[i]);
        }
    }

    // Active compressor (see thesis repo for others)
    gtsst::compressors::CompactionV5TCompressor compressor;

    // Set bench settings
    constexpr int compression_iterations = 10;
    constexpr int decompression_iterations = 10;
    constexpr bool strict_checking =
        true; // Exit program when a single decompression mismatch occurs, otherwise only report it

    // Run benchmark (use_dir=true to process all files in the directory)
    const bool match = gtsst::bench::full_cycle_directory(directories, true, compression_iterations,
                                                          decompression_iterations, compressor, false, strict_checking);
    if (!match) {
        std::cerr << "Cycle data mismatch." << std::endl;
        return 1;
    }

    return 0;
}
