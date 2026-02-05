#ifndef GTSST_HPP
#define GTSST_HPP
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define checkedCUDACall(ans)                                                                                           \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stdout, "CUDA error: %s (%s) %s %d\n", cudaGetErrorString(code), cudaGetErrorName(code), file, line);
        if (abort)
            exit(code);
    }
}

namespace gtsst
{
    struct CompressionStatistics
    {
        std::chrono::microseconds table_generation;
        std::chrono::microseconds precomputation;
        std::chrono::microseconds encoding;
        std::chrono::microseconds postprocessing;
    };

    struct CompressionConfiguration
    {
        size_t input_buffer_size;
        size_t compression_buffer_size;
        size_t temp_buffer_size;

        size_t min_alignment_input;
        size_t min_alignment_output;
        size_t min_alignment_temp;
        bool must_pad_alignment;

        size_t block_size;
        size_t table_range;
        bool must_pad_block;

        uint8_t escape_symbol;
        uint8_t padding_symbol;
        bool padding_enabled;

        bool device_buffers;
    };

    struct DecompressionConfiguration
    {
        size_t input_buffer_size;
        size_t decompression_buffer_size;

        bool device_buffers;

        // Output: kernel-only execution time in microseconds (set by decompress)
        float kernel_time_us = 0.0f;
    };

    enum GTSSTStatus
    {
        gtsstSuccess,
        gtsstErrorBadBlockSize,
        gtsstErrorBadAlignment,
        gtsstErrorAlignment,
        gtsstErrorBlockAlignment,
        gtsstErrorWordAlignment,
        gtsstErrorCudaError,
        gtsstErrorCorruptHeader,
        gtsstErrorCorruptBlock,
        gtsstErrorCorrupt,
        gtsstErrorInternal,
        gtsstErrorTooBig,
    };

    struct CompressionManager
    {
        virtual ~CompressionManager() = default;
        virtual CompressionConfiguration configure_compression(size_t buf_size) = 0;
        virtual GTSSTStatus compress(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                     CompressionConfiguration& config, size_t* out_size,
                                     CompressionStatistics& stats) = 0;

        virtual DecompressionConfiguration configure_decompression(size_t buf_size) = 0;

        virtual DecompressionConfiguration configure_decompression_from_compress(
            const size_t buf_size, CompressionConfiguration& config)
        {
            return DecompressionConfiguration{
                .input_buffer_size = buf_size,
                .decompression_buffer_size = config.input_buffer_size,
            };
        }

        virtual GTSSTStatus decompress(const uint8_t* src, uint8_t* dst, DecompressionConfiguration& config,
                                       size_t* out_size) = 0;

    private:
        virtual GTSSTStatus validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                         CompressionConfiguration& config) = 0;
    };
} // namespace gtsst

#endif // GTSST_HPP
