#include <compressors/compactionv5t/compaction-compressor.cuh>
#include <compressors/compactionv5t/compaction-decode.cuh>
#include <compressors/compactionv5t/compaction-defines.cuh>
#include <compressors/compactionv5t/compaction-encode.cuh>
#include <compressors/shared.cuh>

namespace gtsst::compressors {
    DecompressionConfiguration CompactionV5TCompressor::configure_decompression(size_t buf_size) {
        return DecompressionConfiguration{
            .input_buffer_size = buf_size,
            .decompression_buffer_size = buf_size * 3,
        };
    }

    DecompressionConfiguration CompactionV5TCompressor::configure_decompression_from_compress(
        size_t buf_size, CompressionConfiguration& config) {
        return DecompressionConfiguration{
            .input_buffer_size = buf_size,
            .decompression_buffer_size = config.input_buffer_size,
        };
    }

    // GPU decompression for v1 format
    static GTSSTStatus gpu_decompress_v1(const uint8_t* src, uint8_t* dst,
                                          const CompactionV5TFileHeader& file_header,
                                          size_t header_data_offset,
                                          const CompactionV5TBlockHeader* block_headers_host,
                                          const CompactionV5TSplitHeader* split_headers_host,
                                          const std::vector<fsst::DecodingTable>& decoders,
                                          size_t* out_size,
                                          float* kernel_time_us) {
        const uint32_t num_blocks = file_header.num_blocks;
        const uint32_t num_tables = file_header.num_tables;
        const uint32_t num_splits = file_header.num_splits;

        // Compute block data offsets (cumulative compressed sizes)
        std::vector<uint64_t> block_data_offsets(num_blocks);
        uint64_t offset = 0;
        for (uint32_t i = 0; i < num_blocks; i++) {
            block_data_offsets[i] = offset;
            offset += block_headers_host[i].compressed_size;
        }
        const uint64_t total_compressed_data = offset;

        // Prepare fsst_decoder_t array for GPU
        std::vector<fsst_decoder_t> decoder_array(num_tables);
        for (uint32_t i = 0; i < num_tables; i++) {
            decoder_array[i] = decoders[i].decoder;
        }

        // Allocate device memory
        fsst_decoder_t* d_decoders;
        CompactionV5TBlockHeader* d_block_headers;
        CompactionV5TSplitHeader* d_split_headers;
        uint64_t* d_block_data_offsets;
        uint8_t* d_compressed_data;

        safeCUDACall(cudaMalloc(&d_decoders, num_tables * sizeof(fsst_decoder_t)));
        safeCUDACall(cudaMalloc(&d_block_headers, num_blocks * sizeof(CompactionV5TBlockHeader)));
        safeCUDACall(cudaMalloc(&d_split_headers, num_blocks * sizeof(CompactionV5TSplitHeader)));
        safeCUDACall(cudaMalloc(&d_block_data_offsets, num_blocks * sizeof(uint64_t)));
        safeCUDACall(cudaMalloc(&d_compressed_data, total_compressed_data));

        // Allocate device output buffer
        const uint64_t output_size = file_header.uncompressed_size;
        uint8_t* d_output;
        safeCUDACall(cudaMalloc(&d_output, output_size));

        // Copy to device
        safeCUDACall(cudaMemcpy(d_decoders, decoder_array.data(),
                                num_tables * sizeof(fsst_decoder_t), cudaMemcpyHostToDevice));
        safeCUDACall(cudaMemcpy(d_block_headers, block_headers_host,
                                num_blocks * sizeof(CompactionV5TBlockHeader), cudaMemcpyHostToDevice));
        safeCUDACall(cudaMemcpy(d_split_headers, split_headers_host,
                                num_blocks * sizeof(CompactionV5TSplitHeader), cudaMemcpyHostToDevice));
        safeCUDACall(cudaMemcpy(d_block_data_offsets, block_data_offsets.data(),
                                num_blocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
        safeCUDACall(cudaMemcpy(d_compressed_data, src + header_data_offset,
                                total_compressed_data, cudaMemcpyHostToDevice));

        // Launch kernel with CUDA event timing
        cudaEvent_t start, stop;
        safeCUDACall(cudaEventCreate(&start));
        safeCUDACall(cudaEventCreate(&stop));
        safeCUDACall(cudaEventRecord(start));

        compactionv5t::gpu_decompress<<<num_blocks, num_splits>>>(
            d_decoders, d_block_headers, d_split_headers,
            d_compressed_data, d_block_data_offsets,
            d_output, num_blocks, compactionv5t::BLOCK_SIZE,
            compactionv5t::SUPER_BLOCK_SIZE, num_splits
        );

        safeCUDACall(cudaEventRecord(stop));
        safeCUDACall(cudaEventSynchronize(stop));

        float kernel_ms = 0.0f;
        safeCUDACall(cudaEventElapsedTime(&kernel_ms, start, stop));
        *kernel_time_us = kernel_ms * 1000.0f;

        safeCUDACall(cudaEventDestroy(start));
        safeCUDACall(cudaEventDestroy(stop));

        // Copy output back to host
        safeCUDACall(cudaMemcpy(dst, d_output, output_size, cudaMemcpyDeviceToHost));

        // Set output size
        *out_size = output_size;

        // Free device memory
        safeCUDACall(cudaFree(d_output));
        safeCUDACall(cudaFree(d_decoders));
        safeCUDACall(cudaFree(d_block_headers));
        safeCUDACall(cudaFree(d_split_headers));
        safeCUDACall(cudaFree(d_block_data_offsets));
        safeCUDACall(cudaFree(d_compressed_data));

        printf("decomp blocks (GPU): %d/%d, throughput: ", num_blocks, num_blocks);

        return gtsstSuccess;
    }

    GTSSTStatus CompactionV5TCompressor::decompress(const uint8_t* src, uint8_t* dst,
                                                    DecompressionConfiguration& config, size_t* out_size) {
        // Read file header
        CompactionV5TFileHeader file_header;
        memcpy(&file_header, src, sizeof(CompactionV5TFileHeader));

        std::vector<fsst::DecodingTable> decoders;
        size_t in = sizeof(CompactionV5TFileHeader);

        // First read all tables
        for (int table_id = 0; table_id < file_header.num_tables; table_id++) {
            fsst::DecodingTable dec{};
            const size_t table_len = dec.import_table(src + in);

            decoders.emplace_back(dec);
            in += table_len;
        }

        // Return error if table reading went wrong
        if (in - sizeof(CompactionV5TFileHeader) != file_header.table_size) {
            return gtsstErrorCorruptHeader;
        }

        // Also return corrupt header if number of subblocks doesn't match with the compiled constants
        if (file_header.num_sub_blocks != compactionv5t::SUB_BLOCKS) {
            return gtsstErrorCorruptHeader;
        }

        // Then read all block headers
        const size_t block_header_size = file_header.num_blocks * sizeof(CompactionV5TBlockHeader);
        auto* block_headers = (CompactionV5TBlockHeader*)malloc(block_header_size);
        memcpy(block_headers, src + in, block_header_size);
        in += block_header_size;

        // v1 format with splits: use GPU decompression
        if (file_header.format_version >= 1 && file_header.num_splits > 0) {
            const size_t split_headers_size = file_header.num_blocks * sizeof(CompactionV5TSplitHeader);
            auto* split_headers = (CompactionV5TSplitHeader*)malloc(split_headers_size);
            memcpy(split_headers, src + in, split_headers_size);
            in += split_headers_size;

            GTSSTStatus status = gpu_decompress_v1(src, dst, file_header, in,
                                                    block_headers, split_headers, decoders, out_size,
                                                    &config.kernel_time_us);
            free(block_headers);
            free(split_headers);
            return status;
        }

        // v0 format: CPU decompression (original path)
        const auto block_mem = (uint8_t*)malloc(compactionv5t::BLOCK_SIZE);

        // Then decode all blocks
        uint64_t out = 0;
        int num_encoded_blocks = 0;
        for (int block_id = 0; block_id < file_header.num_blocks; block_id++) {
            fsst::DecodingTable decoder = decoders[block_id / compactionv5t::SUPER_BLOCK_SIZE];
            const uint32_t block_size = block_headers[block_id].compressed_size;
            memcpy(block_mem, src + in, block_size);

            // If there are flushes, this is a compressed block. Otherwise plain-text, then just copy
            if (block_headers[block_id].flushes > 0) {
                const uint32_t block_out = seq_decompress(decoder, block_mem, dst + out, block_size);

                // If output size doesn't match, the block is corrupt
                if (block_out != block_headers[block_id].uncompressed_size) {
                    free(block_headers);
                    free(block_mem);
                    return gtsstErrorCorruptBlock;
                }

                out += block_out;
                num_encoded_blocks += 1;
            } else {
                memcpy(dst + out, block_mem, block_size);
                out += block_size;
            }

            in += block_size;
        }

        // Free header buffer
        free(block_headers);
        free(block_mem);

        // Do final check if we consumed all data, and produced the expected amount of data
        if (in != file_header.compressed_size || out != file_header.uncompressed_size) {
            return gtsstErrorCorrupt;
        }

        // Update output
        *out_size = out;

        // Print some stats for me :)
        printf("decomp blocks: %d/%d, throughput: ", num_encoded_blocks, file_header.num_blocks);

        return gtsstSuccess;
    }
} // namespace gtsst::compressors
