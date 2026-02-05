#include <compressors/compactionv5t/compaction-compressor.cuh>
#include <compressors/compactionv5t/compaction-defines.cuh>
#include <compressors/compactionv5t/compaction-encode.cuh>
#include <compressors/shared.cuh>
#include <fstream>
#include <thread>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace gtsst::compressors {

    __global__ void gpu_compute_split_headers(const uint8_t* __restrict__ compressed_data,
                                              const uint64_t* __restrict__ block_data_offsets,
                                              const CompactionV5TBlockHeader* __restrict__ block_headers,
                                              const fsst_decoder_t* __restrict__ decoders,
                                              CompactionV5TSplitHeader* __restrict__ split_headers,
                                              uint32_t num_blocks,
                                              uint32_t super_block_size,
                                              uint32_t num_splits) {
        const uint32_t block_id = blockIdx.x;
        if (block_id >= num_blocks) return;
        if (threadIdx.x != 0) return;

        const auto& bh = block_headers[block_id];
        auto* sh = &split_headers[block_id];

        // Uncompressed block: evenly divide the data.
        if (bh.flushes == 0) {
            for (uint32_t s = 0; s < num_splits; s++) {
                sh->compressed_offsets[s] =
                    (uint32_t)((uint64_t)s * bh.compressed_size / num_splits);
                sh->uncompressed_offsets[s] =
                    (uint32_t)((uint64_t)s * bh.uncompressed_size / num_splits);
            }
            return;
        }

        const uint32_t table_idx = block_id / super_block_size;
        const uint8_t* block_data = compressed_data + block_data_offsets[block_id];
        const uint32_t compressed_size = bh.compressed_size;

        const uint32_t target_split_size = compressed_size / num_splits;
        uint32_t compressed_pos = 0;
        uint32_t uncompressed_pos = 0;
        uint32_t next_split = 1;

        sh->compressed_offsets[0] = 0;
        sh->uncompressed_offsets[0] = 0;

        while (compressed_pos < compressed_size && next_split < num_splits) {
            const uint8_t code = block_data[compressed_pos];

            if (code == FSST_ESC) {
                compressed_pos += 2;
                uncompressed_pos += 1;
            } else {
                compressed_pos += 1;
                uncompressed_pos += decoders[table_idx].len[code];
            }

            if (compressed_pos >= next_split * target_split_size) {
                sh->compressed_offsets[next_split] = compressed_pos;
                sh->uncompressed_offsets[next_split] = uncompressed_pos;
                next_split++;
            }
        }

        while (next_split < num_splits) {
            sh->compressed_offsets[next_split] = compressed_size;
            sh->uncompressed_offsets[next_split] = uncompressed_pos;
            next_split++;
        }
    }

    void compute_split_boundaries(const uint8_t* block_data, const uint32_t compressed_size,
                                  const fsst::DecodingTable& decoder, const uint32_t num_splits,
                                  CompactionV5TSplitHeader& split_header) {
        const uint32_t target_split_size = compressed_size / num_splits;
        uint32_t compressed_pos = 0;
        uint32_t uncompressed_pos = 0;
        uint32_t next_split = 1;

        // First split always starts at offset 0
        split_header.compressed_offsets[0] = 0;
        split_header.uncompressed_offsets[0] = 0;

        while (compressed_pos < compressed_size && next_split < num_splits) {
            const uint8_t code = block_data[compressed_pos];

            if (code == fsst::Symbol::escape) {
                // Escape: 2 bytes of input → 1 byte of output
                compressed_pos += 2;
                uncompressed_pos += 1;
            } else {
                // Symbol code → len bytes of output
                compressed_pos += 1;
                uncompressed_pos += decoder.decoder.len[code];
            }

            // Check if we've crossed the next split boundary
            if (compressed_pos >= next_split * target_split_size && next_split < num_splits) {
                split_header.compressed_offsets[next_split] = compressed_pos;
                split_header.uncompressed_offsets[next_split] = uncompressed_pos;
                next_split++;
            }
        }

        // Fill remaining splits at the end (edge case: very small blocks)
        while (next_split < num_splits) {
            split_header.compressed_offsets[next_split] = compressed_size;
            split_header.uncompressed_offsets[next_split] = uncompressed_pos;
            next_split++;
        }
    }

    CompressionConfiguration CompactionV5TCompressor::configure_compression(const size_t buf_size) {
        const uint64_t num_blocks = buf_size / compactionv5t::BLOCK_SIZE;
        const uint64_t split_headers_overhead = sizeof(CompactionV5TSplitHeader) * num_blocks;
        return CompressionConfiguration{.input_buffer_size = buf_size,
                                        .compression_buffer_size = buf_size + 76800 + split_headers_overhead,
                                        .temp_buffer_size = buf_size,
                                        .min_alignment_input = compactionv5t::WORD_ALIGNMENT,
                                        .min_alignment_output = compactionv5t::WORD_ALIGNMENT,
                                        .min_alignment_temp = compactionv5t::TMP_WORD_ALIGNMENT,
                                        .must_pad_alignment = true,
                                        .block_size = compactionv5t::BLOCK_SIZE,
                                        .table_range = compactionv5t::BLOCK_SIZE * compactionv5t::SUPER_BLOCK_SIZE,
                                        .must_pad_block = true,

                                        .escape_symbol = fsst::Symbol::escape,
                                        .padding_symbol = fsst::Symbol::ignore,
                                        .padding_enabled = true,

                                        .device_buffers = true};
    }

    GTSSTStatus CompactionV5TCompressor::validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                                      CompressionConfiguration& config) {
        if (config.input_buffer_size > compactionv5t::BLOCK_SIZE * (size_t)0xFFFFFFFF) {
            return gtsstErrorTooBig;
        }

        if (config.block_size != compactionv5t::BLOCK_SIZE) {
            return gtsstErrorBadBlockSize;
        }

        if (config.min_alignment_input != compactionv5t::WORD_ALIGNMENT ||
            config.min_alignment_output != compactionv5t::WORD_ALIGNMENT ||
            config.min_alignment_temp != compactionv5t::TMP_WORD_ALIGNMENT) {
            return gtsstErrorBadBlockSize;
        }

        if ((uintptr_t)src % compactionv5t::WORD_ALIGNMENT != 0 ||
            (uintptr_t)dst % compactionv5t::WORD_ALIGNMENT != 0 ||
            (uintptr_t)tmp % compactionv5t::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorBadAlignment;
        }

        if (config.input_buffer_size % compactionv5t::BLOCK_SIZE != 0 ||
            config.temp_buffer_size % compactionv5t::TMP_OUT_BLOCK_SIZE != 0) {
            return gtsstErrorBlockAlignment;
        }

        if (config.input_buffer_size % compactionv5t::WORD_ALIGNMENT != 0 ||
            config.temp_buffer_size % compactionv5t::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorWordAlignment;
        }

        return gtsstSuccess;
    }

    GTSSTStatus CompactionV5TCompressor::compress(const uint8_t* src, uint8_t* dst,
                                                  uint8_t* tmp, CompressionConfiguration& config, size_t* out_size,
                                                  CompressionStatistics& stats) {
        if (const GTSSTStatus buffer_validation = validate_compression_buffers(src, dst, tmp, config);
            buffer_validation != gtsstSuccess) {
            return buffer_validation;
        }

        // Some bookkeeping
        const uint64_t number_of_blocks = config.input_buffer_size / compactionv5t::BLOCK_SIZE;
        const uint64_t number_of_tables = (number_of_blocks - 1) / compactionv5t::SUPER_BLOCK_SIZE + 1;
        const uint64_t metadata_mem_size = sizeof(compactionv5t::GCompactionMetadata) * number_of_tables;
        const uint64_t block_headers_mem_size = sizeof(CompactionV5TBlockHeader) * number_of_blocks;
        const uint64_t split_headers_mem_size = sizeof(CompactionV5TSplitHeader) * number_of_blocks;
        const uint64_t approx_header_mem_size =
            sizeof(CompactionV5TFileHeader) + number_of_tables * sizeof(GBaseHeader) + block_headers_mem_size + split_headers_mem_size;
        const uint64_t sample_data_mem_size = number_of_tables * FSST_SAMPLEMAXSZ;

        // Update the device queue for internal transpose launches
        cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, compactionv5t::CUDA_QUEUE_LEN);
        assert(number_of_blocks < compactionv5t::CUDA_QUEUE_LEN);
        // If there are too many blocks, we cannot compress this file in one go (and some margin)
        if (number_of_blocks > compactionv5t::CUDA_QUEUE_LEN - 10) {
            return gtsstErrorTooBig;
        }

        compactionv5t::GCompactionMetadata* metadata_host;
        GBaseHeader* table_headers_host;
        CompactionV5TBlockHeader* block_headers_host;
        uint8_t* sample_data_host;
        uint64_t* block_fix_list;
        safeCUDACall(cudaMallocHost(&metadata_host, metadata_mem_size));
        safeCUDACall(cudaMallocHost(&table_headers_host, sizeof(GBaseHeader) * number_of_tables));
        safeCUDACall(cudaMallocHost(&block_headers_host, block_headers_mem_size));
        safeCUDACall(cudaMallocHost(&sample_data_host, sample_data_mem_size));
        safeCUDACall(cudaMallocHost(&block_fix_list, number_of_blocks * sizeof(uint64_t)));
        block_fix_list[0] = 1; // Block 0 is special (the only valid number for block 0 is 0, while it's the opposite for every other block)

        // Some CUDA bookkeeping
        compactionv5t::GCompactionMetadata* metadata_gpu;
        CompactionV5TBlockHeader* block_headers_gpu;
        uint8_t* header_gpu;
        uint8_t* sample_data_gpu;

        // Allocate some CUDA buffers
        safeCUDACall(cudaMalloc(&metadata_gpu, metadata_mem_size));
        safeCUDACall(cudaMalloc(&block_headers_gpu, block_headers_mem_size));
        safeCUDACall(cudaMalloc(&header_gpu, approx_header_mem_size));
        safeCUDACall(cudaMalloc(&sample_data_gpu, sample_data_mem_size));

        // Phase 1: Symbol generation (CPU for now)
        const auto symbol_start = std::chrono::high_resolution_clock::now();

        // Sample data and copy it to the CPU
        gpu_sampling<<<number_of_tables, FSST_SAMPLELINE / 4>>>(
            src, sample_data_gpu, compactionv5t::BLOCK_SIZE * compactionv5t::SUPER_BLOCK_SIZE,
            config.input_buffer_size);
        safeCUDACall(cudaMemcpy(sample_data_host, sample_data_gpu, sample_data_mem_size, cudaMemcpyDeviceToHost));
        safeCUDACall(cudaDeviceSynchronize());

        // Then generate tables using the sampled data
        std::vector<std::thread> threads;
        threads.reserve(number_of_tables);
        for (uint32_t i = 0; i < number_of_tables; i++) {
            threads.emplace_back(gpu_create_metadata_with_samples<symbols::SmallSymbolMatchTableData>, i, metadata_host,
                                 table_headers_host, sample_data_host);
        }
        for (std::thread& t : threads) {
            t.join();
        }

        // Phase 2: Precomputation
        const auto precomputation_start = std::chrono::high_resolution_clock::now();
        // Copy metadata to GPU memory
        safeCUDACall(cudaMemcpyAsync(metadata_gpu, metadata_host, metadata_mem_size, cudaMemcpyHostToDevice));

        safeCUDACall(cudaPeekAtLastError());
        safeCUDACall(cudaDeviceSynchronize());

        // Phase 3: Encoding (GPU)
        const auto encoding_start = std::chrono::high_resolution_clock::now();

        // Run all blocks
        compactionv5t::gpu_compaction<<<number_of_blocks, compactionv5t::THREAD_COUNT>>>(
            metadata_gpu, block_headers_gpu, src, tmp, dst);
        safeCUDACall(cudaPeekAtLastError());
        safeCUDACall(cudaDeviceSynchronize());

        // Phase 4: Postprocessing (Partial CPU for now)
        const auto post_start = std::chrono::high_resolution_clock::now();

        // Copy comp headers & temp_dst to CPU
        safeCUDACall(cudaMemcpy(block_headers_host, block_headers_gpu, block_headers_mem_size, cudaMemcpyDeviceToHost));

        // Gather total output size
        uint64_t total_data_size = 0;
        for (uint32_t block_id = 0; block_id < number_of_blocks; block_id++) {
            total_data_size += block_headers_host[block_id].compressed_size;
        }

        // Calculate header positions (v1 format: includes split headers)
        CompactionV5TFileHeader file_header{
            {
                total_data_size + block_headers_mem_size + split_headers_mem_size + sizeof(CompactionV5TFileHeader),
                config.input_buffer_size,
                (uint32_t)number_of_tables,
                0,
                (uint32_t)number_of_blocks,
            },
            compactionv5t::SUB_BLOCKS,
            1, // format_version = 1 (with splits)
            (uint8_t)compactionv5t::NUM_SPLITS,
        };
        size_t header_size = sizeof(CompactionV5TFileHeader);

        // Copy tables
        for (int table_id = 0; table_id < number_of_tables; table_id++) {
            safeCUDACall(cudaMemcpyAsync(header_gpu + header_size, &table_headers_host[table_id],
                                         metadata_host[table_id].header_offset, cudaMemcpyHostToDevice));

            header_size += metadata_host[table_id].header_offset;
            file_header.table_size += metadata_host[table_id].header_offset;
        }

        // Copy block headers
        safeCUDACall(cudaMemcpyAsync(header_gpu + header_size, block_headers_host, block_headers_mem_size,
                                     cudaMemcpyHostToDevice));
        header_size += block_headers_mem_size;

        // Reserve space for split headers (will fill after stream compaction)
        const size_t split_headers_offset = header_size;
        header_size += split_headers_mem_size;

        // Copy file header (update compressed_size to include table_size and split_headers)
        file_header.compressed_size += file_header.table_size;
        safeCUDACall(
            cudaMemcpyAsync(header_gpu, &file_header, sizeof(CompactionV5TFileHeader), cudaMemcpyHostToDevice));

        // Then gather data
        uint64_t running_length = 0;
        uint64_t running_filtered_length = 0; // Also keep track of running_length after filtering (so running_length - count(0xFE))
        for (uint32_t block_id = 0; block_id < number_of_blocks; block_id++) {
            assert(block_headers_host[block_id].flushes <= compactionv5t::tile_out_len_words); // A block cannot overflow its buffer

            uint8_t* running_dst = tmp + running_length;
            uint16_t block_flushes = block_headers_host[block_id].flushes;
            size_t block_size = block_flushes * compactionv5t::THREAD_COUNT * sizeof(uint32_t);

            if (block_flushes > 0) {
                safeCUDACall(cudaMemcpyAsync(running_dst, dst + compactionv5t::TMP_OUT_BLOCK_SIZE * block_id, block_size,
                                             cudaMemcpyDeviceToDevice));
            } else {
                // Block couldn't be compressed, for now put in some filler data. Will fix later (cannot run original data through filtering, might contain 0xFE)
                block_size = compactionv5t::BLOCK_SIZE;
                safeCUDACall(cudaMemsetAsync(running_dst, 0x42, block_size)); // TODO: I don't like this, probably nicer way somehow
                block_fix_list[block_id] = running_filtered_length; // Keep track of where we will need to place original data for this block
            }

            running_length += block_size;
            running_filtered_length += block_headers_host[block_id].compressed_size;
        }

        // Then do stream compaction on the actual data
        const thrust::device_ptr<uint8_t> thrust_gpu_in = thrust::device_pointer_cast(tmp);
        const thrust::device_ptr<uint8_t> thrust_gpu_out = thrust::device_pointer_cast(dst + header_size);
        const thrust::device_ptr<uint8_t> thrust_new_end =
            copy_if(thrust::device, thrust_gpu_in, thrust_gpu_in + running_length, thrust_gpu_out, is_not_ignore());
        const size_t thrust_out_size = thrust_new_end - thrust_gpu_out;
        const size_t out = thrust_out_size + header_size;

        // Copy header to dst
        safeCUDACall(cudaMemcpy(dst, header_gpu, split_headers_offset, cudaMemcpyDeviceToDevice));

        // And now fix blocks that couldn't be encoded
        for (uint32_t block_id = 0; block_id < number_of_blocks; block_id++) {
            uint64_t fix_location = block_fix_list[block_id];
            bool need_to_fix = (block_id == 0) == (fix_location == 0);

            if (need_to_fix) {
                const uint8_t* fix_src = src + (size_t) compactionv5t::BLOCK_SIZE * block_id;
                uint8_t* fix_dst = dst + header_size + fix_location;
                safeCUDACall(cudaMemcpy(fix_dst, fix_src, compactionv5t::BLOCK_SIZE, cudaMemcpyDeviceToDevice));
            }
        }

        // Phase 4b: Compute split boundaries for GPU decompression
        // Build decoders for each table (host) and copy to GPU for split header generation
        std::vector<fsst::DecodingTable> decoders(number_of_tables);
        for (uint32_t table_id = 0; table_id < number_of_tables; table_id++) {
            decoders[table_id].import_table(table_headers_host[table_id].decoding_table);
        }

        std::vector<fsst_decoder_t> decoder_array(number_of_tables);
        for (uint32_t table_id = 0; table_id < number_of_tables; table_id++) {
            decoder_array[table_id] = decoders[table_id].decoder;
        }

        // Compute block offsets (host) and copy to GPU
        std::vector<uint64_t> block_data_offsets(number_of_blocks);
        uint64_t block_data_offset = 0;
        for (uint32_t block_id = 0; block_id < number_of_blocks; block_id++) {
            block_data_offsets[block_id] = block_data_offset;
            block_data_offset += block_headers_host[block_id].compressed_size;
        }

        fsst_decoder_t* d_decoders = nullptr;
        uint64_t* d_block_data_offsets = nullptr;
        safeCUDACall(cudaMalloc(&d_decoders, number_of_tables * sizeof(fsst_decoder_t)));
        safeCUDACall(cudaMalloc(&d_block_data_offsets, number_of_blocks * sizeof(uint64_t)));
        safeCUDACall(cudaMemcpy(d_decoders, decoder_array.data(), number_of_tables * sizeof(fsst_decoder_t),
                                cudaMemcpyHostToDevice));
        safeCUDACall(cudaMemcpy(d_block_data_offsets, block_data_offsets.data(), number_of_blocks * sizeof(uint64_t),
                                cudaMemcpyHostToDevice));

        CompactionV5TSplitHeader* split_headers_tmp = nullptr;
        safeCUDACall(cudaMalloc(&split_headers_tmp, split_headers_mem_size));
        gpu_compute_split_headers<<<number_of_blocks, 1>>>(
            dst + header_size, d_block_data_offsets, block_headers_gpu, d_decoders, split_headers_tmp,
            number_of_blocks, compactionv5t::SUPER_BLOCK_SIZE, compactionv5t::NUM_SPLITS);
        safeCUDACall(cudaPeekAtLastError());
        safeCUDACall(cudaDeviceSynchronize());

        // Copy to final (possibly unaligned) destination in the output buffer.
        safeCUDACall(cudaMemcpy(dst + split_headers_offset, split_headers_tmp, split_headers_mem_size,
                                cudaMemcpyDeviceToDevice));
        safeCUDACall(cudaFree(split_headers_tmp));

        safeCUDACall(cudaFree(d_decoders));
        safeCUDACall(cudaFree(d_block_data_offsets));

        // Finally, free buffers
        safeCUDACall(cudaFreeHost(metadata_host));
        safeCUDACall(cudaFreeHost(table_headers_host));
        safeCUDACall(cudaFreeHost(block_headers_host));
        safeCUDACall(cudaFreeHost(sample_data_host));
        safeCUDACall(cudaFreeHost(block_fix_list));

        // And free cuda buffers
        safeCUDACall(cudaFree(metadata_gpu));
        safeCUDACall(cudaFree(block_headers_gpu));
        safeCUDACall(cudaFree(header_gpu));
        safeCUDACall(cudaFree(sample_data_gpu));

        // Check and update output size
        assert(file_header.compressed_size - sizeof(CompactionV5TFileHeader) - file_header.table_size -
                   block_headers_mem_size - split_headers_mem_size ==
               total_data_size);
        assert(thrust_out_size == total_data_size);
        *out_size = out;

        // Update statistics
        stats.table_generation =
            std::chrono::duration_cast<std::chrono::microseconds>(precomputation_start - symbol_start);
        stats.precomputation =
            std::chrono::duration_cast<std::chrono::microseconds>(encoding_start - precomputation_start);
        stats.encoding = std::chrono::duration_cast<std::chrono::microseconds>(post_start - encoding_start);
        stats.postprocessing = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - post_start);

        return gtsstSuccess;
    }
} // namespace gtsst::compressors
