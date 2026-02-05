#include <cassert>
#include <compressors/compactionv5t/compaction-defines.cuh>
#include <compressors/compactionv5t/compaction-encode.cuh>
#include <compressors/shared.cuh>
#include <cub/cub.cuh>
#include <fsst/fsst-lib.cuh>
#include <gtsst/gtsst-symbols.cuh>

namespace gtsst::compressors::compactionv5t {
    __device__ void pack_results_local(uint32_t result[tile_out_word_buf_size][THREAD_COUNT], const uint32_t offset,
                                       const uint32_t val) {
        assert(offset < n_out_symbols_buf);

        const uint32_t shift = (offset & 3) * 8;
        const uint8_t res = offset / 4;
        const uint32_t val_mask = val << shift;
        const uint32_t clean_mask = ~(0xFFU << shift);

        const uint32_t current = result[res][threadIdx.x];

        // If the value to be overwritten is not 0xFE (padding), there is an overflow.
        assert((current & ~clean_mask) >> shift == 0xFE);

        result[res][threadIdx.x] = current & clean_mask | val_mask;
    }

    __device__ void shift_registers_with_base(uint32_t load[tile_out_word_buf_size][THREAD_COUNT],
                                              const uint8_t bytes_used, uint8_t* first_block_offset,
                                              uint8_t* block_offset, uint32_t* first, uint32_t* second,
                                              uint32_t* third) {
        // Shift all blocks if needed
        const uint8_t shift = (*first_block_offset + bytes_used) / sizeof(uint32_t);
        for (int i = 0; i < shift; i++) {
            *block_offset = min(n_regs_per_chunk - 1, *block_offset + 1);
            *first = *second;
            *second = *third;
            *third = load[*block_offset][threadIdx.x];
        }

        *first_block_offset = (*first_block_offset + bytes_used) % sizeof(uint32_t);
    }

    __device__ inline bool can_flush(const uint8_t active_out_block, const uint8_t current_out) {
        /*
         * This function checks if there is any data to be flushed. This is the case in the following cases:
         * 1: The current active block is not 0xFEFEFEFE (if flushed, would be in reset state)
         * 2: When the current_out is not pointing to the start of active_out_block
         *
         * We use the second method in this function. We assume all threads are active.
         */
        assert(__activemask() == 0xFFFFFFFF);

        // This is the block this thread is going to write to
        const uint8_t current_out_block = current_out / sizeof(uint32_t);

        // If the current output block is not the same as the active block, we can assume we are ahead (as being behind
        // is impossible/illegal)
        const bool current_is_ahead = active_out_block != current_out_block;

        // If the current out pointer is not aligned to block boundaries, we know there is data in the current_out_block
        const bool current_block_has_unflushed_data = current_out % sizeof(uint32_t) != 0;

        // There is still unflushed data if the current block is ahead of the active block,
        // or if there is still data in the current block
        const bool has_unflushed_data = current_is_ahead || current_block_has_unflushed_data;

        // If any thread has unflushed data, we must flush
        const bool any_thread_has_unflushed_data = __popc(__ballot_sync(0xFFFFFFFF, has_unflushed_data)) > 0;

        return any_thread_has_unflushed_data;
    }

    __device__ inline bool must_flush(const uint8_t active_out_block, const uint8_t current_out) {
        /* We must flush the current active block in two scenarios:
         * 1: All threads are ahead of the current active_out_block
         *      - Every 4 cycles
         *      - Ballot to see if everyone is done
         * 2: One (or more) threads risk overflowing their buffer
         *      - A thread is in the block before the active block and can only write one more byte
         */

        // We assume current_out to be already updated in this function. So current_out = the next location that will be
        // written to. All threads need to check this, since this is a collaborative write
        assert(__activemask() == 0xFFFFFFFF);

        // This is the block this thread is going to write to
        const uint8_t current_out_block = current_out / sizeof(uint32_t);

        // If the current output block is not the same as the active block, we can assume we are ahead (as being behind
        // is impossible/illegal)
        const bool current_is_ahead = active_out_block != current_out_block;

        // If the next block is the currently active block, we are using the last available block
        const bool last_free_block = (current_out_block + 1) % tile_out_word_buf_size == active_out_block;
        // If the current location we are writing to is 3 (assuming uint32, that means the last byte in the block), we
        // risk overflowing if we get another escape
        const bool escape_can_overflow = current_out % sizeof(uint32_t) >= 2;
        // If we are using the last free block and we risk a block overflow, we are risking a buffer overflow. Need to
        // flush
        const bool risk_overflow = last_free_block && escape_can_overflow;

        // If any thread has the potential to overflow, flush
        const bool any_thread_risk_overflow = __popc(__ballot_sync(0xFFFFFFFF, risk_overflow)) > 0;
        // If all threads are ahead of the current buffer, flush
        const int threads_ahead = __popc(__ballot_sync(0xFFFFFFFF, current_is_ahead));
        const bool all_thread_ahead = threads_ahead == 32;

        return any_thread_risk_overflow || all_thread_ahead;
    }

    __device__ bool flush(uint8_t* dst, uint32_t result[tile_out_word_buf_size][THREAD_COUNT],
                          const uint32_t flushes_before, const uint8_t active_out_block) {
        // We assume that all threads participate in this, and that they all have the same active_out_block. This should
        // be maintained by calling functions
        assert(__activemask() == 0xFFFFFFFF);

        // This block is about to overrun its buffer, so this specific block is not compressing well at all (CR < 1.0),
        // so return and signal. Just use original data for this block and don't compress it :(
        if (flushes_before >= tile_out_len_words - 1) {
            return false;
        }

        // We also assume there is no buffer overrun. If there are more flushes than tile_out_len_words, the tiles are
        // writing more data than they are allowed to
        assert(flushes_before < tile_out_len_words - 1);

        // Determine block output location
        uint8_t* dst_loc = dst + blockIdx.x * (uint64_t)TMP_OUT_BLOCK_SIZE; // + warp_id * (uint64_t)TMP_OUT_WARP_SIZE;
        auto* dst_aligned = (uint32_t*)dst_loc;
        assert((uintptr_t)dst_loc % TMP_WORD_ALIGNMENT == 0); // Ensure 4-byte alignment

        // Write active block to dst memory
        const uint32_t word = result[active_out_block][threadIdx.x];
        dst_aligned[flushes_before * THREAD_COUNT + threadIdx.x] = word;

        // Reset the current block for next use
        result[active_out_block][threadIdx.x] = 0xFEFEFEFE;
        return true;
    }

    struct EncodeResult {
        uint32_t bytes_written;
        uint32_t flushes_before;
        uint64_t spillover;
        uint8_t spillover_len;
        uint8_t out;
        uint8_t active_out_block;
        bool flush_failed;
    };

    __device__ inline bool is_early_flush(const uint8_t out, const uint8_t active_out_block) {
        /*
         * An early flush means this thread hasn't fully filled the current active_out_block with data, but is
         * forced to flush because another thread in the warp is risking a buffer overflow.
         * If this happens this thread needs to update its out pointer to match the start of the new active block.
         */
        const uint8_t current_out_block = out / sizeof(uint32_t);

        // Whenever the current output block is equal to the active block, this is an early flush
        // since the thread would still write to the active block given the chance
        return active_out_block == current_out_block;
    }

    /*
     * flushes_before: Flushed that have been done before this cycle
     * out: Output pointer on a (thread/tile scale), so within the output blocks for this thread
     * active_out_block: Points to the current block that is being written to by all threads (and not flushed yet)
     */
    __device__ EncodeResult compaction_encode_chunked_local(uint8_t* dst, uint32_t flushes_before, uint8_t out,
                                                            uint8_t active_out_block,
                                                            const symbols::SmallSymbolMatchTableData& symbol_table,
                                                            uint32_t result[tile_out_word_buf_size][THREAD_COUNT],
                                                            uint32_t load[tile_word_buf_size][THREAD_COUNT],
                                                            uint64_t spillover, const uint8_t spillover_len,
                                                            const bool last_chunk) {
        // In and out pointers / counts from a local perspective
        uint8_t local_out = 0;
        uint8_t local_in = 0;

        uint32_t first_load_block = load[0][threadIdx.x];
        uint32_t second_load_block = load[1][threadIdx.x];
        uint32_t third_load_block = load[2][threadIdx.x];
        uint8_t first_block_offset = 0;
        uint8_t load_block_offset = 2; // Points to the location of the third_block in memory

        constexpr uint16_t search_len =
            n_regs_per_chunk * sizeof(uint32_t); // can look ahead at all available bytes (capped at 8 anyways)
        constexpr uint16_t encode_len = search_len - 7;
        // cannot encode when there are less than 8 bytes available, unless in last chunk

        // First handle spillover
        for (int i = 0; i < 8; i++) {
            if (local_in == i && i < spillover_len) {
                const symbols::GPUSymbol s = create_symbol_with_spillover_local(
                    spillover, spillover_len - local_in, first_load_block, second_load_block, fsst::Symbol::maxLength);
                uint16_t code = symbol_table.findLongestSymbol(s);
                auto sym = (uint8_t)code;
                auto sym_len = (uint8_t)(code >> 8);
                uint8_t escape = sym == fsst::Symbol::escape;

                assert(out + 1 != active_out_block * sizeof(uint32_t)); // Checks for buffer overrun on escape
                assert(sym != fsst::Symbol::ignore);
                pack_results_local(result, out, sym);
                if (escape)
                    pack_results_local(result, (out + 1) % n_out_symbols_buf,
                                       (uint8_t)get_first_byte_local(s.val.num));

                // Update pointers
                local_out += 1 + escape;
                local_in += sym_len;

                out += 1 + escape;
                out %= n_out_symbols_buf; // Make sure that output wraps around

                // Bookkeeping of spillover data
                spillover >>= 8 * sym_len;
            }

            // Check if we have to flush, and do it if needed
            if (must_flush(active_out_block, out)) {
                // If this is an early flush (== active block is not yet filled by this thread) we need to add some
                // padding
                if (is_early_flush(out, active_out_block)) {
                    const uint8_t padding = sizeof(uint32_t) - out % sizeof(uint32_t);
                    out += padding;
                    out %= n_out_symbols_buf; // Make sure that output wraps around
                    assert(out % sizeof(uint32_t) == 0);
                }

                if (!flush(dst, result, flushes_before, active_out_block)) return EncodeResult{.flush_failed = true};
                active_out_block = (active_out_block + 1) % tile_out_word_buf_size;
                flushes_before += 1;
            }
        }

        // Update registers for possible shift after overusage of spillover
        shift_registers_with_base(load, local_in - spillover_len, &first_block_offset, &load_block_offset,
                                  &first_load_block, &second_load_block, &third_load_block);

        // Then handle regular block
        const uint8_t encode_range = last_chunk ? search_len : encode_len;
        for (int i = 0; i < encode_range; i++) {
            if (local_in == i + spillover_len) {
                const symbols::GPUSymbol s = create_symbol_no_spillover_local(
                    first_load_block, second_load_block, third_load_block, first_block_offset,
                    min((int)fsst::Symbol::maxLength, search_len + spillover_len - local_in));
                uint16_t code = symbol_table.findLongestSymbol(s);
                auto sym = (uint8_t)code;
                auto sym_len = (uint8_t)(code >> 8);
                uint8_t escape = sym == fsst::Symbol::escape;

                assert(out + 1 != active_out_block * sizeof(uint32_t)); // Checks for buffer overrun on escape
                assert(sym != fsst::Symbol::ignore);
                pack_results_local(result, out, sym);
                if (escape)
                    pack_results_local(result, (out + 1) % n_out_symbols_buf,
                                       (uint8_t)get_first_byte_local(s.val.num));

                // Update pointers
                local_out += 1 + escape;
                local_in += sym_len;

                out += 1 + escape;
                out %= n_out_symbols_buf; // Make sure that output wraps around

                // Bookkeeping of input data
                shift_registers_with_base(load, sym_len, &first_block_offset, &load_block_offset, &first_load_block,
                                          &second_load_block, &third_load_block);
            }

            // Check if we have to flush, and do it if needed
            if (must_flush(active_out_block, out)) {
                // If this is an early flush (== active block is not yet filled by this thread) we need to add some
                // padding
                if (is_early_flush(out, active_out_block)) {
                    const uint8_t padding = sizeof(uint32_t) - out % sizeof(uint32_t);
                    out += padding;
                    out %= n_out_symbols_buf; // Make sure that output wraps around
                    assert(out % sizeof(uint32_t) == 0);
                }

                if (!flush(dst, result, flushes_before, active_out_block)) return EncodeResult{.flush_failed = true};
                active_out_block = (active_out_block + 1) % tile_out_word_buf_size;
                flushes_before += 1;
            }
        }

        // Then create new spillover
        const uint8_t spilled_bytes = search_len + spillover_len - local_in;
        const symbols::GPUSymbol s = create_symbol_no_spillover_local(
            first_load_block, second_load_block, third_load_block, first_block_offset, spilled_bytes);

        return EncodeResult{local_out, flushes_before, s.val.num, spilled_bytes, out, active_out_block};
    }

    __global__ void gpu_compaction(GCompactionMetadata* metadata, CompactionV5TBlockHeader* headers,
                                   const uint8_t* src, uint8_t* tmp, uint8_t* dst) {
        uint32_t global_size = 0;
        __shared__ uint16_t n_flushes[SUB_BLOCKS];
        __shared__ uint32_t load[tile_word_buf_size][THREAD_COUNT];
        __shared__ uint32_t result[tile_out_word_buf_size][THREAD_COUNT];

        // Load metadata into shared mem
        __shared__ GCompactionMetadata m;
        load_metadata_local<GCompactionMetadata>(metadata, (uint32_t*)&m, SUPER_BLOCK_SIZE);

        // Checks
        assert((uintptr_t)src % WORD_ALIGNMENT == 0); // Ensure 8-byte alignment of source
        assert((uintptr_t)tmp % TMP_WORD_ALIGNMENT == 0); // Ensure 4-byte alignment of tmp
        assert((uintptr_t)dst % TMP_WORD_ALIGNMENT == 0); // Ensure 4-byte alignment of dest
        assert(m.symbol_table.singleCodes[fsst::Symbol::ignore].code() !=
            fsst::Symbol::ignore); // Ensure that ignore symbol has been properly mapped

        // Active data
        uint64_t spillover = 0;
        uint8_t spillover_len = 0;

        // Output tracking data
        uint32_t number_of_words_flushed = 0;
        uint8_t block_out_location = 0;
        uint8_t active_out_block = 0;

        // Keep track of encoding failure
        bool encode_failed = false;

        const auto aligned_src = (uint32_t*)src;

        // Make sure that result is in the correct state
        for (int i = 0; i < tile_out_word_buf_size; i++) {
            result[i][threadIdx.x] = 0xFEFEFEFE;
        }

        for (uint32_t chunk_id = 0; chunk_id < n_chunks; chunk_id++) {
            // Step 1: Coalesced load into working memory
            for (int i = 0; i < tile_word_buf_size; i++) {
                load[i][threadIdx.x] = aligned_src[n_words_per_block * blockIdx.x + n_words_per_tile * threadIdx.x +
                        (chunk_id * tile_word_buf_size + i)];
            }

            // Step 2: Run chunked compression on spillover and loaded window
            const auto encode_result = compaction_encode_chunked_local(
                tmp, number_of_words_flushed, block_out_location, active_out_block, m.symbol_table, result, load,
                spillover, spillover_len, chunk_id == n_chunks - 1);

            // Step 3: Update active data
            spillover = encode_result.spillover;
            spillover_len = encode_result.spillover_len;
            global_size += encode_result.bytes_written;
            number_of_words_flushed = encode_result.flushes_before;
            block_out_location = encode_result.out;
            active_out_block = encode_result.active_out_block;

            // If we failed to flush, this block shouldn't be encoded. Just stop in that case
            if (encode_result.flush_failed) {
                encode_failed = true;
                break;
            }
        }

        // Flush the buffers until all threads have fully written their data
        while (!encode_failed && can_flush(active_out_block, block_out_location)) {
            assert(__activemask() == 0xFFFFFFFF);

            // If this is an early flush (== active block is not yet filled by this thread) we need to add some padding
            if (is_early_flush(block_out_location, active_out_block)) {
                const uint8_t padding = sizeof(uint32_t) - block_out_location % sizeof(uint32_t);
                block_out_location += padding;
                block_out_location %= n_out_symbols_buf; // Make sure that output wraps around
            }

            if (!flush(tmp, result, number_of_words_flushed, active_out_block)) encode_failed = true;
            active_out_block = (active_out_block + 1) % tile_out_word_buf_size;
            number_of_words_flushed += 1;
        }

        assert(encode_failed || block_out_location % sizeof(uint32_t) == 0);

        // Update number of flushes in smem
        if (threadIdx.x % 32 == 0) {
            // Make sure that encoding failure in a warp propagates through the entire thread block (or at least thread 0)
            if (encode_failed) {
                number_of_words_flushed = 0;
            }

            n_flushes[threadIdx.x / 32] = number_of_words_flushed;
        }

        __syncthreads();

        // Find block max
        uint32_t target_number_of_words_flushed = number_of_words_flushed;
        for (const uint16_t flushes : n_flushes) {
            // If any thread flushed 0 times, its encoding failed. So the entire block fails
            if (flushes == 0) {
                encode_failed = true;
            }

            if (flushes > target_number_of_words_flushed) {
                target_number_of_words_flushed = flushes;
            }
        }

        // Make sure we can transpose it
        if (const uint32_t leftover_flushes = target_number_of_words_flushed % 32; leftover_flushes > 0)
            target_number_of_words_flushed += 32 - leftover_flushes;

        // Complete block
        active_out_block = 0;
        result[active_out_block][threadIdx.x] = 0xFEFEFEFE;
        while (!encode_failed && number_of_words_flushed < target_number_of_words_flushed) {
            assert(__activemask() == 0xFFFFFFFF);
            if (!flush(tmp, result, number_of_words_flushed, active_out_block)) encode_failed = true;
            number_of_words_flushed += 1;
        }

        __syncthreads();

        // Run block-level transpose (if the encoding step didn't fail)
        if (threadIdx.x == 0 && !encode_failed) {
            assert(number_of_words_flushed % 32 == 0);

            const auto aligned_transpose_src = (uint32_t*)(tmp + (uint64_t)blockIdx.x * TMP_OUT_BLOCK_SIZE);
            const auto aligned_transpose_dst = (uint32_t*)(dst + (uint64_t)blockIdx.x * TMP_OUT_BLOCK_SIZE);

            dim3 dimGrid(THREAD_COUNT / 32, number_of_words_flushed / 32, 1);
            dim3 dimBlock(32, 8, 1);
            transpose_no_bank_conflicts<32, 8><<<dimGrid, dimBlock>>>(aligned_transpose_dst, aligned_transpose_src);

            // Check that there were no launch errors
            assert(cudaSuccess == cudaGetLastError());
        }

        // Sum block output
        using BlockReduce = cub::BlockReduce<uint32_t, THREAD_COUNT>;
        __shared__ BlockReduce::TempStorage temp_storage;
        const uint32_t aggregate = BlockReduce(temp_storage).Sum(global_size);

        // Create block header
        if (threadIdx.x == 0) {
            CompactionV5TBlockHeader header = {
                {
                    .uncompressed_size = BLOCK_SIZE, // TODO: change this later to support padding in input
                    .compressed_size = encode_failed ? BLOCK_SIZE : aggregate, // If failed, compressed == uncompressed
                },
                header.flushes = encode_failed ? 0 : number_of_words_flushed, // If failed, act as if there were no flushed (==no data)
            };

            headers[blockIdx.x] = header;
        }
    }
} // namespace gtsst::compressors::compactionv5t
