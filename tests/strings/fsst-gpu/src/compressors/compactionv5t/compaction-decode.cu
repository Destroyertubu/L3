#include <compressors/compactionv5t/compaction-decode.cuh>
#include <cstdint>

namespace gtsst::compressors::compactionv5t {

    static constexpr uint32_t kOutAlignBytes = 8;

    __device__ __forceinline__ uint64_t mask_low_bytes(const uint32_t nbytes) {
        if (nbytes == 0) return 0;
        if (nbytes >= 8) return 0xFFFFFFFFFFFFFFFFULL;
        return 0xFFFFFFFFFFFFFFFFULL >> ((8 - nbytes) * 8);
    }

    __device__ __forceinline__ void flush_word_if_possible(uint8_t*& dst, uint64_t& out_word, uint32_t& out_count) {
        // If destination is unaligned, flush single bytes until aligned.
        while (out_count > 0 && (reinterpret_cast<uintptr_t>(dst) & (kOutAlignBytes - 1)) != 0) {
            *dst++ = static_cast<uint8_t>(out_word);
            out_word >>= 8;
            out_count--;
        }

        // Only perform aligned 8B store when we have a full word.
        if (out_count == 8) {
            *reinterpret_cast<uint64_t*>(dst) = out_word;
            dst += 8;
            out_word = 0;
            out_count = 0;
        }
    }

    __device__ __forceinline__ void append_bytes(uint8_t*& dst, uint64_t& out_word, uint32_t& out_count,
                                                 uint64_t bytes, uint32_t nbytes) {
        // Append up to 8 bytes (little-endian order) into the staging word, flushing aligned words when possible.
        while (nbytes > 0) {
            const uint32_t space = 8 - out_count;
            const uint32_t take = (nbytes < space) ? nbytes : space;
            out_word |= (bytes & mask_low_bytes(take)) << (out_count * 8);
            out_count += take;
            bytes >>= (take * 8);
            nbytes -= take;

            if (out_count == 8) {
                flush_word_if_possible(dst, out_word, out_count);
            }
        }
    }

    __global__ void gpu_decompress(
        const fsst_decoder_t* __restrict__ decoders,
        const CompactionV5TBlockHeader* __restrict__ block_headers,
        const CompactionV5TSplitHeader* __restrict__ split_headers,
        const uint8_t* __restrict__ compressed_data,
        const uint64_t* __restrict__ block_data_offsets,
        uint8_t* __restrict__ output,
        uint32_t num_blocks,
        uint32_t block_size,
        uint32_t super_block_size,
        uint32_t num_splits
    ) {
        if (blockIdx.x >= num_blocks) return;

        // === 1. Load symbol table into shared memory ===
        __shared__ uint8_t s_len[255];
        __shared__ uint64_t s_symbol[255];

        const uint32_t table_idx = blockIdx.x / super_block_size;
        for (int i = threadIdx.x; i < 255; i += blockDim.x) {
            s_len[i] = decoders[table_idx].len[i];
            s_symbol[i] = decoders[table_idx].symbol[i];
        }
        __syncthreads();

        // === 2. Get block and split info ===
        const auto& bh = block_headers[blockIdx.x];
        const uint32_t split_id = threadIdx.x;
        const uint64_t block_in = block_data_offsets[blockIdx.x];
        const uint64_t block_out = (uint64_t)blockIdx.x * block_size;

        // Uncompressed block: cooperative memcpy
        if (bh.flushes == 0) {
            const uint32_t size = bh.compressed_size;
            for (uint32_t i = split_id; i < size; i += blockDim.x) {
                output[block_out + i] = compressed_data[block_in + i];
            }
            return;
        }

        if (split_id >= num_splits) return;

        const auto& sh = split_headers[blockIdx.x];

        // Compressed input range for this split
        const uint32_t in_start = sh.compressed_offsets[split_id];
        const uint32_t in_end = (split_id < num_splits - 1)
            ? sh.compressed_offsets[split_id + 1]
            : bh.compressed_size;

        // Output range for this split
        const uint32_t out_start = sh.uncompressed_offsets[split_id];
        const uint32_t out_end = (split_id < num_splits - 1)
            ? sh.uncompressed_offsets[split_id + 1]
            : bh.uncompressed_size;

        // === 3. Decode ===
        const uint8_t* src = compressed_data + block_in + in_start;
        const uint8_t* src_end = compressed_data + block_in + in_end;
        uint8_t* dst = output + block_out + out_start;
        const uint8_t* dst_end = output + block_out + out_end;

        uint64_t out_word = 0;
        uint32_t out_count = 0;

        // Decode loop with 4B-aligned loads when possible.
        while (src < src_end) {
            // Unaligned prefix (align input pointer to 4B).
            while (src < src_end && (reinterpret_cast<uintptr_t>(src) & 3) != 0) {
                const uint8_t code = *src++;
                if (code == FSST_ESC) {
                    if (src >= src_end) break;
                    append_bytes(dst, out_word, out_count, static_cast<uint64_t>(*src++), 1);
                } else {
                    const uint8_t len = s_len[code];
                    const uint64_t sym = s_symbol[code] & mask_low_bytes(len);
                    append_bytes(dst, out_word, out_count, sym, len);
                }
            }

            if (src + 4 > src_end) break;
            if ((reinterpret_cast<uintptr_t>(src) & 3) != 0) continue;

            // Aligned main step: read 4 bytes of codes at a time.
            uint32_t packed_codes = *reinterpret_cast<const uint32_t*>(src);
            src += 4;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint8_t code = static_cast<uint8_t>(packed_codes);
                packed_codes >>= 8;

                if (code == FSST_ESC) {
                    uint8_t literal = 0;
                    if (i < 3) {
                        literal = static_cast<uint8_t>(packed_codes);
                        packed_codes >>= 8;
                        i++; // Skip the payload byte inside this 4B chunk
                    } else {
                        if (src >= src_end) {
                            src = src_end;
                            break;
                        }
                        literal = *src++;
                    }
                    append_bytes(dst, out_word, out_count, static_cast<uint64_t>(literal), 1);
                } else {
                    const uint8_t len = s_len[code];
                    const uint64_t sym = s_symbol[code] & mask_low_bytes(len);
                    append_bytes(dst, out_word, out_count, sym, len);
                }
            }
        }

        // Tail bytes (also handles cases where an escape payload made the stream unaligned again).
        while (src < src_end) {
            const uint8_t code = *src++;
            if (code == FSST_ESC) {
                if (src >= src_end) break;
                append_bytes(dst, out_word, out_count, static_cast<uint64_t>(*src++), 1);
            } else {
                const uint8_t len = s_len[code];
                const uint64_t sym = s_symbol[code] & mask_low_bytes(len);
                append_bytes(dst, out_word, out_count, sym, len);
            }
        }

        // Flush remaining bytes (<= 7) conservatively.
        while (out_count > 0 && dst < dst_end) {
            *dst++ = static_cast<uint8_t>(out_word);
            out_word >>= 8;
            out_count--;
        }
    }

} // namespace gtsst::compressors::compactionv5t
