#include <bench/gtsst-bench.cuh>
#include <bench/gtsst-prepare.cuh>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtsst/gtsst.cuh>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include <compressors/compactionv5t/compaction-compressor.cuh>

namespace gtsst::bench {
    namespace {
        struct RandomAccessResult {
            uint64_t queries = 0;
            uint64_t bytes = 0;
            double mqps = 0.0;
            double gbps = 0.0;
            double avg_len = 0.0;
        };

        struct LineStats {
            uint32_t lines = 0;
            uint64_t bytes = 0;
            double avg_len = 0.0;
        };

        struct QueryKey {
            uint32_t line_id;
            uint32_t pos;
        };

        struct BoundaryKey {
            uint32_t block_id;
            uint32_t u;
            uint32_t q;
        };

        static LineStats compute_line_stats(const uint8_t* data, const size_t len) {
            uint32_t lines = 0;
            uint64_t total_bytes = 0;
            size_t start = 0;
            for (size_t i = 0; i < len; i++) {
                if (data[i] == '\n') {
                    total_bytes += (i - start);
                    lines++;
                    start = i + 1;
                }
            }
            if (start < len) {
                total_bytes += (len - start);
                lines++;
            }
            const double avg = lines ? (static_cast<double>(total_bytes) / static_cast<double>(lines)) : 0.0;
            return LineStats{lines, total_bytes, avg};
        }

        static bool parse_compaction_v5t_payload(const uint8_t* compressed, const size_t compressed_len,
                                                 gtsst::compressors::CompactionV5TFileHeader* out_file_header,
                                                 std::vector<fsst_decoder_t>* out_decoders,
                                                 std::vector<gtsst::compressors::CompactionV5TBlockHeader>* out_block_headers,
                                                 const uint8_t** out_payload,
                                                 size_t* out_payload_len,
                                                 std::vector<uint64_t>* out_block_data_offsets) {
            if (compressed_len < sizeof(gtsst::compressors::CompactionV5TFileHeader)) {
                return false;
            }

            gtsst::compressors::CompactionV5TFileHeader file_header{};
            std::memcpy(&file_header, compressed, sizeof(file_header));

            size_t in = sizeof(gtsst::compressors::CompactionV5TFileHeader);
            if (file_header.num_tables == 0 || file_header.num_blocks == 0) {
                return false;
            }

            std::vector<fsst_decoder_t> decoder_array(file_header.num_tables);
            for (uint32_t table_id = 0; table_id < file_header.num_tables; table_id++) {
                gtsst::fsst::DecodingTable dec{};
                const size_t table_len = dec.import_table(compressed + in);
                if (table_len == 0) return false;
                decoder_array[table_id] = dec.decoder;
                in += table_len;
            }

            const size_t block_header_bytes =
                static_cast<size_t>(file_header.num_blocks) * sizeof(gtsst::compressors::CompactionV5TBlockHeader);
            if (in + block_header_bytes > compressed_len) return false;
            std::vector<gtsst::compressors::CompactionV5TBlockHeader> block_headers(file_header.num_blocks);
            std::memcpy(block_headers.data(), compressed + in, block_header_bytes);
            in += block_header_bytes;

            if (file_header.format_version >= 1 && file_header.num_splits > 0) {
                const size_t split_header_bytes =
                    static_cast<size_t>(file_header.num_blocks) * sizeof(gtsst::compressors::CompactionV5TSplitHeader);
                if (in + split_header_bytes > compressed_len) return false;
                in += split_header_bytes;
            }

            std::vector<uint64_t> block_data_offsets(file_header.num_blocks);
            uint64_t running = 0;
            for (uint32_t b = 0; b < file_header.num_blocks; b++) {
                block_data_offsets[b] = running;
                running += block_headers[b].compressed_size;
            }

            if (in + running > compressed_len) return false;

            *out_file_header = file_header;
            *out_decoders = std::move(decoder_array);
            *out_block_headers = std::move(block_headers);
            *out_payload = compressed + in;
            *out_payload_len = static_cast<size_t>(running);
            *out_block_data_offsets = std::move(block_data_offsets);
            return true;
        }

        static void map_queries_to_lines(const uint8_t* original, const size_t original_len,
                                         std::vector<QueryKey>& keys_sorted,
                                         std::vector<uint32_t>& out_offsets,
                                         std::vector<uint32_t>& out_lengths) {
            uint32_t cur_line = 0;
            size_t start = 0;
            size_t i = 0;

            while (start <= original_len && i < keys_sorted.size()) {
                size_t end = start;
                while (end < original_len && original[end] != '\n') end++;
                const uint32_t len = static_cast<uint32_t>(end - start);

                while (i < keys_sorted.size() && keys_sorted[i].line_id == cur_line) {
                    const uint32_t pos = keys_sorted[i].pos;
                    out_offsets[pos] = static_cast<uint32_t>(start);
                    out_lengths[pos] = len;
                    i++;
                }

                cur_line++;
                if (end >= original_len) break;
                start = end + 1;
            }
        }

        static void map_boundaries_to_cpos_skip(const uint8_t* block_data,
                                                const uint32_t compressed_size,
                                                const fsst_decoder_t& decoder,
                                                std::vector<BoundaryKey>::const_iterator begin,
                                                std::vector<BoundaryKey>::const_iterator end,
                                                std::vector<uint32_t>& out_cpos,
                                                std::vector<uint8_t>& out_skip) {
            uint32_t cp = 0;
            uint32_t up = 0;
            auto it = begin;

            while (it != end) {
                const uint32_t target = it->u;
                while (cp < compressed_size) {
                    const uint8_t code = block_data[cp];
                    const uint32_t in_len = (code == FSST_ESC) ? 2u : 1u;
                    const uint32_t out_len = (code == FSST_ESC) ? 1u : static_cast<uint32_t>(decoder.len[code]);

                    if (target < up + out_len) {
                        out_cpos[it->q] = cp;
                        out_skip[it->q] = static_cast<uint8_t>(target - up);
                        ++it;
                        break;
                    }

                    cp += in_len;
                    up += out_len;
                }

                if (cp >= compressed_size) {
                    // Corrupt stream or boundary out of range; map to end.
                    out_cpos[it->q] = compressed_size;
                    out_skip[it->q] = 0;
                    ++it;
                }
            }
        }

        __global__ void random_access_lines_kernel(
            const fsst_decoder_t* __restrict__ decoders,
            const gtsst::compressors::CompactionV5TBlockHeader* __restrict__ block_headers,
            const uint64_t* __restrict__ block_data_offsets,
            const uint8_t* __restrict__ compressed_data,
            const uint32_t* __restrict__ q_block_id,
            const uint32_t* __restrict__ q_cpos,
            const uint8_t* __restrict__ q_skip,
            const uint32_t* __restrict__ q_len,
            const uint32_t q_count,
            const uint32_t table_idx,
            uint32_t* __restrict__ out_hash
        ) {
            __shared__ uint8_t s_len[255];
            __shared__ uint64_t s_symbol[255];

            for (int i = threadIdx.x; i < 255; i += blockDim.x) {
                s_len[i] = decoders[table_idx].len[i];
                s_symbol[i] = decoders[table_idx].symbol[i];
            }
            __syncthreads();

            const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= q_count) return;

            const uint32_t block_id = q_block_id[tid];
            const uint32_t cp0 = q_cpos[tid];
            const uint32_t need0 = q_len[tid];
            uint32_t need = need0;
            uint32_t skip = q_skip[tid];

            const uint64_t block_in = block_data_offsets[block_id];
            const uint32_t compressed_size = block_headers[block_id].compressed_size;
            const uint8_t* src = compressed_data + block_in + cp0;
            const uint8_t* src_end = compressed_data + block_in + compressed_size;

            // FNV-1a over decoded bytes (prevents dead-code elimination).
            uint32_t h = 2166136261u;

            if (block_headers[block_id].flushes == 0) {
                // Uncompressed block: direct byte reads.
                const uint8_t* raw = src;
                while (need > 0 && raw < src_end) {
                    h ^= static_cast<uint32_t>(*raw++);
                    h *= 16777619u;
                    need--;
                }
                out_hash[tid] = h;
                return;
            }

            while (need > 0 && src < src_end) {
                const uint8_t code = *src;
                if (code == FSST_ESC) {
                    if (src + 1 >= src_end) break;
                    const uint8_t literal = *(src + 1);
                    if (skip == 0) {
                        h ^= static_cast<uint32_t>(literal);
                        h *= 16777619u;
                        need--;
                    } else {
                        skip--;
                    }
                    src += 2;
                    continue;
                }

                const uint8_t len = s_len[code];
                const uint64_t sym = s_symbol[code];
                for (uint32_t j = skip; j < len && need > 0; j++) {
                    const uint8_t b = static_cast<uint8_t>(sym >> (8u * j));
                    h ^= static_cast<uint32_t>(b);
                    h *= 16777619u;
                    need--;
                }
                skip = 0;
                src += 1;
            }

            out_hash[tid] = h;
        }

        static RandomAccessResult benchmark_random_access_lines(const uint8_t* original, const size_t original_len,
                                                                const uint8_t* compressed, const size_t compressed_len,
                                                                const int iterations) {
            RandomAccessResult r{};
            if (iterations <= 0) return r;

            const LineStats ls = compute_line_stats(original, original_len);
            if (ls.lines == 0) return r;

            const uint32_t queries = std::min<uint32_t>(1000000u, ls.lines);

            gtsst::compressors::CompactionV5TFileHeader file_header{};
            std::vector<fsst_decoder_t> decoder_array;
            std::vector<gtsst::compressors::CompactionV5TBlockHeader> block_headers;
            const uint8_t* payload = nullptr;
            size_t payload_len = 0;
            std::vector<uint64_t> block_data_offsets;
            if (!parse_compaction_v5t_payload(compressed, compressed_len, &file_header, &decoder_array, &block_headers,
                                              &payload, &payload_len, &block_data_offsets)) {
                return r;
            }

            // 1) Sample random line ids (with replacement), then map them to (offset,len) in one pass.
            std::vector<QueryKey> keys_sorted;
            keys_sorted.reserve(queries);
            std::mt19937 rng(42);
            std::uniform_int_distribution<uint32_t> dist(0, ls.lines - 1);
            for (uint32_t i = 0; i < queries; i++) {
                keys_sorted.push_back(QueryKey{dist(rng), i});
            }
            std::sort(keys_sorted.begin(), keys_sorted.end(),
                      [](const QueryKey& a, const QueryKey& b) { return a.line_id < b.line_id; });

            std::vector<uint32_t> q_offsets(queries, 0);
            std::vector<uint32_t> q_lengths(queries, 0);
            map_queries_to_lines(original, original_len, keys_sorted, q_offsets, q_lengths);

            // 2) Filter to queries that fit within a single block (rarely violated for line-delimited datasets).
            constexpr uint32_t kBlockSize = gtsst::compressors::compactionv5t::BLOCK_SIZE;
            std::vector<uint32_t> q_block_id;
            std::vector<uint32_t> q_u;
            std::vector<uint32_t> q_len;
            q_block_id.reserve(queries);
            q_u.reserve(queries);
            q_len.reserve(queries);

            for (uint32_t i = 0; i < queries; i++) {
                const uint32_t len = q_lengths[i];
                if (len == 0) continue;
                const uint32_t off = q_offsets[i];
                const uint64_t end = static_cast<uint64_t>(off) + static_cast<uint64_t>(len) - 1;
                const uint32_t b0 = off / kBlockSize;
                const uint32_t b1 = static_cast<uint32_t>(end / kBlockSize);
                if (b0 != b1) continue;
                if (b0 >= file_header.num_blocks) continue;
                q_block_id.push_back(b0);
                q_u.push_back(off - b0 * kBlockSize);
                q_len.push_back(len);
                r.bytes += len;
            }

            if (q_block_id.empty()) return r;
            r.queries = q_block_id.size();
            r.avg_len = r.queries ? (static_cast<double>(r.bytes) / static_cast<double>(r.queries)) : 0.0;

            // 3) Compute (cpos,skip) per query by scanning each block's compressed stream once.
            std::vector<uint32_t> q_cpos(r.queries, 0);
            std::vector<uint8_t> q_skip(r.queries, 0);

            std::vector<BoundaryKey> boundaries;
            boundaries.reserve(r.queries);
            for (uint32_t i = 0; i < r.queries; i++) {
                boundaries.push_back(BoundaryKey{q_block_id[i], q_u[i], i});
            }
            std::sort(boundaries.begin(), boundaries.end(), [](const BoundaryKey& a, const BoundaryKey& b) {
                if (a.block_id != b.block_id) return a.block_id < b.block_id;
                return a.u < b.u;
            });

            uint32_t cur_block = std::numeric_limits<uint32_t>::max();
            auto group_begin = boundaries.begin();
            for (auto it = boundaries.begin(); it != boundaries.end(); ++it) {
                if (it->block_id != cur_block) {
                    if (cur_block != std::numeric_limits<uint32_t>::max()) {
                        const auto& bh = block_headers[cur_block];
                        const uint8_t* block_data = payload + block_data_offsets[cur_block];
                        if (bh.flushes == 0) {
                            for (auto jt = group_begin; jt != it; ++jt) {
                                q_cpos[jt->q] = jt->u;
                                q_skip[jt->q] = 0;
                            }
                        } else {
                            const uint32_t table_idx = cur_block / gtsst::compressors::compactionv5t::SUPER_BLOCK_SIZE;
                            map_boundaries_to_cpos_skip(block_data, bh.compressed_size, decoder_array[table_idx],
                                                        group_begin, it, q_cpos, q_skip);
                        }
                    }
                    cur_block = it->block_id;
                    group_begin = it;
                }
            }
            if (cur_block != std::numeric_limits<uint32_t>::max()) {
                const auto& bh = block_headers[cur_block];
                const uint8_t* block_data = payload + block_data_offsets[cur_block];
                if (bh.flushes == 0) {
                    for (auto jt = group_begin; jt != boundaries.end(); ++jt) {
                        q_cpos[jt->q] = jt->u;
                        q_skip[jt->q] = 0;
                    }
                } else {
                    const uint32_t table_idx = cur_block / gtsst::compressors::compactionv5t::SUPER_BLOCK_SIZE;
                    map_boundaries_to_cpos_skip(block_data, bh.compressed_size, decoder_array[table_idx],
                                                group_begin, boundaries.end(), q_cpos, q_skip);
                }
            }

            // 4) Group queries by table id to allow shared staging of the decoder table.
            std::vector<uint32_t> counts(file_header.num_tables, 0);
            for (uint32_t i = 0; i < r.queries; i++) {
                const uint32_t t = q_block_id[i] / gtsst::compressors::compactionv5t::SUPER_BLOCK_SIZE;
                if (t < counts.size()) counts[t]++;
            }

            std::vector<uint32_t> starts(file_header.num_tables + 1, 0);
            for (uint32_t t = 0; t < file_header.num_tables; t++) starts[t + 1] = starts[t] + counts[t];

            std::vector<uint32_t> cursor = starts;
            std::vector<uint32_t> g_block_id(r.queries);
            std::vector<uint32_t> g_cpos(r.queries);
            std::vector<uint8_t> g_skip(r.queries);
            std::vector<uint32_t> g_len(r.queries);
            for (uint32_t i = 0; i < r.queries; i++) {
                const uint32_t t = q_block_id[i] / gtsst::compressors::compactionv5t::SUPER_BLOCK_SIZE;
                const uint32_t dst = cursor[t]++;
                g_block_id[dst] = q_block_id[i];
                g_cpos[dst] = q_cpos[i];
                g_skip[dst] = q_skip[i];
                g_len[dst] = q_len[i];
            }

            // 5) Device copies (excluded from timing).
            cudaStream_t stream{};
            checkedCUDACall(cudaStreamCreate(&stream));

            fsst_decoder_t* d_decoders = nullptr;
            gtsst::compressors::CompactionV5TBlockHeader* d_block_headers = nullptr;
            uint64_t* d_block_data_offsets = nullptr;
            uint8_t* d_payload = nullptr;

            uint32_t* d_q_block_id = nullptr;
            uint32_t* d_q_cpos = nullptr;
            uint8_t* d_q_skip = nullptr;
            uint32_t* d_q_len = nullptr;
            uint32_t* d_hash = nullptr;

            checkedCUDACall(cudaMalloc(&d_decoders, decoder_array.size() * sizeof(fsst_decoder_t)));
            checkedCUDACall(cudaMalloc(&d_block_headers, block_headers.size() * sizeof(block_headers[0])));
            checkedCUDACall(cudaMalloc(&d_block_data_offsets, block_data_offsets.size() * sizeof(uint64_t)));
            checkedCUDACall(cudaMalloc(&d_payload, payload_len));

            checkedCUDACall(cudaMemcpyAsync(d_decoders, decoder_array.data(), decoder_array.size() * sizeof(fsst_decoder_t),
                                            cudaMemcpyHostToDevice, stream));
            checkedCUDACall(cudaMemcpyAsync(d_block_headers, block_headers.data(), block_headers.size() * sizeof(block_headers[0]),
                                            cudaMemcpyHostToDevice, stream));
            checkedCUDACall(cudaMemcpyAsync(d_block_data_offsets, block_data_offsets.data(), block_data_offsets.size() * sizeof(uint64_t),
                                            cudaMemcpyHostToDevice, stream));
            checkedCUDACall(cudaMemcpyAsync(d_payload, payload, payload_len, cudaMemcpyHostToDevice, stream));

            checkedCUDACall(cudaMalloc(&d_q_block_id, r.queries * sizeof(uint32_t)));
            checkedCUDACall(cudaMalloc(&d_q_cpos, r.queries * sizeof(uint32_t)));
            checkedCUDACall(cudaMalloc(&d_q_skip, r.queries * sizeof(uint8_t)));
            checkedCUDACall(cudaMalloc(&d_q_len, r.queries * sizeof(uint32_t)));
            checkedCUDACall(cudaMalloc(&d_hash, r.queries * sizeof(uint32_t)));

            checkedCUDACall(cudaMemcpyAsync(d_q_block_id, g_block_id.data(), r.queries * sizeof(uint32_t),
                                            cudaMemcpyHostToDevice, stream));
            checkedCUDACall(cudaMemcpyAsync(d_q_cpos, g_cpos.data(), r.queries * sizeof(uint32_t),
                                            cudaMemcpyHostToDevice, stream));
            checkedCUDACall(cudaMemcpyAsync(d_q_skip, g_skip.data(), r.queries * sizeof(uint8_t),
                                            cudaMemcpyHostToDevice, stream));
            checkedCUDACall(cudaMemcpyAsync(d_q_len, g_len.data(), r.queries * sizeof(uint32_t),
                                            cudaMemcpyHostToDevice, stream));
            checkedCUDACall(cudaStreamSynchronize(stream));

            auto launch_all = [&](const int iters) {
                constexpr int threads = 256;
                for (int rep = 0; rep < iters; rep++) {
                    for (uint32_t t = 0; t < file_header.num_tables; t++) {
                        const uint32_t begin = starts[t];
                        const uint32_t count = starts[t + 1] - starts[t];
                        if (count == 0) continue;
                        const int blocks = static_cast<int>((count + threads - 1) / threads);
                        random_access_lines_kernel<<<blocks, threads, 0, stream>>>(
                            d_decoders, d_block_headers, d_block_data_offsets, d_payload,
                            d_q_block_id + begin, d_q_cpos + begin, d_q_skip + begin, d_q_len + begin,
                            count, t, d_hash + begin);
                    }
                }
            };

            // Warmup
            launch_all(1);
            checkedCUDACall(cudaPeekAtLastError());
            checkedCUDACall(cudaStreamSynchronize(stream));

            cudaEvent_t ev_start{}, ev_stop{};
            checkedCUDACall(cudaEventCreate(&ev_start));
            checkedCUDACall(cudaEventCreate(&ev_stop));
            checkedCUDACall(cudaEventRecord(ev_start, stream));
            launch_all(iterations);
            checkedCUDACall(cudaEventRecord(ev_stop, stream));
            checkedCUDACall(cudaEventSynchronize(ev_stop));
            float elapsed_ms = 0.0f;
            checkedCUDACall(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
            checkedCUDACall(cudaEventDestroy(ev_start));
            checkedCUDACall(cudaEventDestroy(ev_stop));

            const double avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(iterations);
            const double sec = avg_ms / 1000.0;
            if (sec > 0.0) {
                r.mqps = (static_cast<double>(r.queries) / 1.0e6) / sec;
                r.gbps = (static_cast<double>(r.bytes) / 1.0e9) / sec;
            }

            checkedCUDACall(cudaFree(d_hash));
            checkedCUDACall(cudaFree(d_q_len));
            checkedCUDACall(cudaFree(d_q_skip));
            checkedCUDACall(cudaFree(d_q_cpos));
            checkedCUDACall(cudaFree(d_q_block_id));

            checkedCUDACall(cudaFree(d_payload));
            checkedCUDACall(cudaFree(d_block_data_offsets));
            checkedCUDACall(cudaFree(d_block_headers));
            checkedCUDACall(cudaFree(d_decoders));
            checkedCUDACall(cudaStreamDestroy(stream));

            return r;
        }
    } // namespace

    size_t read_file_size(const char* filename) {
        // Open the file in binary mode
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return 0;
        }

        // Get the size of the file
        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        file.close();

        return file_size;
    }

    uint8_t* read_file(const char* filename, const size_t data_to_read, const size_t buffer_size, const bool silent) {
        assert(buffer_size >= data_to_read);
        // Open the file in binary mode
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return nullptr;
        }

        // Create a buffer to hold the data
        uint8_t* buffer = (uint8_t*)malloc(buffer_size);

        file.seekg(0, std::ios::beg);

        // Read the file into the buffer
        if (!file.read(reinterpret_cast<char*>(buffer), data_to_read)) {
            std::cerr << "Error: Failed to read file " << filename << std::endl;
            free(buffer);
            return nullptr;
        }

        file.close();

        if (!silent) {
            std::cout << "File read successfully. Size: " << data_to_read << " bytes." << std::endl;
        }

        return buffer;
    }

    void write_file(const char* filename, const char* data, const size_t len) {
        std::ofstream out_file(filename, std::ios::binary);
        out_file.write(data, len);
    }

    CompressionStats compress_single(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                     CompressionConfiguration& compression_configuration,
                                     CompressionManager& compression_manager) {
        CompressionStats stats = {.original_len = compression_configuration.input_buffer_size};

        const auto start = std::chrono::high_resolution_clock::now();
        const auto status = compression_manager.compress(src, dst, tmp, compression_configuration,
                                                         &stats.compress_len, stats.internal_stats);
        const auto end = std::chrono::high_resolution_clock::now();

        if (status != gtsstSuccess) {
            std::cerr << "Compression error: " << status << std::endl;
            exit(1);
        }

        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        stats.compression_duration = duration;
        stats.compression_throughput = (double)stats.original_len / (1.0e9 * (duration.count() * 1.0e-6));

        return stats;
    }

    AggregatedCompressionStats compress_repeat(const uint8_t* src, uint8_t* dst,
                                               uint8_t* tmp, CompressionConfiguration& compression_configuration,
                                               CompressionManager& compression_manager, int iterations) {
        AggregatedCompressionStats aggregated_stats = {};

        if (iterations == 0) {
            return aggregated_stats;
        }

        std::vector<std::chrono::microseconds> compression_duration;
        std::vector<double> compression_throughput;

        std::vector<std::chrono::microseconds> table_generation;
        std::vector<std::chrono::microseconds> precomputation;
        std::vector<std::chrono::microseconds> encoding;
        std::vector<std::chrono::microseconds> postprocessing;

        for (int i = 0; i < iterations; i++) {
            CompressionStats stats =
                compress_single(src, dst, tmp, compression_configuration, compression_manager);
            aggregated_stats.stats.push_back(stats);

            compression_duration.push_back(stats.compression_duration);
            compression_throughput.push_back(stats.compression_throughput);

            table_generation.push_back(stats.internal_stats.table_generation);
            precomputation.push_back(stats.internal_stats.precomputation);
            encoding.push_back(stats.internal_stats.encoding);
            postprocessing.push_back(stats.internal_stats.postprocessing);

            printf("encoding: %.3f\n",
                   (double)stats.original_len / (1.0e9 * (stats.internal_stats.encoding.count() * 1.0e-6)));
        }

        aggregated_stats.compression_duration =
            std::accumulate(compression_duration.begin(), compression_duration.end(), std::chrono::microseconds(0)) /
            compression_duration.size();
        aggregated_stats.compression_throughput =
            std::accumulate(compression_throughput.begin(), compression_throughput.end(), 0.) /
            static_cast<double>(compression_throughput.size());

        aggregated_stats.internal_stats.table_generation =
            std::accumulate(table_generation.begin(), table_generation.end(), std::chrono::microseconds(0)) /
            table_generation.size();
        aggregated_stats.internal_stats.precomputation =
            std::accumulate(precomputation.begin(), precomputation.end(), std::chrono::microseconds(0)) /
            precomputation.size();
        aggregated_stats.internal_stats.encoding =
            std::accumulate(encoding.begin(), encoding.end(), std::chrono::microseconds(0)) / encoding.size();
        aggregated_stats.internal_stats.postprocessing =
            std::accumulate(postprocessing.begin(), postprocessing.end(), std::chrono::microseconds(0)) /
            postprocessing.size();

        aggregated_stats.internal_throughputs.table_generation_throughput =
            (double)aggregated_stats.stats[0].original_len /
            (1.0e9 * (aggregated_stats.internal_stats.table_generation.count() * 1.0e-6));
        aggregated_stats.internal_throughputs.precomputation_throughput =
            (double)aggregated_stats.stats[0].original_len /
            (1.0e9 * (aggregated_stats.internal_stats.precomputation.count() * 1.0e-6));
        aggregated_stats.internal_throughputs.encoding_throughput = (double)aggregated_stats.stats[0].original_len /
            (1.0e9 * (aggregated_stats.internal_stats.encoding.count() * 1.0e-6));
        aggregated_stats.internal_throughputs.postprocessing_throughput =
            (double)aggregated_stats.stats[0].original_len /
            (1.0e9 * (aggregated_stats.internal_stats.postprocessing.count() * 1.0e-6));

        return aggregated_stats;
    }

    bool data_equal(const uint8_t* src, const uint8_t* src_other, const size_t size, const bool strict) {
        if (src == nullptr || src_other == nullptr) {
            return false;
        }

        for (size_t i = 0; i < size; i++) {
            if (src[i] != src_other[i]) {
                printf("error: %zu -> %d != %d\n", i, src[i], src_other[i]);

                // You don't want this, but GSST decompression leaves me no choice..
                if (strict) {
                    return false;
                }
            }
        }

        return true;
    }

    bool decompress_single(const char* filename, CompressionStats& stats, const uint8_t* src, uint8_t* dst,
                           const uint8_t* original_data, DecompressionConfiguration& decompression_configuration,
                           CompressionManager& compression_manager, const bool strict_checking, const int iterations) {
        std::vector<std::chrono::microseconds> decompression_times;

        for (int i = 0; i < iterations; i++) {
            const auto start = std::chrono::high_resolution_clock::now();
            const auto status =
                compression_manager.decompress(src, dst, decompression_configuration, &stats.decompress_len);
            const auto end = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            if (status != gtsstSuccess) {
                std::cerr << "Decompression error (" << filename << "): " << status << std::endl;
                exit(1);
            }

            // Use kernel-only time if available, otherwise use wall-clock time
            auto effective_duration = duration;
            if (decompression_configuration.kernel_time_us > 0) {
                effective_duration = std::chrono::microseconds(
                    static_cast<long long>(decompression_configuration.kernel_time_us));
            }

            printf("%.3f\n", (double)stats.original_len / (1.0e9 * (effective_duration.count() * 1.0e-6)));
            decompression_times.push_back(effective_duration);
        }

        stats.decompression_duration =
            std::accumulate(decompression_times.begin(), decompression_times.end(), std::chrono::microseconds(0)) /
            decompression_times.size();
        stats.decompression_throughput =
            (double)stats.decompress_len / (1.0e9 * (stats.decompression_duration.count() * 1.0e-6));
        stats.ratio = static_cast<float>(stats.original_len) / static_cast<float>(stats.compress_len);

        // Equality check, if available
        bool length_match = stats.decompress_len == stats.original_len;
        bool match = length_match;

        if (!length_match) {
            printf("error: decompression length mismatch. Expected %lu, was %lu\n", stats.original_len, stats.decompress_len);
        }

        if (original_data != nullptr && length_match) {
            uint8_t* dst_buf = dst;
            if (decompression_configuration.device_buffers) {
                dst_buf = (uint8_t*)malloc(stats.decompress_len);
                checkedCUDACall(cudaMemcpy(dst_buf, dst, stats.decompress_len, cudaMemcpyDeviceToHost));
            }

            match = data_equal(original_data, dst_buf, stats.original_len, strict_checking);

            if (decompression_configuration.device_buffers) {
                free(dst_buf);
            }
        }

        return match;
    }

    bool decompress_all(const char* filename, AggregatedCompressionStats& aggregated_stats, const uint8_t* src,
                        uint8_t* tmp, const uint8_t* original_data,
                        DecompressionConfiguration& decompression_configuration,
                        CompressionManager& compression_manager, const bool strict_checking, const int iterations) {
        bool all_match = true;

        if (aggregated_stats.stats.empty()) {
            return all_match;
        }

        std::vector<std::chrono::microseconds> decompression_duration;
        std::vector<double> decompression_throughput;
        std::vector<float> ratio;

        for (CompressionStats& stats : aggregated_stats.stats) {
            bool match = decompress_single(filename, stats, src, tmp, original_data, decompression_configuration,
                                           compression_manager, strict_checking, iterations);
            all_match &= match;

            decompression_duration.push_back(stats.decompression_duration);
            decompression_throughput.push_back(stats.decompression_throughput);
            ratio.push_back(stats.ratio);

            break;
        }

        aggregated_stats.decompression_duration =
            std::accumulate(decompression_duration.begin(), decompression_duration.end(),
                            std::chrono::microseconds(0)) /
            decompression_duration.size();
        aggregated_stats.decompression_throughput =
            std::accumulate(decompression_throughput.begin(), decompression_throughput.end(), 0.) /
            static_cast<double>(decompression_throughput.size());
        aggregated_stats.ratio = std::accumulate(ratio.begin(), ratio.end(), 0.f) / static_cast<float>(ratio.size());

        return all_match;
    }

    bool full_cycle(const char* filename, const int compression_iterations, const int decompression_iterations,
                    CompressionManager& compression_manager, const bool print_csv, const bool strict_checking) {
        size_t file_size = read_file_size(filename);
        const auto file_compression_config = compression_manager.configure_compression(file_size);

        // Use padding approach to support files smaller than one block
        const size_t buffer_size = check_buffer_required_length(file_size, file_compression_config);
        const auto src = read_file(filename, file_size, buffer_size);

        // Fix buffer if needed
        size_t data_len = file_size;
        if (const bool valid_buffer = fix_buffer(src, file_size, buffer_size, &data_len, file_compression_config);
            !valid_buffer) {
            std::cerr << "Unable to fix data from file " << filename << std::endl;
            return false;
        }

        // Print warning
        if (!strict_checking) {
            std::cerr << "Strict output checking is disabled, decompressed data might not match original data!"
                      << std::endl;
        }

        auto compression_configuration = compression_manager.configure_compression(data_len);
        const bool dev_buf = compression_configuration.device_buffers;

        // Allocate buffers
        auto* dst = (uint8_t*)malloc(compression_configuration.compression_buffer_size);
        uint8_t* tmp;
        uint8_t* func_src;
        uint8_t* func_dst;

        cudaStream_t mem_stream;
        checkedCUDACall(cudaStreamCreate(&mem_stream));

        if (dev_buf) {
            checkedCUDACall(cudaMallocAsync(&tmp, compression_configuration.temp_buffer_size, mem_stream));
            checkedCUDACall(cudaMallocAsync(&func_src, compression_configuration.input_buffer_size, mem_stream));
            checkedCUDACall(cudaMallocAsync(&func_dst, compression_configuration.compression_buffer_size, mem_stream));
            checkedCUDACall(cudaMemcpyAsync(func_src, src, data_len, cudaMemcpyHostToDevice, mem_stream));
            checkedCUDACall(cudaStreamSynchronize(mem_stream));
        } else {
            tmp = (uint8_t*)malloc(compression_configuration.temp_buffer_size);
            func_src = src;
            func_dst = dst;
        }

        // Run compression
        AggregatedCompressionStats compression_stats =
            compress_repeat(func_src, func_dst, tmp, compression_configuration, compression_manager,
                            compression_iterations);

        // Run decompression
        DecompressionConfiguration decompression_configuration =
            compression_manager.configure_decompression_from_compress(compression_stats.stats[0].compress_len,
                                                                      compression_configuration);
        bool dev_buf_decomp = decompression_configuration.device_buffers;

        if (dev_buf_decomp && !dev_buf) {
            printf("Either both comp & decomp need to use device buffers, or only comp. Not only decomp!");
            return false;
        }

        // Move dst to host if needed
        if (dev_buf && !dev_buf_decomp) {
            checkedCUDACall(cudaMemcpyAsync(dst, func_dst, compression_stats.stats[0].compress_len,
                                            cudaMemcpyDeviceToHost, mem_stream));
            checkedCUDACall(cudaStreamSynchronize(mem_stream));
        }

        uint8_t* decomp_tmp;

        if (dev_buf_decomp) {
            decomp_tmp = func_src;
        } else {
            decomp_tmp = (uint8_t*)malloc(decompression_configuration.decompression_buffer_size);
        }

        const bool matching_decompression =
            decompress_all(filename, compression_stats, dev_buf_decomp ? func_dst : dst, decomp_tmp, src,
                           decompression_configuration, compression_manager, strict_checking, decompression_iterations);

        // Random access benchmark (line-level): measure GPU kernel time with compressed data resident on GPU.
        RandomAccessResult ra{};
        try {
            // Use the original (unpadded) file bytes to define line boundaries.
            ra = benchmark_random_access_lines(src, file_size, dst, compression_stats.stats[0].compress_len, 10);
        } catch (...) {
            // Best-effort benchmarking: do not fail the full cycle on random-access errors.
        }

        // Free buffers
        if (dev_buf) {
            checkedCUDACall(cudaFreeAsync(tmp, mem_stream));
            checkedCUDACall(cudaFreeAsync(func_src, mem_stream));
            checkedCUDACall(cudaFreeAsync(func_dst, mem_stream));
        } else {
            free(tmp);
            free(decomp_tmp);
        }

        checkedCUDACall(cudaStreamSynchronize(mem_stream));
        checkedCUDACall(cudaStreamDestroy(mem_stream));
        checkedCUDACall(cudaDeviceReset());

        free(dst);
        free(src);

        // Print results
        if (print_csv) {
            printf("%lu,%lu,%lu,%lu,%.3f,%lu,%lu,%.3f,%.4f,%.3f,%lu,%.3f,%lu,%.3f,%lu,%.3f,%lu\n",
                   compression_configuration.block_size, data_len, compression_configuration.table_range,
                   compression_stats.compression_duration.count(), compression_stats.compression_throughput,
                   compression_stats.stats[0].compress_len, compression_stats.decompression_duration.count(),
                   compression_stats.decompression_throughput, compression_stats.ratio,
                   compression_stats.internal_throughputs.table_generation_throughput,
                   compression_stats.internal_stats.table_generation.count(),
                   compression_stats.internal_throughputs.precomputation_throughput,
                   compression_stats.internal_stats.precomputation.count(),
                   compression_stats.internal_throughputs.encoding_throughput,
                   compression_stats.internal_stats.encoding.count(),
                   compression_stats.internal_throughputs.postprocessing_throughput,
                   compression_stats.internal_stats.postprocessing.count());
        } else {
            printf("Cycles (%d, %d) completed. Stats:\n"
                   "\tParameters:\n"
                   "\t\tBlock size: %lu\n"
                   "\t\tInput size: %lu\n"
                   "\t\tEffective table size: %lu\n"
                   "\t\tFile name: %s\n"
                   "\tCompression:\n"
                   "\t\tDuration (us): %lu \n"
                   "\t\tThroughput (GB/s): %.3f\n"
                   "\t\tCompressed size: %lu\n"
                   "\tDecompression:\n"
                   "\t\tDuration (us): %lu\n"
                   "\t\tThroughput (GB/s): %.3f\n"
                   "\t\tRatio: %.4f\n"
                   "\tCompression phases:\n"
                   "\t\tTable generation (GB/s, us): %.3f (%lu)\n"
                   "\t\tPrecomputation (GB/s, us): %.3f (%lu)\n"
                   "\t\tEncoding (GB/s, us): %.3f (%lu)\n"
	                   "\t\tPostprocessing (GB/s, us): %.3f (%lu)\n",
	                   compression_iterations, decompression_iterations, compression_configuration.block_size, data_len,
	                   compression_configuration.table_range, filename, compression_stats.compression_duration.count(),
	                   compression_stats.compression_throughput, compression_stats.stats[0].compress_len,
	                   compression_stats.decompression_duration.count(), compression_stats.decompression_throughput,
	                   compression_stats.ratio, compression_stats.internal_throughputs.table_generation_throughput,
	                   compression_stats.internal_stats.table_generation.count(),
	                   compression_stats.internal_throughputs.precomputation_throughput,
	                   compression_stats.internal_stats.precomputation.count(),
	                   compression_stats.internal_throughputs.encoding_throughput,
	                   compression_stats.internal_stats.encoding.count(),
	                   compression_stats.internal_throughputs.postprocessing_throughput,
	                   compression_stats.internal_stats.postprocessing.count());

	            if (ra.queries > 0) {
	                printf("\tRandom access (lines): %.3f M q/s, %.3f GB/s (queries=%lu, avg_len=%.2f)\n",
	                       ra.mqps, ra.gbps, ra.queries, ra.avg_len);
	            }
	        }

	        return matching_decompression;
	    }

    bool full_cycle_directory(const std::vector<std::string>& directories, const bool use_dir,
                              const int compression_iterations, const int decompression_iterations,
                              CompressionManager& compression_manager, const bool print_csv,
                              const bool strict_checking) {
        for (auto& file : directories) {
            try {
                for (const auto& entry : std::filesystem::directory_iterator(file)) {
                    const bool match =
                        full_cycle(entry.path().c_str(), compression_iterations, decompression_iterations,
                                   compression_manager, print_csv, strict_checking);

                    // If any of the matches failed, return false to indicate cycle mismatch
                    if (!match) {
                        return false;
                    }

                    // If not using directory mode, just break out of loop after first iteration
                    if (!use_dir) {
                        break;
                    }
                }
            } catch (const std::filesystem::filesystem_error& err) {
                std::cerr << "Error: " << err.what() << "\n";
            }
        }

        return true;
    }
} // namespace gtsst::bench
