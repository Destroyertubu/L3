#ifndef COMPACTION_COMPRESSOR5t_CUH
#define COMPACTION_COMPRESSOR5t_CUH
#include <compressors/compactionv5t/compaction-defines.cuh>
#include <compressors/shared.cuh>
#include <gtsst/gtsst.cuh>

namespace gtsst::compressors {
  struct CompactionV5TBlockHeader : BlockHeader {
    uint16_t flushes;
  };

  struct CompactionV5TFileHeader : FileHeader {
    uint16_t num_sub_blocks;
    uint8_t format_version;   // 0 = v0 (no splits), 1 = v1 (with splits)
    uint8_t num_splits;       // Number of splits per block (e.g., 32)
  };

  // Split metadata per block for GPU decompression (v1 format)
  struct CompactionV5TSplitHeader {
    uint32_t compressed_offsets[compactionv5t::NUM_SPLITS];    // Compressed data start offset per split
    uint32_t uncompressed_offsets[compactionv5t::NUM_SPLITS];  // Decompressed output start offset per split
  };

  struct CompactionV5TCompressor : CompressionManager {
    CompressionConfiguration configure_compression(size_t buf_size) override;
    GTSSTStatus compress(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                         CompressionConfiguration& config, size_t* out_size, CompressionStatistics& stats) override;

    DecompressionConfiguration configure_decompression(size_t buf_size) override;
    DecompressionConfiguration configure_decompression_from_compress(
        size_t buf_size, CompressionConfiguration& config) override;
    GTSSTStatus decompress(const uint8_t* src, uint8_t* dst, DecompressionConfiguration& config,
                           size_t* out_size) override;

    GTSSTStatus validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                             CompressionConfiguration& config) override;
  };

  // Compute split boundaries by scanning compressed block data
  void compute_split_boundaries(const uint8_t* block_data, uint32_t compressed_size,
                                const fsst::DecodingTable& decoder, uint32_t num_splits,
                                CompactionV5TSplitHeader& split_header);
} // namespace gtsst::compressors

#endif // COMPACTION_COMPRESSOR5t_CUH
