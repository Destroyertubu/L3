#ifndef COMPACTION_DECODE_CUH
#define COMPACTION_DECODE_CUH

#include <compressors/compactionv5t/compaction-compressor.cuh>
#include <compressors/compactionv5t/compaction-defines.cuh>
#include <fsst/fsst.cuh>

namespace gtsst::compressors::compactionv5t {

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
    );

} // namespace gtsst::compressors::compactionv5t

#endif // COMPACTION_DECODE_CUH
