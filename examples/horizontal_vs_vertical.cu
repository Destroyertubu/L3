/**
 * Horizontal vs Vertical Layout Comparison
 *
 * This educational example demonstrates the performance difference between
 * Horizontal (traditional) and Vertical (GPU-optimized) bit-packing layouts.
 *
 * Key Concepts:
 *
 * 1. HORIZONTAL LAYOUT (Traditional):
 *    - Deltas stored sequentially: δ₀, δ₁, δ₂, δ₃, ...
 *    - Each delta may span word boundaries
 *    - Random bit positions require conditional logic
 *    - Poor memory coalescing on GPU
 *
 * 2. VERTICAL LAYOUT (FastLanes-inspired):
 *    - 256 values organized as 32 lanes × 8 values
 *    - Lane i contains: v[i], v[i+32], v[i+64], ...
 *    - CRITICAL: Data is stored in TRANSPOSED order so that
 *      32 threads in a warp access CONSECUTIVE memory addresses
 *    - Perfect memory coalescing, no branch divergence
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *                    MEMORY LAYOUT COMPARISON
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * HORIZONTAL (Sequential packing):
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ Word 0: [δ₀][δ₁][δ₂][δ₃...     │ Word 1: ...δ₃][δ₄][δ₅][δ₆...        │
 * │         ↑    ↑    ↑  ↑                      ↑                          │
 * │         T0   T1   T2 T3 spans boundary!     T4                         │
 * └─────────────────────────────────────────────────────────────────────────┘
 * Problem: Thread 3 needs to read TWO words (branch divergence!)
 *
 * VERTICAL (Word-interleaved / Transposed):
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ Word 0:  Lane0 bits 0-31   ← Thread 0 reads                            │
 * │ Word 1:  Lane1 bits 0-31   ← Thread 1 reads                            │
 * │ Word 2:  Lane2 bits 0-31   ← Thread 2 reads                            │
 * │ ...                                                                     │
 * │ Word 31: Lane31 bits 0-31  ← Thread 31 reads                           │
 * │ ─────────────────────────────────────────────                          │
 * │ Word 32: Lane0 bits 32-63  ← Thread 0 reads (next iteration)           │
 * │ Word 33: Lane1 bits 32-63  ← Thread 1 reads                            │
 * │ ...                                                                     │
 * └─────────────────────────────────────────────────────────────────────────┘
 * Perfect: All 32 threads read consecutive words = 128-byte coalesced access!
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cstring>

// Configuration
constexpr int WARP_SIZE = 32;
constexpr int VALUES_PER_LANE = 8;
constexpr int MINI_VECTOR_SIZE = WARP_SIZE * VALUES_PER_LANE;  // 256

// ============================================================================
// HORIZONTAL LAYOUT: Traditional sequential bit-packing
// ============================================================================

/**
 * Horizontal Encoder (CPU)
 * Packs deltas sequentially: δ₀ at bit 0, δ₁ at bit w, δ₂ at bit 2w, ...
 */
void encodeHorizontal(const uint32_t* deltas, int n, int bit_width, uint32_t* packed) {
    int64_t bit_offset = 0;
    for (int i = 0; i < n; i++) {
        uint32_t delta = deltas[i] & ((1ULL << bit_width) - 1);
        int word_idx = bit_offset / 32;
        int bit_in_word = bit_offset % 32;

        // Write lower bits
        packed[word_idx] |= (delta << bit_in_word);

        // Handle word boundary crossing
        if (bit_in_word + bit_width > 32) {
            int bits_in_first = 32 - bit_in_word;
            packed[word_idx + 1] |= (delta >> bits_in_first);
        }

        bit_offset += bit_width;
    }
}

/**
 * Horizontal Decoder Kernel (GPU)
 * Each thread extracts one delta from sequential bit positions
 *
 * Problem: Different threads access different bit positions,
 * leading to scattered memory access and branch divergence.
 */
__global__ void decodeHorizontalKernel(
    const uint32_t* __restrict__ packed,
    uint32_t* __restrict__ output,
    int n,
    int bit_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Calculate bit position for this delta
    int64_t bit_offset = (int64_t)idx * bit_width;
    int word_idx = bit_offset / 32;
    int bit_in_word = bit_offset % 32;

    uint32_t mask = (1ULL << bit_width) - 1;

    // Read first word
    uint32_t word0 = packed[word_idx];
    uint32_t delta = (word0 >> bit_in_word) & mask;

    // Branch: Handle word boundary (causes divergence!)
    if (bit_in_word + bit_width > 32) {
        uint32_t word1 = packed[word_idx + 1];
        int bits_from_first = 32 - bit_in_word;
        delta |= (word1 << bits_from_first) & mask;
    }

    output[idx] = delta;
}

// ============================================================================
// VERTICAL LAYOUT (NAIVE): Lane-major storage (NOT coalesced!)
// ============================================================================

/**
 * Vertical Encoder NAIVE (CPU) - Lane-Major Layout
 *
 * This is the NAIVE approach that stores each lane's data together:
 *   [Lane0 all words][Lane1 all words]...[Lane31 all words]
 *
 * Memory access pattern:
 *   Thread 0 reads word 0
 *   Thread 1 reads word 4  (stride = words_per_lane!)
 *   Thread 2 reads word 8
 *   ...
 *   → STRIDED ACCESS, not coalesced!
 */
void encodeVerticalNaive(const uint32_t* deltas, int n, int bit_width, uint32_t* packed) {
    int num_mini_vectors = n / MINI_VECTOR_SIZE;
    int bits_per_lane = bit_width * VALUES_PER_LANE;
    int words_per_lane = (bits_per_lane + 31) / 32;
    int words_per_mini_vector = words_per_lane * WARP_SIZE;

    for (int mv = 0; mv < num_mini_vectors; mv++) {
        uint32_t* mv_base = packed + mv * words_per_mini_vector;

        for (int lane = 0; lane < WARP_SIZE; lane++) {
            // Clear this lane's words
            for (int w = 0; w < words_per_lane; w++) {
                mv_base[lane * words_per_lane + w] = 0;
            }

            // Pack 8 values into this lane (Lane-Major: lane's words are contiguous)
            int64_t bit_offset = 0;
            for (int v = 0; v < VALUES_PER_LANE; v++) {
                int src_idx = mv * MINI_VECTOR_SIZE + lane + v * WARP_SIZE;
                uint32_t delta = deltas[src_idx] & ((1ULL << bit_width) - 1);

                int word_in_lane = bit_offset / 32;
                int bit_in_word = bit_offset % 32;

                mv_base[lane * words_per_lane + word_in_lane] |= (delta << bit_in_word);

                if (bit_in_word + bit_width > 32 && word_in_lane + 1 < words_per_lane) {
                    int bits_in_first = 32 - bit_in_word;
                    mv_base[lane * words_per_lane + word_in_lane + 1] |= (delta >> bits_in_first);
                }

                bit_offset += bit_width;
            }
        }
    }

    // Handle tail
    int tail_start = num_mini_vectors * MINI_VECTOR_SIZE;
    int tail_count = n - tail_start;
    if (tail_count > 0) {
        uint32_t* tail_base = packed + num_mini_vectors * words_per_mini_vector;
        int64_t bit_offset = 0;
        for (int i = 0; i < tail_count; i++) {
            uint32_t delta = deltas[tail_start + i] & ((1ULL << bit_width) - 1);
            int word_idx = bit_offset / 32;
            int bit_in_word = bit_offset % 32;

            tail_base[word_idx] |= (delta << bit_in_word);
            if (bit_in_word + bit_width > 32) {
                tail_base[word_idx + 1] |= (delta >> (32 - bit_in_word));
            }
            bit_offset += bit_width;
        }
    }
}

/**
 * Vertical Decoder Kernel NAIVE (GPU) - Lane-Major Access
 *
 * Thread 'lane' reads word at offset 'lane * words_per_lane + w'
 * This creates STRIDED access with stride = words_per_lane
 */
__global__ void decodeVerticalNaiveKernel(
    const uint32_t* __restrict__ packed,
    uint32_t* __restrict__ output,
    int n,
    int bit_width
) {
    int bits_per_lane = bit_width * VALUES_PER_LANE;
    int words_per_lane = (bits_per_lane + 31) / 32;
    int words_per_mini_vector = words_per_lane * WARP_SIZE;
    int num_mini_vectors = n / MINI_VECTOR_SIZE;

    int mv_idx = blockIdx.x;
    int lane = threadIdx.x % WARP_SIZE;

    if (mv_idx >= num_mini_vectors) return;

    const uint32_t* mv_base = packed + mv_idx * words_per_mini_vector;

    // NAIVE ACCESS: Thread 'lane' reads word at offset 'lane * words_per_lane + w'
    // Stride = words_per_lane (e.g., 4), NOT consecutive!
    uint32_t regs[4];
    #pragma unroll
    for (int w = 0; w < words_per_lane && w < 4; w++) {
        regs[w] = mv_base[lane * words_per_lane + w];  // STRIDED ACCESS!
    }

    uint32_t mask = (1ULL << bit_width) - 1;

    #pragma unroll
    for (int v = 0; v < VALUES_PER_LANE; v++) {
        int64_t bit_offset = (int64_t)v * bit_width;
        int word_in_lane = bit_offset / 32;
        int bit_in_word = bit_offset % 32;

        uint32_t delta = (regs[word_in_lane] >> bit_in_word) & mask;

        if (bit_in_word + bit_width > 32 && word_in_lane + 1 < words_per_lane) {
            int bits_from_first = 32 - bit_in_word;
            delta |= (regs[word_in_lane + 1] << bits_from_first) & mask;
        }

        int out_idx = mv_idx * MINI_VECTOR_SIZE + lane + v * WARP_SIZE;
        output[out_idx] = delta;
    }
}

// ============================================================================
// VERTICAL LAYOUT (TRANSPOSED): Word-interleaved storage (COALESCED!)
// ============================================================================

/**
 * Vertical Encoder TRANSPOSED (CPU) - TRUE COALESCED LAYOUT
 *
 * This is the key insight of FastLanes:
 *
 * Instead of storing lanes sequentially:
 *   [Lane0 all words][Lane1 all words]...[Lane31 all words]  <- BAD (strided access)
 *
 * We TRANSPOSE the layout:
 *   [All lanes word0][All lanes word1]...[All lanes wordN]   <- GOOD (coalesced!)
 *
 * Memory layout for one mini-vector (256 values):
 *   Offset 0:  Lane 0, word 0 (bits 0-31)
 *   Offset 1:  Lane 1, word 0 (bits 0-31)
 *   ...
 *   Offset 31: Lane 31, word 0 (bits 0-31)
 *   Offset 32: Lane 0, word 1 (bits 32-63)
 *   Offset 33: Lane 1, word 1 (bits 32-63)
 *   ...
 *
 * Now 32 threads reading offsets 0-31 is a SINGLE 128-byte memory transaction!
 */
void encodeVerticalTransposed(const uint32_t* deltas, int n, int bit_width, uint32_t* packed) {
    int num_mini_vectors = n / MINI_VECTOR_SIZE;
    int bits_per_lane = bit_width * VALUES_PER_LANE;  // e.g., 13 * 8 = 104 bits
    int words_per_lane = (bits_per_lane + 31) / 32;   // e.g., ceil(104/32) = 4 words
    int words_per_mini_vector = words_per_lane * WARP_SIZE;  // e.g., 4 * 32 = 128 words

    for (int mv = 0; mv < num_mini_vectors; mv++) {
        // Temporary buffer to pack each lane's data
        uint32_t lane_data[32][4] = {0};  // [lane][word_in_lane]

        // Step 1: Pack each lane's 8 values into lane_data
        for (int lane = 0; lane < WARP_SIZE; lane++) {
            int64_t bit_offset = 0;
            for (int v = 0; v < VALUES_PER_LANE; v++) {
                int src_idx = mv * MINI_VECTOR_SIZE + lane + v * WARP_SIZE;
                uint32_t delta = deltas[src_idx] & ((1ULL << bit_width) - 1);

                int word_in_lane = bit_offset / 32;
                int bit_in_word = bit_offset % 32;

                lane_data[lane][word_in_lane] |= (delta << bit_in_word);

                if (bit_in_word + bit_width > 32 && word_in_lane + 1 < words_per_lane) {
                    int bits_in_first = 32 - bit_in_word;
                    lane_data[lane][word_in_lane + 1] |= (delta >> bits_in_first);
                }

                bit_offset += bit_width;
            }
        }

        // Step 2: TRANSPOSE - Write in word-interleaved order for coalesced access
        // This is the KEY difference from naive vertical layout!
        uint32_t* mv_base = packed + mv * words_per_mini_vector;
        for (int w = 0; w < words_per_lane; w++) {
            for (int lane = 0; lane < WARP_SIZE; lane++) {
                // Transposed layout: word w of all lanes are consecutive
                mv_base[w * WARP_SIZE + lane] = lane_data[lane][w];
            }
        }
    }

    // Handle tail (values after complete mini-vectors) - use horizontal packing
    int tail_start = num_mini_vectors * MINI_VECTOR_SIZE;
    int tail_count = n - tail_start;
    if (tail_count > 0) {
        uint32_t* tail_base = packed + num_mini_vectors * words_per_mini_vector;
        int64_t bit_offset = 0;
        for (int i = 0; i < tail_count; i++) {
            uint32_t delta = deltas[tail_start + i] & ((1ULL << bit_width) - 1);
            int word_idx = bit_offset / 32;
            int bit_in_word = bit_offset % 32;

            tail_base[word_idx] |= (delta << bit_in_word);
            if (bit_in_word + bit_width > 32) {
                tail_base[word_idx + 1] |= (delta >> (32 - bit_in_word));
            }
            bit_offset += bit_width;
        }
    }
}

/**
 * Vertical Decoder Kernel (GPU) - TRUE COALESCED ACCESS
 *
 * Memory access pattern with transposed layout:
 *
 *   First load (w=0):
 *     Thread 0 reads packed[0]    (Lane 0, word 0)
 *     Thread 1 reads packed[1]    (Lane 1, word 0)
 *     Thread 2 reads packed[2]    (Lane 2, word 0)
 *     ...
 *     Thread 31 reads packed[31]  (Lane 31, word 0)
 *     → Single 128-byte coalesced memory transaction!
 *
 *   Second load (w=1):
 *     Thread 0 reads packed[32]   (Lane 0, word 1)
 *     Thread 1 reads packed[33]   (Lane 1, word 1)
 *     ...
 *     → Another 128-byte coalesced transaction!
 */
__global__ void decodeVerticalKernel(
    const uint32_t* __restrict__ packed,
    uint32_t* __restrict__ output,
    int n,
    int bit_width
) {
    int bits_per_lane = bit_width * VALUES_PER_LANE;
    int words_per_lane = (bits_per_lane + 31) / 32;
    int words_per_mini_vector = words_per_lane * WARP_SIZE;
    int num_mini_vectors = n / MINI_VECTOR_SIZE;

    // Process mini-vectors: one warp per mini-vector
    int mv_idx = blockIdx.x;
    int lane = threadIdx.x % WARP_SIZE;

    if (mv_idx >= num_mini_vectors) return;

    const uint32_t* mv_base = packed + mv_idx * words_per_mini_vector;

    // Load this lane's packed words into registers
    // KEY: With transposed layout, consecutive threads read consecutive addresses!
    uint32_t regs[4];
    #pragma unroll
    for (int w = 0; w < words_per_lane && w < 4; w++) {
        // TRANSPOSED ACCESS: Thread 'lane' reads word at offset 'w * WARP_SIZE + lane'
        //
        // For w=0: Thread 0 reads [0], Thread 1 reads [1], ..., Thread 31 reads [31]
        //          → 32 consecutive addresses = 128-byte coalesced load!
        //
        // For w=1: Thread 0 reads [32], Thread 1 reads [33], ..., Thread 31 reads [63]
        //          → Another 128-byte coalesced load!
        regs[w] = mv_base[w * WARP_SIZE + lane];
    }

    // Extract 8 values from registers (NO memory access, NO divergence!)
    uint32_t mask = (1ULL << bit_width) - 1;

    #pragma unroll
    for (int v = 0; v < VALUES_PER_LANE; v++) {
        int64_t bit_offset = (int64_t)v * bit_width;
        int word_in_lane = bit_offset / 32;
        int bit_in_word = bit_offset % 32;

        uint32_t delta = (regs[word_in_lane] >> bit_in_word) & mask;

        // Handle spanning - UNIFORM for all threads (NO divergence!)
        // Because all threads process the same bit positions within their lane
        if (bit_in_word + bit_width > 32 && word_in_lane + 1 < words_per_lane) {
            int bits_from_first = 32 - bit_in_word;
            delta |= (regs[word_in_lane + 1] << bits_from_first) & mask;
        }

        // Write output: reconstruct original index
        int out_idx = mv_idx * MINI_VECTOR_SIZE + lane + v * WARP_SIZE;
        output[out_idx] = delta;
    }
}

// ============================================================================
// Visualization and Analysis
// ============================================================================

void printMemoryAccessPattern(int bit_width) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║           MEMORY ACCESS PATTERN ANALYSIS (bit_width=" << std::setw(2) << bit_width << ")         ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ HORIZONTAL Layout: Sequential packing                          │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ Thread │ Bit Range   │ Word(s)   │ Spans Boundary?              │" << std::endl;
    std::cout << "├────────┼─────────────┼───────────┼──────────────────────────────┤" << std::endl;

    int divergent_count = 0;
    for (int t = 0; t < 8; t++) {
        int64_t bit_start = t * bit_width;
        int64_t bit_end = bit_start + bit_width - 1;
        int word_start = bit_start / 32;
        int word_end = bit_end / 32;
        bool spans = (word_start != word_end);
        if (spans) divergent_count++;

        std::cout << "│ " << std::setw(6) << t
                  << " │ " << std::setw(4) << bit_start << "-" << std::setw(4) << bit_end
                  << " │ " << word_start;
        if (spans) std::cout << "," << word_end << "      ";
        else std::cout << "        ";
        std::cout << " │ " << (spans ? "YES → BRANCH DIVERGENCE!" : "No")
                  << std::string(spans ? 4 : 26, ' ') << "│" << std::endl;
    }
    std::cout << "│   ...  │    ...      │    ...    │ ...                          │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;

    // Calculate divergence rate for full warp
    int total_divergent = 0;
    for (int t = 0; t < 32; t++) {
        int64_t bit_start = t * bit_width;
        int64_t bit_end = bit_start + bit_width - 1;
        if (bit_start / 32 != bit_end / 32) total_divergent++;
    }
    std::cout << "  → " << total_divergent << "/32 threads span boundaries ("
              << (total_divergent * 100 / 32) << "% divergence rate)" << std::endl;

    int words_per_lane = (bit_width * 8 + 31) / 32;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ VERTICAL Layout: Transposed/Word-interleaved packing           │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ Memory Layout (Transposed for coalesced access):               │" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│   Word  0: Lane 0, bits 0-31    ← Thread 0 reads               │" << std::endl;
    std::cout << "│   Word  1: Lane 1, bits 0-31    ← Thread 1 reads               │" << std::endl;
    std::cout << "│   Word  2: Lane 2, bits 0-31    ← Thread 2 reads               │" << std::endl;
    std::cout << "│   ...                                                          │" << std::endl;
    std::cout << "│   Word 31: Lane 31, bits 0-31   ← Thread 31 reads              │" << std::endl;
    std::cout << "│   ──────────────────────────────────────────────               │" << std::endl;
    std::cout << "│   Word 32: Lane 0, bits 32-63   ← Thread 0 reads (iter 2)      │" << std::endl;
    std::cout << "│   Word 33: Lane 1, bits 32-63   ← Thread 1 reads               │" << std::endl;
    std::cout << "│   ...                                                          │" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ Access Pattern (first load, w=0):                              │" << std::endl;
    std::cout << "│   Thread 0  reads word[0]  ─┐                                  │" << std::endl;
    std::cout << "│   Thread 1  reads word[1]   │                                  │" << std::endl;
    std::cout << "│   Thread 2  reads word[2]   ├─→ 32 CONSECUTIVE addresses       │" << std::endl;
    std::cout << "│   ...                       │    = 128-byte COALESCED load!    │" << std::endl;
    std::cout << "│   Thread 31 reads word[31] ─┘                                  │" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ → " << words_per_lane << " coalesced loads per warp, each thread extracts 8 values    │" << std::endl;
    std::cout << "│ → 0% branch divergence (all threads same code path)            │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;
}

void printDetailedComparison(int bit_width) {
    int words_per_lane = (bit_width * 8 + 31) / 32;

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    WHY VERTICAL IS FASTER                        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ 1. MEMORY COALESCING                                            │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ HORIZONTAL:                                                     │" << std::endl;
    std::cout << "│   Each thread calculates its own bit offset                     │" << std::endl;
    std::cout << "│   Thread 0: bit 0    → word 0                                   │" << std::endl;
    std::cout << "│   Thread 1: bit " << std::setw(2) << bit_width << "   → word " << (bit_width / 32) << std::endl;
    std::cout << "│   Thread 2: bit " << std::setw(2) << (2*bit_width) << "   → word " << (2*bit_width / 32) << std::endl;
    std::cout << "│   → Scattered access pattern, multiple memory transactions      │" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ VERTICAL (Transposed):                                          │" << std::endl;
    std::cout << "│   Thread 0: word[0], Thread 1: word[1], ..., Thread 31: word[31]│" << std::endl;
    std::cout << "│   → SINGLE 128-byte memory transaction for entire warp!         │" << std::endl;
    std::cout << "│   → " << words_per_lane << " such transactions extract all 256 values                   │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ 2. BRANCH DIVERGENCE                                            │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ HORIZONTAL:                                                     │" << std::endl;
    std::cout << "│   if (bit_in_word + bit_width > 32) {  // Some threads: YES     │" << std::endl;
    std::cout << "│       // read second word              // Other threads: NO      │" << std::endl;
    std::cout << "│   }                                    // → DIVERGENCE!         │" << std::endl;
    std::cout << "│   GPU must serialize: first run YES threads, then NO threads    │" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ VERTICAL:                                                       │" << std::endl;
    std::cout << "│   All threads process SAME bit positions within their lane      │" << std::endl;
    std::cout << "│   if (bit_in_word + bit_width > 32) {  // ALL threads: same!    │" << std::endl;
    std::cout << "│       // read second word                                       │" << std::endl;
    std::cout << "│   }                                    // → NO DIVERGENCE!      │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ 3. REGISTER REUSE & ARITHMETIC INTENSITY                        │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ HORIZONTAL:                                                     │" << std::endl;
    std::cout << "│   1 thread processes 1 value                                    │" << std::endl;
    std::cout << "│   Memory loads: 1-2 per value                                   │" << std::endl;
    std::cout << "│   Arithmetic ops: ~5 per value                                  │" << std::endl;
    std::cout << "│   Ratio: ~3 ops/load (memory bound)                             │" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ VERTICAL:                                                       │" << std::endl;
    std::cout << "│   1 thread processes 8 values                                   │" << std::endl;
    std::cout << "│   Memory loads: " << words_per_lane << " words → 8 values                                │" << std::endl;
    std::cout << "│   Arithmetic ops: ~40 per 8 values                              │" << std::endl;
    std::cout << "│   Ratio: ~" << (40 / words_per_lane) << " ops/load (better compute utilization)              │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;
}

// ============================================================================
// Benchmark and Comparison
// ============================================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║      Horizontal vs Vertical Layout Performance Comparison        ║" << std::endl;
    std::cout << "║                                                                  ║" << std::endl;
    std::cout << "║  Comparing THREE implementations:                                ║" << std::endl;
    std::cout << "║  1. Horizontal: Traditional sequential bit-packing              ║" << std::endl;
    std::cout << "║  2. Vertical Naive: Lane-major storage (strided access)         ║" << std::endl;
    std::cout << "║  3. Vertical Transposed: Word-interleaved (coalesced access)    ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    // Test configuration - LARGE DATA SIZE to stress L2 cache
    const int64_t N = 256LL * 819200;  // ~200M values (~800MB uncompressed)
    const int BIT_WIDTH = 13;          // Odd bit width to maximize boundary crossings
    const int NUM_RUNS = 20;

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  - Number of values: " << N << " (" << (N * sizeof(uint32_t) / 1024.0 / 1024.0) << " MB uncompressed)" << std::endl;
    std::cout << "  - Bit width: " << BIT_WIDTH << " bits" << std::endl;
    std::cout << "  - Compressed size: " << (N * BIT_WIDTH / 8.0 / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  - Number of runs: " << NUM_RUNS << std::endl;

    // Show memory access pattern analysis
    printMemoryAccessPattern(BIT_WIDTH);

    // Allocate host memory
    std::vector<uint32_t> h_deltas(N);
    std::vector<uint32_t> h_horizontal_packed;
    std::vector<uint32_t> h_vertical_naive_packed;
    std::vector<uint32_t> h_vertical_transposed_packed;
    std::vector<uint32_t> h_output_horizontal(N);
    std::vector<uint32_t> h_output_vertical_naive(N);
    std::vector<uint32_t> h_output_vertical_transposed(N);

    // Generate random deltas
    std::cout << "\nGenerating random deltas..." << std::endl;
    uint32_t max_delta = (1U << BIT_WIDTH) - 1;
    srand(42);  // Fixed seed for reproducibility
    for (int64_t i = 0; i < N; i++) {
        h_deltas[i] = rand() % (max_delta + 1);
    }

    // Calculate packed sizes
    int64_t horizontal_words = (N * BIT_WIDTH + 31) / 32;

    int bits_per_lane = BIT_WIDTH * VALUES_PER_LANE;
    int words_per_lane = (bits_per_lane + 31) / 32;
    int words_per_mini_vector = words_per_lane * WARP_SIZE;
    int64_t num_mini_vectors = N / MINI_VECTOR_SIZE;
    int tail_count = N % MINI_VECTOR_SIZE;
    int tail_words = (tail_count * BIT_WIDTH + 31) / 32;
    int64_t vertical_words = num_mini_vectors * words_per_mini_vector + tail_words;

    h_horizontal_packed.resize(horizontal_words, 0);
    h_vertical_naive_packed.resize(vertical_words, 0);
    h_vertical_transposed_packed.resize(vertical_words, 0);

    // Encode with all three layouts
    std::cout << "\nEncoding data with three layouts..." << std::endl;
    encodeHorizontal(h_deltas.data(), N, BIT_WIDTH, h_horizontal_packed.data());
    encodeVerticalNaive(h_deltas.data(), N, BIT_WIDTH, h_vertical_naive_packed.data());
    encodeVerticalTransposed(h_deltas.data(), N, BIT_WIDTH, h_vertical_transposed_packed.data());

    std::cout << "  - Horizontal packed:          " << horizontal_words << " words ("
              << (horizontal_words * 4.0 / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "  - Vertical (both) packed:     " << vertical_words << " words ("
              << (vertical_words * 4.0 / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "  - Vertical overhead:          " << std::fixed << std::setprecision(1)
              << ((vertical_words - horizontal_words) * 100.0 / horizontal_words) << "%" << std::endl;

    // Allocate device memory
    uint32_t *d_horizontal_packed, *d_vertical_naive_packed, *d_vertical_transposed_packed;
    uint32_t *d_output_horizontal, *d_output_vertical_naive, *d_output_vertical_transposed;

    cudaMalloc(&d_horizontal_packed, horizontal_words * sizeof(uint32_t));
    cudaMalloc(&d_vertical_naive_packed, vertical_words * sizeof(uint32_t));
    cudaMalloc(&d_vertical_transposed_packed, vertical_words * sizeof(uint32_t));
    cudaMalloc(&d_output_horizontal, N * sizeof(uint32_t));
    cudaMalloc(&d_output_vertical_naive, N * sizeof(uint32_t));
    cudaMalloc(&d_output_vertical_transposed, N * sizeof(uint32_t));

    cudaMemcpy(d_horizontal_packed, h_horizontal_packed.data(),
               horizontal_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertical_naive_packed, h_vertical_naive_packed.data(),
               vertical_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertical_transposed_packed, h_vertical_transposed_packed.data(),
               vertical_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ==================== Benchmark Horizontal ====================
    std::cout << "\n=== Benchmarking Horizontal Decoding ===" << std::endl;

    int threads_per_block = 256;
    int blocks_horizontal = (N + threads_per_block - 1) / threads_per_block;

    // Warmup
    decodeHorizontalKernel<<<blocks_horizontal, threads_per_block>>>(
        d_horizontal_packed, d_output_horizontal, N, BIT_WIDTH);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int r = 0; r < NUM_RUNS; r++) {
        decodeHorizontalKernel<<<blocks_horizontal, threads_per_block>>>(
            d_horizontal_packed, d_output_horizontal, N, BIT_WIDTH);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float horizontal_ms;
    cudaEventElapsedTime(&horizontal_ms, start, stop);
    horizontal_ms /= NUM_RUNS;

    double horizontal_throughput = (N * sizeof(uint32_t) / 1e9) / (horizontal_ms / 1000.0);
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << horizontal_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << horizontal_throughput << " GB/s" << std::endl;

    // ==================== Benchmark Vertical Naive ====================
    std::cout << "\n=== Benchmarking Vertical NAIVE Decoding (Strided Access) ===" << std::endl;

    int blocks_vertical = num_mini_vectors;

    // Warmup
    decodeVerticalNaiveKernel<<<blocks_vertical, WARP_SIZE>>>(
        d_vertical_naive_packed, d_output_vertical_naive, N, BIT_WIDTH);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int r = 0; r < NUM_RUNS; r++) {
        decodeVerticalNaiveKernel<<<blocks_vertical, WARP_SIZE>>>(
            d_vertical_naive_packed, d_output_vertical_naive, N, BIT_WIDTH);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float vertical_naive_ms;
    cudaEventElapsedTime(&vertical_naive_ms, start, stop);
    vertical_naive_ms /= NUM_RUNS;

    double vertical_naive_throughput = (N * sizeof(uint32_t) / 1e9) / (vertical_naive_ms / 1000.0);
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << vertical_naive_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << vertical_naive_throughput << " GB/s" << std::endl;

    // ==================== Benchmark Vertical Transposed ====================
    std::cout << "\n=== Benchmarking Vertical TRANSPOSED Decoding (Coalesced Access) ===" << std::endl;

    // Warmup
    decodeVerticalKernel<<<blocks_vertical, WARP_SIZE>>>(
        d_vertical_transposed_packed, d_output_vertical_transposed, N, BIT_WIDTH);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int r = 0; r < NUM_RUNS; r++) {
        decodeVerticalKernel<<<blocks_vertical, WARP_SIZE>>>(
            d_vertical_transposed_packed, d_output_vertical_transposed, N, BIT_WIDTH);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float vertical_transposed_ms;
    cudaEventElapsedTime(&vertical_transposed_ms, start, stop);
    vertical_transposed_ms /= NUM_RUNS;

    double vertical_transposed_throughput = (N * sizeof(uint32_t) / 1e9) / (vertical_transposed_ms / 1000.0);
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << vertical_transposed_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << vertical_transposed_throughput << " GB/s" << std::endl;

    // ==================== Verify Correctness ====================
    std::cout << "\n=== Verifying Correctness ===" << std::endl;

    cudaMemcpy(h_output_horizontal.data(), d_output_horizontal, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_vertical_naive.data(), d_output_vertical_naive, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_vertical_transposed.data(), d_output_vertical_transposed, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int horizontal_errors = 0, vertical_naive_errors = 0, vertical_transposed_errors = 0;

    for (int i = 0; i < N; i++) {
        if (h_output_horizontal[i] != h_deltas[i]) horizontal_errors++;
        if (h_output_vertical_naive[i] != h_deltas[i]) vertical_naive_errors++;
        if (h_output_vertical_transposed[i] != h_deltas[i]) vertical_transposed_errors++;
    }

    std::cout << "  Horizontal:           " << (horizontal_errors == 0 ? "PASS" : "FAIL")
              << " (" << horizontal_errors << " errors)" << std::endl;
    std::cout << "  Vertical Naive:       " << (vertical_naive_errors == 0 ? "PASS" : "FAIL")
              << " (" << vertical_naive_errors << " errors)" << std::endl;
    std::cout << "  Vertical Transposed:  " << (vertical_transposed_errors == 0 ? "PASS" : "FAIL")
              << " (" << vertical_transposed_errors << " errors)" << std::endl;

    // ==================== Summary ====================
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                          PERFORMANCE SUMMARY                              ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Layout                │ Time (ms)  │ Throughput │ vs Horiz │ vs Naive   ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Horizontal            │ " << std::setw(10) << std::fixed << std::setprecision(4) << horizontal_ms
              << " │ " << std::setw(7) << std::setprecision(0) << horizontal_throughput << " GB/s"
              << " │   1.00x  │    -       ║" << std::endl;
    std::cout << "║  Vertical Naive        │ " << std::setw(10) << std::fixed << std::setprecision(4) << vertical_naive_ms
              << " │ " << std::setw(7) << std::setprecision(0) << vertical_naive_throughput << " GB/s"
              << " │ " << std::setw(6) << std::setprecision(2) << (horizontal_ms / vertical_naive_ms) << "x"
              << "  │   1.00x    ║" << std::endl;
    std::cout << "║  Vertical Transposed   │ " << std::setw(10) << std::fixed << std::setprecision(4) << vertical_transposed_ms
              << " │ " << std::setw(7) << std::setprecision(0) << vertical_transposed_throughput << " GB/s"
              << " │ " << std::setw(6) << std::setprecision(2) << (horizontal_ms / vertical_transposed_ms) << "x"
              << "  │ " << std::setw(6) << std::setprecision(2) << (vertical_naive_ms / vertical_transposed_ms) << "x    ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│                     KEY OBSERVATIONS                            │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ 1. Vertical Naive vs Horizontal:                                │" << std::endl;
    std::cout << "│    - Still faster due to: no branch divergence, register reuse │" << std::endl;
    std::cout << "│    - But memory access is STRIDED (not optimal)                 │" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ 2. Vertical Transposed vs Vertical Naive:                       │" << std::endl;
    std::cout << "│    - TRANSPOSITION enables TRUE coalesced access                │" << std::endl;
    std::cout << "│    - 32 threads read 32 consecutive words = 128B transaction    │" << std::endl;
    std::cout << "│    - This is the key insight of FastLanes!                      │" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ 3. Memory Access Patterns:                                      │" << std::endl;
    std::cout << "│    Horizontal:  Scattered (different bit offsets per thread)    │" << std::endl;
    std::cout << "│    Naive:       Strided (stride = " << words_per_lane << " words)                       │" << std::endl;
    std::cout << "│    Transposed:  Coalesced (consecutive addresses)               │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;

    // Cleanup
    cudaFree(d_horizontal_packed);
    cudaFree(d_vertical_naive_packed);
    cudaFree(d_vertical_transposed_packed);
    cudaFree(d_output_horizontal);
    cudaFree(d_output_vertical_naive);
    cudaFree(d_output_vertical_transposed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
