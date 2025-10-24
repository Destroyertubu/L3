//this version optimize the compressioin ratio, best ratio - REFACTORED TO SoA

#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <climits>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <fstream>
#include <string>
#include <type_traits>
#include <immintrin.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <stdlib.h> // For posix_memalign
#include <mma.h>

// Fix Kernel random access optimization
// work stealing opt

// CUDA Error Checking Macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Configuration constants
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024
#define TILE_SIZE 4096          // Default partition size for fixed-length
#define MAX_DELTA_BITS 64      // Max bits for a single delta value
#define MIN_PARTITION_SIZE 128  // Minimum partition size for variable-length partitioning
#define SPLIT_THRESHOLD 0.1     // Split threshold for variable-length partitioning

// Model types
enum ModelType {
    MODEL_CONSTANT = 0,
    MODEL_LINEAR = 1,
    MODEL_POLYNOMIAL2 = 2,
    MODEL_POLYNOMIAL3 = 3,
    MODEL_DIRECT_COPY = 4   // New model type for direct copy when overflow detected
};

// Enhanced partition metadata structure - ONLY FOR HOST USE AND SERIALIZATION
struct PartitionInfo {
    int32_t start_idx;
    int32_t end_idx;
    int32_t model_type;
    double model_params[4];
    int32_t delta_bits;
    int64_t delta_array_bit_offset;
    long long error_bound;
    int32_t reserved[1];
};

struct PartitionMetaOpt {
    int32_t start_idx;
    int32_t model_type;
    int32_t delta_bits;
    int32_t partition_len;
    double theta0;
    double theta1;
    int64_t bit_offset_base;
};

// Template compressed data structure - SoA LAYOUT
template<typename T>
struct CompressedData {
    // --- SoA Data Pointers (all are device pointers) ---
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    int32_t* d_model_types;
    double* d_model_params; // Note: This will store params for all partitions contiguously.
                            // For a linear model, layout will be [p0_t0, p0_t1, p1_t0, p1_t1, ...].
    int32_t* d_delta_bits;
    int64_t* d_delta_array_bit_offsets;
    long long* d_error_bounds;

    uint32_t* delta_array;          // This remains the same.

// 383 -------------------------------------------------------
    long long* d_plain_deltas;
// 383 -------------------------------------------------------

    // --- Host-side metadata ---
    int num_partitions;
    int total_values;

    // --- Device-side self pointer ---
    CompressedData<T>* d_self;
};

// Serialized data container (for host-side blob)
struct SerializedData {
    uint8_t* data;
    size_t size;

    SerializedData() : data(nullptr), size(0) {}
    ~SerializedData() {
        if (data) {
            delete[] data;
            data = nullptr;
        }
    }
    SerializedData(const SerializedData&) = delete;
    SerializedData& operator=(const SerializedData&) = delete;
    SerializedData(SerializedData&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    SerializedData& operator=(SerializedData&& other) noexcept {
        if (this != &other) {
            if (data) delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};

// Binary format header for serialized data - UPDATED FOR SoA
struct SerializedHeader {
    uint32_t magic;
    uint32_t version; // Increment this to reflect the new format (4)
    uint32_t total_values;
    uint32_t num_partitions;

    // --- New SoA Table Offsets and Sizes ---
    // All offsets are relative to the beginning of the data blob.
    uint64_t start_indices_offset;
    uint64_t end_indices_offset;
    uint64_t model_types_offset;
    uint64_t model_params_offset;
    uint64_t delta_bits_offset;
    uint64_t delta_array_bit_offsets_offset;
    uint64_t error_bounds_offset;
    uint64_t delta_array_offset; // Offset for the main delta bitstream

    // Field for the size in bytes of the model_params array, as it contains doubles.
    uint64_t model_params_size_bytes;
    uint64_t delta_array_size_bytes; // This remains.

    uint32_t data_type_size;
    uint32_t header_checksum;
    uint32_t reserved[3];
};

// Direct access handle for serialized data - UPDATED FOR SoA
// ALIGNMENT FIX: Ensure proper alignment for CUDA
template<typename T>
struct alignas(256) DirectAccessHandle {  // Increased alignment to 256 bytes
    const uint8_t* data_blob_host;
    const SerializedHeader* header_host;
    
    // Host-side SoA pointers
    const int32_t* start_indices_host;
    const int32_t* end_indices_host;
    const int32_t* model_types_host;
    const double* model_params_host;
    const int32_t* delta_bits_host;
    const int64_t* delta_array_bit_offsets_host;
    const long long* error_bounds_host;
    const uint32_t* delta_array_host;
    
    size_t data_blob_size;

    uint8_t* d_data_blob_device;
    SerializedHeader* d_header_device;
    
    // Device-side SoA pointers
    int32_t* d_start_indices_device;
    int32_t* d_end_indices_device;
    int32_t* d_model_types_device;
    double* d_model_params_device;
    int32_t* d_delta_bits_device;
    int64_t* d_delta_array_bit_offsets_device;
    long long* d_error_bounds_device;
    uint32_t* d_delta_array_device;
    
    // Padding to ensure size is multiple of alignment
    char padding[256 - (sizeof(void*) * 20 + sizeof(size_t)) % 256];
};

// Partition metadata structure for shared memory caching
struct PartitionMeta {
    int32_t start_idx;
    int32_t end_idx;
    int32_t model_type;
    double theta0;
    double theta1;
};

// Helper function to check if values might cause overflow in double precision
template<typename T>
__device__ __host__ inline bool mightOverflowDoublePrecision(T value) {
    if (std::is_signed<T>::value) {
        return false;  // Signed types within long long range are OK
    } else {
        // For unsigned types, check if value exceeds double precision (2^53)
        const uint64_t DOUBLE_PRECISION_LIMIT = (1ULL << 53);
        return static_cast<uint64_t>(value) > DOUBLE_PRECISION_LIMIT;
    }
}

// Helper template for safe delta calculation
template<typename T>
__device__ __host__ inline long long calculateDelta(T actual, T predicted) {
    if (std::is_signed<T>::value) {
        return static_cast<long long>(actual) - static_cast<long long>(predicted);
    } else {
        // For unsigned types
        if (sizeof(T) == 8) {
            // For 64-bit unsigned types (unsigned long long)
            unsigned long long actual_ull = static_cast<unsigned long long>(actual);
            unsigned long long pred_ull = static_cast<unsigned long long>(predicted);
            
            if (actual_ull >= pred_ull) {
                unsigned long long diff = actual_ull - pred_ull;
                if (diff <= static_cast<unsigned long long>(LLONG_MAX)) {
                    return static_cast<long long>(diff);
                } else {
                    return LLONG_MAX;
                }
            } else {
                unsigned long long diff = pred_ull - actual_ull;
                if (diff <= static_cast<unsigned long long>(LLONG_MAX)) {
                    return -static_cast<long long>(diff);
                } else {
                    return LLONG_MIN;
                }
            }
        } else {
            // For smaller unsigned types, direct conversion is safe
            return static_cast<long long>(actual) - static_cast<long long>(predicted);
        }
    }
}

// Helper template for applying delta to prediction
template<typename T>
__device__ __host__ inline T applyDelta(T predicted, long long delta) {
    if (std::is_signed<T>::value) {
        // For signed types, simple addition
        return predicted + static_cast<T>(delta);
    } else {
        // For unsigned types, use unsigned arithmetic to handle wraparound correctly
        if (sizeof(T) == 8) {
            // For 64-bit unsigned types
            unsigned long long pred_ull = static_cast<unsigned long long>(predicted);
            unsigned long long delta_ull = static_cast<unsigned long long>(delta);
            return static_cast<T>(pred_ull + delta_ull);
        } else if (sizeof(T) == 4) {
            // For 32-bit unsigned types
            unsigned long pred_ul = static_cast<unsigned long>(predicted);
            unsigned long delta_ul = static_cast<unsigned long>(static_cast<long>(delta));
            return static_cast<T>(pred_ul + delta_ul);
        } else if (sizeof(T) == 2) {
            // For 16-bit unsigned types
            unsigned pred_u = static_cast<unsigned>(predicted);
            unsigned delta_u = static_cast<unsigned>(static_cast<int>(delta));
            return static_cast<T>(pred_u + delta_u);
        } else {
            // For 8-bit unsigned types
            unsigned pred_u = static_cast<unsigned>(predicted);
            unsigned delta_u = static_cast<unsigned>(static_cast<int>(delta));
            return static_cast<T>(pred_u + delta_u);
        }
    }
}

// Helper function for delta extraction
template<typename T>
__device__ inline long long extractDelta(const uint32_t* delta_array, 
                                        int64_t bit_offset, 
                                        int delta_bits) {
    if (delta_bits <= 0) return 0;
    
    if (delta_bits <= 32) {
        int word_idx = bit_offset / 32;
        int bit_offset_in_word = bit_offset % 32;
        uint32_t extracted_bits;
        
        if (bit_offset_in_word + delta_bits <= 32) {
            extracted_bits = (delta_array[word_idx] >> bit_offset_in_word) & 
                            ((1U << delta_bits) - 1U);
        } else {
            uint32_t w1 = delta_array[word_idx];
            uint32_t w2 = delta_array[word_idx + 1]; 
            extracted_bits = (w1 >> bit_offset_in_word) | (w2 << (32 - bit_offset_in_word));
            extracted_bits &= ((1U << delta_bits) - 1U);
        }

        // Sign extension
        if (delta_bits < 32) {
            uint32_t sign_bit = 1U << (delta_bits - 1);
            if (extracted_bits & sign_bit) {
                uint32_t sign_extend_mask = ~((1U << delta_bits) - 1U);
                return static_cast<long long>(static_cast<int32_t>(extracted_bits | sign_extend_mask));
            } else {
                return static_cast<long long>(extracted_bits);
            }
        } else {
            return static_cast<long long>(static_cast<int32_t>(extracted_bits));
        }
    } else {
        // Handle > 32 bit deltas
        int start_word_idx = bit_offset / 32;
        int offset_in_word = bit_offset % 32;
        int bits_remaining = delta_bits;
        uint64_t extracted_val_64 = 0;
        int shift = 0;
        int word_idx = start_word_idx;
        
        while (bits_remaining > 0 && shift < 64) {
            int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
            uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
            uint32_t word_val = (delta_array[word_idx] >> offset_in_word) & mask;
            extracted_val_64 |= (static_cast<uint64_t>(word_val) << shift);
            
            shift += bits_in_this_word;
            bits_remaining -= bits_in_this_word;
            word_idx++;
            offset_in_word = 0;
        }
        
        // Sign extension for 64-bit deltas
        if (delta_bits < 64) {
            uint64_t sign_mask_64 = 1ULL << (delta_bits - 1);
            if (extracted_val_64 & sign_mask_64) {
                uint64_t sign_ext_mask_64 = ~((1ULL << delta_bits) - 1ULL);
                return static_cast<long long>(extracted_val_64 | sign_ext_mask_64);
            } else {
                return static_cast<long long>(extracted_val_64);
            }
        } else {
            return static_cast<long long>(extracted_val_64);
        }
    }
}

// Optimized delta extraction function that eliminates branching
// Optimized delta extraction function with reduced branching
template<typename T>
__device__ inline long long extractDelta_Optimized(const uint32_t* __restrict__ delta_array, 
                                                   int64_t bit_offset, 
                                                   int delta_bits) {
    if (delta_bits <= 0) return 0;
    
    // Use bit operations instead of division/modulo
    int word_idx = bit_offset >> 5;  // bit_offset / 32
    int bit_offset_in_word = bit_offset & 31;  // bit_offset % 32
    
    if (delta_bits <= 32) {
        // Always read two words to avoid branching
        uint32_t w1 = __ldg(&delta_array[word_idx]);
        uint32_t w2 = __ldg(&delta_array[word_idx + 1]);
        
        // Combine words into 64-bit value for branchless extraction
        uint64_t combined = (static_cast<uint64_t>(w2) << 32) | static_cast<uint64_t>(w1);
        uint32_t extracted_bits = (combined >> bit_offset_in_word) & ((1U << delta_bits) - 1U);
        
        // Branchless sign extension
        if (delta_bits < 32) {
            uint32_t sign_bit = extracted_bits >> (delta_bits - 1);
            uint32_t sign_mask = -sign_bit;  // All 1s if sign bit set, 0 otherwise
            uint32_t extend_mask = ~((1U << delta_bits) - 1U);
            extracted_bits |= (sign_mask & extend_mask);
        }
        
        return static_cast<long long>(static_cast<int32_t>(extracted_bits));
    } else {
        // Handle > 32 bit deltas with optimized loop
        uint64_t extracted_val_64 = 0;
        
        // First word
        uint32_t first_word = __ldg(&delta_array[word_idx]);
        int bits_from_first = 32 - bit_offset_in_word;
        extracted_val_64 = (first_word >> bit_offset_in_word);
        
        // Middle words (if any)
        int bits_remaining = delta_bits - bits_from_first;
        int shift = bits_from_first;
        word_idx++;
        
        // Unroll for common case of 64-bit values
        if (bits_remaining > 0) {
            uint32_t word = __ldg(&delta_array[word_idx]);
            if (bits_remaining >= 32) {
                extracted_val_64 |= (static_cast<uint64_t>(word) << shift);
                shift += 32;
                bits_remaining -= 32;
                word_idx++;
                
                if (bits_remaining > 0) {
                    word = __ldg(&delta_array[word_idx]);
                    uint32_t mask = (bits_remaining == 32) ? ~0U : ((1U << bits_remaining) - 1U);
                    extracted_val_64 |= (static_cast<uint64_t>(word & mask) << shift);
                }
            } else {
                uint32_t mask = (1U << bits_remaining) - 1U;
                extracted_val_64 |= (static_cast<uint64_t>(word & mask) << shift);
            }
        }
        
        // Branchless sign extension for 64-bit
        if (delta_bits < 64) {
            uint64_t sign_bit = extracted_val_64 >> (delta_bits - 1);
            uint64_t sign_mask = -(int64_t)sign_bit;
            uint64_t extend_mask = ~((1ULL << delta_bits) - 1ULL);
            extracted_val_64 |= (sign_mask & extend_mask);
        }
        
        return static_cast<long long>(extracted_val_64);
    }
}





// GPU-accelerated serialization kernel
__global__ void packToBlobKernel(
    const SerializedHeader* __restrict__ d_header,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const long long* __restrict__ d_error_bounds,
    const uint32_t* __restrict__ d_delta_array,
    int num_partitions,
    uint64_t delta_array_num_bytes,
    uint8_t* __restrict__ d_output_blob) {
    
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;
    
    // Get header from device memory
    SerializedHeader header;
    if (g_idx == 0) {
        header = *d_header;
    }
    __syncthreads();
    
    // Broadcast header to all threads (alternatively, each thread could read it)
    header = *d_header;
    
    // Total work items calculation
    uint64_t header_bytes = sizeof(SerializedHeader);
    uint64_t start_indices_bytes = num_partitions * sizeof(int32_t);
    uint64_t end_indices_bytes = num_partitions * sizeof(int32_t);
    uint64_t model_types_bytes = num_partitions * sizeof(int32_t);
    uint64_t model_params_bytes = num_partitions * 4 * sizeof(double);
    uint64_t delta_bits_bytes = num_partitions * sizeof(int32_t);
    uint64_t bit_offsets_bytes = num_partitions * sizeof(int64_t);
    uint64_t error_bounds_bytes = num_partitions * sizeof(long long);
    
    // Grid-stride loop to handle all copying tasks
    for (uint64_t idx = g_idx; idx < header.delta_array_offset + delta_array_num_bytes; idx += g_stride) {
        
        // Copy header
        if (idx < header_bytes) {
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_header)[idx];
        }
        // Copy start_indices
        else if (idx >= header.start_indices_offset && 
                 idx < header.start_indices_offset + start_indices_bytes) {
            uint64_t local_idx = idx - header.start_indices_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_start_indices)[local_idx];
        }
        // Copy end_indices
        else if (idx >= header.end_indices_offset && 
                 idx < header.end_indices_offset + end_indices_bytes) {
            uint64_t local_idx = idx - header.end_indices_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_end_indices)[local_idx];
        }
        // Copy model_types
        else if (idx >= header.model_types_offset && 
                 idx < header.model_types_offset + model_types_bytes) {
            uint64_t local_idx = idx - header.model_types_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_model_types)[local_idx];
        }
        // Copy model_params
        else if (idx >= header.model_params_offset && 
                 idx < header.model_params_offset + model_params_bytes) {
            uint64_t local_idx = idx - header.model_params_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_model_params)[local_idx];
        }
        // Copy delta_bits
        else if (idx >= header.delta_bits_offset && 
                 idx < header.delta_bits_offset + delta_bits_bytes) {
            uint64_t local_idx = idx - header.delta_bits_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_delta_bits)[local_idx];
        }
        // Copy delta_array_bit_offsets
        else if (idx >= header.delta_array_bit_offsets_offset && 
                 idx < header.delta_array_bit_offsets_offset + bit_offsets_bytes) {
            uint64_t local_idx = idx - header.delta_array_bit_offsets_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_delta_array_bit_offsets)[local_idx];
        }
        // Copy error_bounds
        else if (idx >= header.error_bounds_offset && 
                 idx < header.error_bounds_offset + error_bounds_bytes) {
            uint64_t local_idx = idx - header.error_bounds_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_error_bounds)[local_idx];
        }
        // Copy delta_array
        else if (idx >= header.delta_array_offset && 
                 idx < header.delta_array_offset + delta_array_num_bytes) {
            uint64_t local_idx = idx - header.delta_array_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_delta_array)[local_idx];
        }
    }
}

// GPU-accelerated deserialization kernel
__global__ void unpackFromBlobKernel(
    const uint8_t* __restrict__ d_input_blob,
    int num_partitions,
    uint64_t delta_array_num_bytes,
    int32_t* __restrict__ d_start_indices,
    int32_t* __restrict__ d_end_indices,
    int32_t* __restrict__ d_model_types,
    double* __restrict__ d_model_params,
    int32_t* __restrict__ d_delta_bits,
    int64_t* __restrict__ d_delta_array_bit_offsets,
    long long* __restrict__ d_error_bounds,
    uint32_t* __restrict__ d_delta_array) {
    
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;
    
    // Get header from the beginning of the blob
    const SerializedHeader* header = reinterpret_cast<const SerializedHeader*>(d_input_blob);
    
    // Calculate only what we need
    uint64_t delta_array_words = (delta_array_num_bytes + 3) / 4;
    
    // Grid-stride loop for parallel unpacking
    // Unpack start_indices
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_start_indices[i] = *reinterpret_cast<const int32_t*>(
            d_input_blob + header->start_indices_offset + i * sizeof(int32_t));
    }
    
    // Unpack end_indices
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_end_indices[i] = *reinterpret_cast<const int32_t*>(
            d_input_blob + header->end_indices_offset + i * sizeof(int32_t));
    }
    
    // Unpack model_types
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_model_types[i] = *reinterpret_cast<const int32_t*>(
            d_input_blob + header->model_types_offset + i * sizeof(int32_t));
    }
    
    // Unpack model_params (4 per partition)
    for (int i = g_idx; i < num_partitions * 4; i += g_stride) {
        d_model_params[i] = *reinterpret_cast<const double*>(
            d_input_blob + header->model_params_offset + i * sizeof(double));
    }
    
    // Unpack delta_bits
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_delta_bits[i] = *reinterpret_cast<const int32_t*>(
            d_input_blob + header->delta_bits_offset + i * sizeof(int32_t));
    }
    
    // Unpack delta_array_bit_offsets
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_delta_array_bit_offsets[i] = *reinterpret_cast<const int64_t*>(
            d_input_blob + header->delta_array_bit_offsets_offset + i * sizeof(int64_t));
    }
    
    // Unpack error_bounds
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_error_bounds[i] = *reinterpret_cast<const long long*>(
            d_input_blob + header->error_bounds_offset + i * sizeof(long long));
    }
    
    // Unpack delta_array (word by word for alignment)
    for (uint64_t i = g_idx; i < delta_array_words; i += g_stride) {
        if (i * sizeof(uint32_t) < delta_array_num_bytes) {
            d_delta_array[i] = *reinterpret_cast<const uint32_t*>(
                d_input_blob + header->delta_array_offset + i * sizeof(uint32_t));
        }
    }
}

// Optimized version using cooperative groups for better memory coalescing
__global__ void packToBlobKernelOptimized(
    const SerializedHeader* __restrict__ d_header,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const long long* __restrict__ d_error_bounds,
    const uint32_t* __restrict__ d_delta_array,
    int num_partitions,
    uint64_t delta_array_num_bytes,
    uint8_t* __restrict__ d_output_blob) {
    
    // Use shared memory for coalesced writes
    extern __shared__ uint8_t s_buffer[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Each block handles a specific section
    int section_id = blockIdx.x;
    
    // Get header
    SerializedHeader header = *d_header;
    
    // Section 0: Header copy
    if (section_id == 0) {
        const uint8_t* header_bytes = reinterpret_cast<const uint8_t*>(d_header);
        for (int i = tid; i < sizeof(SerializedHeader); i += block_size) {
            d_output_blob[i] = header_bytes[i];
        }
    }
    // Sections 1-7: Metadata arrays
    else if (section_id >= 1 && section_id <= 7) {
        uint64_t offset, num_bytes;
        const uint8_t* src_ptr;
        
        switch (section_id) {
            case 1: // start_indices
                offset = header.start_indices_offset;
                num_bytes = num_partitions * sizeof(int32_t);
                src_ptr = reinterpret_cast<const uint8_t*>(d_start_indices);
                break;
            case 2: // end_indices
                offset = header.end_indices_offset;
                num_bytes = num_partitions * sizeof(int32_t);
                src_ptr = reinterpret_cast<const uint8_t*>(d_end_indices);
                break;
            case 3: // model_types
                offset = header.model_types_offset;
                num_bytes = num_partitions * sizeof(int32_t);
                src_ptr = reinterpret_cast<const uint8_t*>(d_model_types);
                break;
            case 4: // model_params
                offset = header.model_params_offset;
                num_bytes = num_partitions * 4 * sizeof(double);
                src_ptr = reinterpret_cast<const uint8_t*>(d_model_params);
                break;
            case 5: // delta_bits
                offset = header.delta_bits_offset;
                num_bytes = num_partitions * sizeof(int32_t);
                src_ptr = reinterpret_cast<const uint8_t*>(d_delta_bits);
                break;
            case 6: // delta_array_bit_offsets
                offset = header.delta_array_bit_offsets_offset;
                num_bytes = num_partitions * sizeof(int64_t);
                src_ptr = reinterpret_cast<const uint8_t*>(d_delta_array_bit_offsets);
                break;
            case 7: // error_bounds
                offset = header.error_bounds_offset;
                num_bytes = num_partitions * sizeof(long long);
                src_ptr = reinterpret_cast<const uint8_t*>(d_error_bounds);
                break;
        }
        
        // Coalesced copy
        for (uint64_t i = tid; i < num_bytes; i += block_size) {
            d_output_blob[offset + i] = src_ptr[i];
        }
    }
    // Remaining sections: Delta array (distributed across multiple blocks)
    else {
        int delta_section = section_id - 8;
        uint64_t bytes_per_block = (delta_array_num_bytes + gridDim.x - 9) / (gridDim.x - 8);
        uint64_t start_byte = delta_section * bytes_per_block;
        uint64_t end_byte = min(start_byte + bytes_per_block, delta_array_num_bytes);
        
        const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(d_delta_array);
        for (uint64_t i = start_byte + tid; i < end_byte; i += block_size) {
            d_output_blob[header.delta_array_offset + i] = src_ptr[i];
        }
    }
}



// Combined kernel for overflow checking, model fitting, and metadata computation - UPDATED FOR SoA
template<typename T>
__global__ void wprocessPartitionsKernel(const T* values_device,
                                       int32_t* d_start_indices,
                                       int32_t* d_end_indices,
                                       int32_t* d_model_types,
                                       double* d_model_params,
                                       int32_t* d_delta_bits,
                                       long long* d_error_bounds,
                                       int num_partitions,
                                       int64_t* total_bits_device) {
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;
    
    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;
    
    if (segment_len <= 0) return;
    
    // Shared memory for reduction operations
    extern __shared__ char shared_mem[];
    double* s_sums = reinterpret_cast<double*>(shared_mem);
    long long* s_max_error = reinterpret_cast<long long*>(shared_mem + 4 * blockDim.x * sizeof(double));
    bool* s_overflow = reinterpret_cast<bool*>(shared_mem + 4 * blockDim.x * sizeof(double) + blockDim.x * sizeof(long long));
    
    int tid = threadIdx.x;
    
    // Phase 1: Check for overflow
    bool local_overflow = false;
    for (int i = tid; i < segment_len; i += blockDim.x) {
        if (mightOverflowDoublePrecision(values_device[start_idx + i])) {
            local_overflow = true;
            break;
        }
    }
    
    // Reduce overflow flag
    s_overflow[tid] = local_overflow;
    __syncthreads();
    
    // Simple reduction for overflow flag
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_overflow[tid] = s_overflow[tid] || s_overflow[tid + s];
        }
        __syncthreads();
    }
    
    bool has_overflow = s_overflow[0];
    
    if (tid == 0) {
        if (has_overflow) {
            // Direct copy model for overflow
            d_model_types[partition_idx] = MODEL_DIRECT_COPY;
            d_model_params[partition_idx * 4] = 0.0;
            d_model_params[partition_idx * 4 + 1] = 0.0;
            d_model_params[partition_idx * 4 + 2] = 0.0;
            d_model_params[partition_idx * 4 + 3] = 0.0;
            d_error_bounds[partition_idx] = 0;
            d_delta_bits[partition_idx] = sizeof(T) * 8;
        }
    }
    __syncthreads();
    
    if (!has_overflow) {
        // Phase 2: Fit linear model
        double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
        
        for (int i = tid; i < segment_len; i += blockDim.x) {
            double x = static_cast<double>(i);
            double y = static_cast<double>(values_device[start_idx + i]);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }
        
        s_sums[tid] = sum_x;
        s_sums[tid + blockDim.x] = sum_y;
        s_sums[tid + 2 * blockDim.x] = sum_xx;
        s_sums[tid + 3 * blockDim.x] = sum_xy;
        __syncthreads();
        
        // Reduction for sums
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sums[tid] += s_sums[tid + s];
                s_sums[tid + blockDim.x] += s_sums[tid + s + blockDim.x];
                s_sums[tid + 2 * blockDim.x] += s_sums[tid + s + 2 * blockDim.x];
                s_sums[tid + 3 * blockDim.x] += s_sums[tid + s + 3 * blockDim.x];
            }
            __syncthreads();
        }
        
        __shared__ double theta0, theta1;
        
        if (tid == 0) {
            double n = static_cast<double>(segment_len);
            double determinant = n * s_sums[2 * blockDim.x] - s_sums[0] * s_sums[0];
            
            if (fabs(determinant) > 1e-10) {
                theta1 = (n * s_sums[3 * blockDim.x] - s_sums[0] * s_sums[blockDim.x]) / determinant;
                theta0 = (s_sums[blockDim.x] - theta1 * s_sums[0]) / n;
            } else {
                theta1 = 0.0;
                theta0 = s_sums[blockDim.x] / n;
            }
            
            d_model_types[partition_idx] = MODEL_LINEAR;
            d_model_params[partition_idx * 4] = theta0;
            d_model_params[partition_idx * 4 + 1] = theta1;
            d_model_params[partition_idx * 4 + 2] = 0.0;
            d_model_params[partition_idx * 4 + 3] = 0.0;
        }
        __syncthreads();
        
        theta0 = d_model_params[partition_idx * 4];
        theta1 = d_model_params[partition_idx * 4 + 1];
        
        // Phase 3: Calculate maximum error
        long long max_error = 0;
        
        for (int i = tid; i < segment_len; i += blockDim.x) {
            double predicted = theta0 + theta1 * i;
            T pred_T = static_cast<T>(round(predicted));
            long long delta = calculateDelta(values_device[start_idx + i], pred_T);
            long long abs_error = (delta < 0) ? -delta : delta;
            if (abs_error > max_error) {
                max_error = abs_error;
            }
        }
        
        s_max_error[tid] = max_error;
        __syncthreads();
        
        // Reduction for maximum error
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_max_error[tid + s] > s_max_error[tid]) {
                    s_max_error[tid] = s_max_error[tid + s];
                }
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            d_error_bounds[partition_idx] = s_max_error[0];
            
            // Calculate delta bits
            int delta_bits = 0;
            if (s_max_error[0] > 0) {
                long long max_abs_error = s_max_error[0];
                int bits_for_magnitude = 0;
                unsigned long long temp = static_cast<unsigned long long>(max_abs_error);
                while (temp > 0) {
                    bits_for_magnitude++;
                    temp >>= 1;
                }
                delta_bits = bits_for_magnitude + 1; // +1 for sign bit
                delta_bits = min(delta_bits, MAX_DELTA_BITS);
                delta_bits = max(delta_bits, 0);
            }
            d_delta_bits[partition_idx] = delta_bits;
        }
    }
    
    // Atomic add to total bits counter
    if (tid == 0) {
        int64_t partition_bits = (int64_t)segment_len * d_delta_bits[partition_idx];
        // Use unsigned long long atomicAdd and cast
        atomicAdd(reinterpret_cast<unsigned long long*>(total_bits_device), 
                  static_cast<unsigned long long>(partition_bits));
    }
}

// Kernel to set bit offsets based on cumulative sum - UPDATED FOR SoA
__global__ void setBitOffsetsKernel(int32_t* d_start_indices,
                                   int32_t* d_end_indices,
                                   int32_t* d_delta_bits,
                                   int64_t* d_delta_array_bit_offsets,
                                   int num_partitions) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_partitions) return;
    
    // Calculate cumulative bit offset
    int64_t bit_offset = 0;
    for (int i = 0; i < tid; i++) {
        int seg_len = d_end_indices[i] - d_start_indices[i];
        bit_offset += (int64_t)seg_len * d_delta_bits[i];
    }
    
    d_delta_array_bit_offsets[tid] = bit_offset;
}

// Extract value for direct copy model
template<typename T>
__device__ inline T extractDirectValue(const uint32_t* delta_array, 
                                      int64_t bit_offset, 
                                      int value_bits) {
    if (value_bits <= 0) return static_cast<T>(0);
    
    if (value_bits <= 32) {
        int word_idx = bit_offset / 32;
        int bit_offset_in_word = bit_offset % 32;
        uint32_t extracted_bits;
        
        if (bit_offset_in_word + value_bits <= 32) {
            extracted_bits = (delta_array[word_idx] >> bit_offset_in_word) & 
                            ((1U << value_bits) - 1U);
        } else {
            uint32_t w1 = delta_array[word_idx];
            uint32_t w2 = delta_array[word_idx + 1]; 
            extracted_bits = (w1 >> bit_offset_in_word) | (w2 << (32 - bit_offset_in_word));
            extracted_bits &= ((1U << value_bits) - 1U);
        }
        
        return static_cast<T>(extracted_bits);
    } else {
        // Handle > 32 bit values
        int start_word_idx = bit_offset / 32;
        int offset_in_word = bit_offset % 32;
        int bits_remaining = value_bits;
        uint64_t extracted_val_64 = 0;
        int shift = 0;
        int word_idx = start_word_idx;
        
        while (bits_remaining > 0 && shift < 64) {
            int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
            uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
            uint32_t word_val = (delta_array[word_idx] >> offset_in_word) & mask;
            extracted_val_64 |= (static_cast<uint64_t>(word_val) << shift);
            
            shift += bits_in_this_word;
            bits_remaining -= bits_in_this_word;
            word_idx++;
            offset_in_word = 0;
        }
        
        return static_cast<T>(extracted_val_64);
    }
}


// 383 -------------------------------------------------------

// Kernel to pre-unpack all deltas from bit-packed format to plain long long array
template<typename T>
__global__ void unpackAllDeltasKernel(
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const uint32_t* __restrict__ delta_array,
    long long* __restrict__ d_plain_deltas_output,
    int num_partitions,
    int total_values) {
    
    // Grid-stride loop
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;
    
    for (int idx = g_idx; idx < total_values; idx += g_stride) {
        // Binary search to find the partition
        int p_left = 0, p_right = num_partitions - 1;
        int p_found_idx = -1;
        
        while (p_left <= p_right) {
            int p_mid = p_left + (p_right - p_left) / 2;
            int32_t current_start = d_start_indices[p_mid];
            int32_t current_end = d_end_indices[p_mid];
            
            if (idx >= current_start && idx < current_end) {
                p_found_idx = p_mid;
                break;
            } else if (idx < current_start) {
                p_right = p_mid - 1;
            } else {
                p_left = p_mid + 1;
            }
        }
        
        if (p_found_idx == -1) {
            d_plain_deltas_output[idx] = 0;
            continue;
        }
        
        // Extract delta for this element
        int32_t start_idx = d_start_indices[p_found_idx];
        int32_t model_type = d_model_types[p_found_idx];
        int32_t delta_bits = d_delta_bits[p_found_idx];
        int64_t bit_offset_base = d_delta_array_bit_offsets[p_found_idx];
        int local_idx = idx - start_idx;
        
        long long delta = 0;
        
        if (model_type == MODEL_DIRECT_COPY) {
            // For direct copy model, extract the full value (not a delta)
            if (delta_bits > 0 && delta_array) {
                int64_t bit_offset = bit_offset_base + (int64_t)local_idx * delta_bits;
                // We store the raw value as the "delta" for direct copy
                delta = static_cast<long long>(extractDirectValue<T>(delta_array, bit_offset, delta_bits));
            }
        } else {
            // Normal delta extraction
            if (delta_bits > 0 && delta_array) {
                int64_t bit_offset = bit_offset_base + (int64_t)local_idx * delta_bits;
                delta = extractDelta_Optimized<T>(delta_array, bit_offset, delta_bits);
            }
        }
        
        d_plain_deltas_output[idx] = delta;
    }
}
// 383 -------------------------------------------------------

// Optimized delta packing kernel with direct copy support - UPDATED FOR SoA
template<typename T>
__global__ void packDeltasKernelOptimized(const T* values_device,
                                          const int32_t* d_start_indices,
                                          const int32_t* d_end_indices,
                                          const int32_t* d_model_types,
                                          const double* d_model_params,
                                          const int32_t* d_delta_bits,
                                          const int64_t* d_delta_array_bit_offsets,
                                          int num_partitions_val,
                                          uint32_t* delta_array_device) {
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;

    if (num_partitions_val == 0) return;

    int max_idx_to_process = d_end_indices[num_partitions_val - 1];

    for (int current_idx = g_idx; current_idx < max_idx_to_process; current_idx += g_stride) {
        // Binary search for partition
        int p_left = 0, p_right = num_partitions_val - 1;
        int found_partition_idx = -1;

        while (p_left <= p_right) {
            int p_mid = p_left + (p_right - p_left) / 2;
            int32_t current_start = d_start_indices[p_mid];
            int32_t current_end = d_end_indices[p_mid];
            
            if (current_idx >= current_start && current_idx < current_end) {
                found_partition_idx = p_mid; 
                break;
            } else if (current_idx < current_start) {
                p_right = p_mid - 1;
            } else {
                p_left = p_mid + 1;
            }
        }

        if (found_partition_idx == -1) continue;

        // Get partition data using found index
        int32_t current_model_type = d_model_types[found_partition_idx];
        int32_t current_delta_bits = d_delta_bits[found_partition_idx];
        int64_t current_bit_offset_base = d_delta_array_bit_offsets[found_partition_idx];
        int32_t current_start_idx = d_start_indices[found_partition_idx];

        // For direct copy model, we store the full value
        if (current_model_type == MODEL_DIRECT_COPY) {
            int local_idx = current_idx - current_start_idx;
            int64_t bit_offset = current_bit_offset_base + 
                                (int64_t)local_idx * current_delta_bits;
            
            // Store the full value as "delta"
            T value = values_device[current_idx];
            uint64_t value_to_store = static_cast<uint64_t>(value);
            
            // Pack the value into the delta array
            int start_word_idx = bit_offset / 32;
            int offset_in_word = bit_offset % 32;
            int bits_remaining = current_delta_bits;
            int word_idx = start_word_idx;
            
            while (bits_remaining > 0) {
                int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
                uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                uint32_t value_part = (value_to_store & mask) << offset_in_word;
                atomicOr(&delta_array_device[word_idx], value_part);
                
                value_to_store >>= bits_in_this_word;
                bits_remaining -= bits_in_this_word;
                word_idx++;
                offset_in_word = 0;
            }
        } else {
            // Normal delta encoding
            int current_local_idx = current_idx - current_start_idx;

            double pred_double = d_model_params[found_partition_idx * 4] + 
                                d_model_params[found_partition_idx * 4 + 1] * current_local_idx;
            if (current_model_type == MODEL_POLYNOMIAL2) {
                pred_double += d_model_params[found_partition_idx * 4 + 2] * current_local_idx * current_local_idx;
            }

            T pred_T_val = static_cast<T>(round(pred_double));
            long long current_delta_ll = calculateDelta(values_device[current_idx], pred_T_val);

            if (current_delta_bits > 0) {
                int64_t current_bit_offset_val = current_bit_offset_base + 
                                                 (int64_t)current_local_idx * current_delta_bits;
                
                // Handle deltas up to 64 bits
                if (current_delta_bits <= 32) {
                    uint32_t final_packed_delta = static_cast<uint32_t>(current_delta_ll & 
                                                                       ((1ULL << current_delta_bits) - 1ULL));
                    
                    int target_word_idx = current_bit_offset_val / 32;
                    int offset_in_word = current_bit_offset_val % 32;

                    if (current_delta_bits + offset_in_word <= 32) {
                        atomicOr(&delta_array_device[target_word_idx], final_packed_delta << offset_in_word);
                    } else {
                        atomicOr(&delta_array_device[target_word_idx], final_packed_delta << offset_in_word); 
                        atomicOr(&delta_array_device[target_word_idx + 1], 
                                final_packed_delta >> (32 - offset_in_word));
                    }
                } else {
                    // For deltas > 32 bits
                    uint64_t final_packed_delta_64 = static_cast<uint64_t>(current_delta_ll & 
                        ((current_delta_bits == 64) ? ~0ULL : ((1ULL << current_delta_bits) - 1ULL)));
                    
                    int start_word_idx = current_bit_offset_val / 32;
                    int offset_in_word = current_bit_offset_val % 32;
                    int bits_remaining = current_delta_bits;
                    int word_idx = start_word_idx;
                    uint64_t delta_to_write = final_packed_delta_64;
                    
                    while (bits_remaining > 0) {
                        int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
                        uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                        uint32_t value = (delta_to_write & mask) << offset_in_word;
                        atomicOr(&delta_array_device[word_idx], value);
                        
                        delta_to_write >>= bits_in_this_word;
                        bits_remaining -= bits_in_this_word;
                        word_idx++;
                        offset_in_word = 0;
                    }
                }
            }
        }
    }
}



template<typename T>
__global__ void decompressFullFile_OnTheFly_Optimized_V2(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements) {

    // Use shared memory to cache the metadata for the partition this block is responsible for.
    __shared__ PartitionMetaOpt s_meta;

    // Map the CUDA block directly to a partition index.
    int partition_idx = blockIdx.x;

    // Ensure this block is not assigned to a non-existent partition.
    if (partition_idx >= compressed_data->num_partitions) {
        return;
    }

    // A single thread (thread 0) in the block loads the partition's metadata from global memory
    // into the fast shared memory. This is done once per block.
    if (threadIdx.x == 0) {
        s_meta.start_idx = compressed_data->d_start_indices[partition_idx];
        s_meta.model_type = compressed_data->d_model_types[partition_idx];
        s_meta.delta_bits = compressed_data->d_delta_bits[partition_idx];
        s_meta.bit_offset_base = compressed_data->d_delta_array_bit_offsets[partition_idx];
        
        // Load model parameters for the linear model.
        int params_base_idx = partition_idx * 4;
        s_meta.theta0 = compressed_data->d_model_params[params_base_idx];
        s_meta.theta1 = compressed_data->d_model_params[params_base_idx + 1];

        // Also cache the length of the partition.
        s_meta.partition_len = compressed_data->d_end_indices[partition_idx] - s_meta.start_idx;
    }

    // Synchronize all threads within the block to ensure that the shared memory is populated
    // before any thread attempts to use it.
    __syncthreads();

    // Use a grid-stride loop where each thread processes elements within this partition.
    // This ensures that all memory accesses are localized and coalesced.
    for (int local_idx = threadIdx.x; local_idx < s_meta.partition_len; local_idx += blockDim.x) {
        
        int global_idx = s_meta.start_idx + local_idx;

        // Ensure we don't write past the end of the total data array.
        if (global_idx >= total_elements) continue;

        long long delta = 0;
        T final_value;

        // Check the model type from shared memory.
        if (s_meta.model_type == MODEL_DIRECT_COPY) {
            // For direct copy, the "delta" array actually stores the full value.
            if (compressed_data->d_plain_deltas != nullptr) {
                // High-throughput mode: value is already unpacked.
                final_value = static_cast<T>(compressed_data->d_plain_deltas[global_idx]);
            } else if (s_meta.delta_bits > 0 && compressed_data->delta_array != nullptr) {
                // Standard mode: extract the value from the bit-packed array.
                int64_t bit_offset = s_meta.bit_offset_base + (int64_t)local_idx * s_meta.delta_bits;
                final_value = extractDirectValue<T>(compressed_data->delta_array, bit_offset, s_meta.delta_bits);
            } else {
                final_value = static_cast<T>(0);
            }
        } else {
            // For model-based decompression (e.g., MODEL_LINEAR)
            // Calculate the predicted value using the cached model parameters from shared memory.
            double predicted_double = fma(s_meta.theta1, static_cast<double>(local_idx), s_meta.theta0);
            
            // Extract the delta value.
            if (compressed_data->d_plain_deltas != nullptr) {
                 // High-throughput mode: delta is already unpacked.
                 delta = compressed_data->d_plain_deltas[global_idx];
            } else if (s_meta.delta_bits > 0 && compressed_data->delta_array != nullptr) {
                // Standard mode: calculate the bit offset and extract from the bit-packed array.
                int64_t bit_offset = s_meta.bit_offset_base + (int64_t)local_idx * s_meta.delta_bits;
                delta = extractDelta_Optimized<T>(compressed_data->delta_array, bit_offset, s_meta.delta_bits);
            }

            // Round prediction and apply the delta to get the final value.
            T predicted_T = static_cast<T>(round(predicted_double));
            final_value = applyDelta(predicted_T, delta);
        }

        // Write the final, perfectly coalesced result to global memory.
        output_device[global_idx] = final_value;
    }
}





// Define this constant before the kernels
const double PARTITION_MODEL_SIZE_BYTES = sizeof(PartitionInfo);



// Helper function for warp reduction - sum
__device__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Helper function for warp reduction - max
__device__ long long warpReduceMax(long long val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Helper for block reduction - sum
__device__ double blockReduceSum(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Helper for block reduction - max
__device__ long long blockReduceMax(long long val) {
    __shared__ long long shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    val = warpReduceMax(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceMax(val);
    
    return val;
}


// Optimized variance calculation using grid-stride loops and better parallelism
template<typename T>
__global__ void analyzeDataVarianceFast(
    const T* __restrict__ data,
    int data_size,
    int block_size,
    float* __restrict__ variances,
    int num_blocks) {
    
    // Grid-stride loop for better GPU utilization
    for (int bid = blockIdx.x; bid < num_blocks; bid += gridDim.x) {
        int start = bid * block_size;
        int end = min(start + block_size, data_size);
        int n = end - start;
        
        if (n <= 0) continue;
        
        // Use Kahan summation for better numerical stability
        double sum = 0.0;
        double sum_sq = 0.0;
        double c1 = 0.0, c2 = 0.0;
        
        // Coalesced access with grid-stride
        for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
            double val = static_cast<double>(data[i]);
            
            // Kahan summation
            double y1 = val - c1;
            double t1 = sum + y1;
            c1 = (t1 - sum) - y1;
            sum = t1;
            
            double y2 = val * val - c2;
            double t2 = sum_sq + y2;
            c2 = (t2 - sum_sq) - y2;
            sum_sq = t2;
        }
        
        // Warp reduction
        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);
        
        // Write result by first thread of each warp
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&variances[bid], static_cast<float>(sum_sq / n - (sum / n) * (sum / n)));
        }
    }
}

// MODIFICATION 1: Kernel signature changed to be more generic
// Fast partition creation with pre-computed thresholds
template<typename T>
__global__ void createPartitionsFast(
    int data_size,
    int base_size,
    const float* __restrict__ variances,
    int num_variance_blocks,
    int* __restrict__ partition_starts,
    int* __restrict__ partition_ends,
    int* __restrict__ num_partitions,
    const float* __restrict__ variance_thresholds,
    const int* __restrict__ partition_sizes_for_buckets, // New parameter
    int num_thresholds, // New parameter
    int variance_block_multiplier) // New parameter
{
    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < num_variance_blocks; 
         i += blockDim.x * gridDim.x) {
        
        float var = variances[i];
        int block_start = i * base_size * variance_block_multiplier; // Use multiplier
        int block_end = min(block_start + base_size * variance_block_multiplier, data_size); // Use multiplier
        
        // MODIFICATION 2: Dynamic partition size decision based on thresholds
        int partition_size = partition_sizes_for_buckets[num_thresholds]; // Default to smallest size
        for (int j = 0; j < num_thresholds; ++j) {
            if (var < variance_thresholds[j]) {
                partition_size = partition_sizes_for_buckets[j];
                break;
            }
        }
        
        // Create partitions
        if (partition_size > 0) { // Safety check
            for (int j = block_start; j < block_end; j += partition_size) {
                if (j < data_size) {
                    int idx = atomicAdd(num_partitions, 1);
                    // Increased safety check size
                    if (idx < data_size / MIN_PARTITION_SIZE) { 
                        partition_starts[idx] = j;
                        partition_ends[idx] = min(j + partition_size, data_size);
                    }
                }
            }
        }
    }
}


// --- 核心优化内核: "一个Block处理一个Partition" ---
template<typename T>
__global__ void fitPartitionsBatched_Optimized(
    const T* __restrict__ data,
    const int* __restrict__ partition_starts,
    const int* __restrict__ partition_ends,
    int* __restrict__ model_types,
    double* __restrict__ theta0_array,
    double* __restrict__ theta1_array,
    int* __restrict__ delta_bits_array,
    long long* __restrict__ max_errors,
    double* __restrict__ costs,
    int num_partitions)
{
    // **架构优化**: 每个Block直接通过 blockIdx.x 获取自己的分区ID
    const int pid = blockIdx.x;
    if (pid >= num_partitions) {
        return; // 边界检查
    }

    // 为这个Block内共享的数据声明静态共享内存
    __shared__ double s_theta0;
    __shared__ double s_theta1;
    __shared__ int s_has_overflow_flag;

    const int start = partition_starts[pid];
    const int end = partition_ends[pid];
    const int n = end - start;

    // --- 阶段1: 检查分区是否为空或存在溢出风险 ---
    if (threadIdx.x == 0) {
        s_has_overflow_flag = false;
    }
     __syncthreads();
     
    if (n <= 0) {
        if (threadIdx.x == 0) {
            model_types[pid] = MODEL_DIRECT_COPY;
            costs[pid] = 0.0;
        }
        return;
    }

    bool local_overflow = false;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (mightOverflowDoublePrecision(data[start + i])) {
            local_overflow = true;
            break;
        }
    }

    if (local_overflow) {
        atomicExch(&s_has_overflow_flag, true);
    }
    __syncthreads();

    if (s_has_overflow_flag) {
        if (threadIdx.x == 0) {
            model_types[pid] = MODEL_DIRECT_COPY;
            theta0_array[pid] = 0.0;
            theta1_array[pid] = 0.0;
            delta_bits_array[pid] = sizeof(T) * 8;
            max_errors[pid] = 0;
            costs[pid] = PARTITION_MODEL_SIZE_BYTES + n * sizeof(T);
        }
        return;
    }
    
    // --- 阶段2: 快速线性回归计算 ---
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(data[start + i]);
        sum_x += x;
        sum_y += y;
        // **FMA优化**: 将 x*x + sum_xx 合并为单条 fma 指令
        sum_xx = fma(x, x, sum_xx);
        sum_xy = fma(x, y, sum_xy);
    }

    // 使用共享内存进行高效的块内归约
    sum_x = blockReduceSum(sum_x);
    sum_y = blockReduceSum(sum_y);
    sum_xx = blockReduceSum(sum_xx);
    sum_xy = blockReduceSum(sum_xy);

    // --- 阶段3: 计算模型参数 (仅由线程0完成) ---
    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);
        // **FMA优化**: dn * sum_xx - sum_x * sum_x
        double determinant = fma(dn, sum_xx, -(sum_x * sum_x));
        
        if (fabs(determinant) > 1e-10) {
            // **FMA优化**: dn * sum_xy - sum_x * sum_y
            s_theta1 = fma(dn, sum_xy, -(sum_x * sum_y)) / determinant;
            // **FMA优化**: sum_y - s_theta1 * sum_x
            s_theta0 = fma(-s_theta1, sum_x, sum_y) / dn;
        } else {
            s_theta1 = 0.0;
            s_theta0 = sum_y / dn;
        }
        model_types[pid] = MODEL_LINEAR;
        theta0_array[pid] = s_theta0;
        theta1_array[pid] = s_theta1;
    }
    __syncthreads();

    // --- 阶段4: 计算最大误差 ---
    double theta0 = theta0_array[pid];
    double theta1 = theta1_array[pid];
    long long local_max_error = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        // **FMA优化**: theta1 * i + theta0
        double predicted = fma(theta1, static_cast<double>(i), theta0);
        T pred_T = static_cast<T>(round(predicted));
        long long delta = calculateDelta(data[start + i], pred_T);
        local_max_error = max(local_max_error, llabs(delta));
    }

    long long partition_max_error = blockReduceMax(local_max_error);

    // --- 阶段5: 写回最终结果 (仅由线程0完成) ---
    if (threadIdx.x == 0) {
        max_errors[pid] = partition_max_error;
        
        int delta_bits = 0;
        if (partition_max_error > 0) {
            delta_bits = 64 - __clzll(static_cast<unsigned long long>(partition_max_error)) + 1;
        }
        delta_bits_array[pid] = delta_bits;
        
        double delta_array_bytes = static_cast<double>(n) * delta_bits / 8.0;
        costs[pid] = PARTITION_MODEL_SIZE_BYTES + delta_array_bytes;
    }
}



// Optimized GPU partitioner V6 focused on speed
template<typename T>
class GPUVariableLengthPartitionerV6 {
private:
    T* d_data;
    int data_size;
    int base_partition_size;
    cudaStream_t stream;
    // MODIFICATION 3: Added member variables for new parameters
    int variance_block_multiplier;
    int num_thresholds;
    
public:
    // MODIFICATION 4: Constructor updated to accept new parameters
    GPUVariableLengthPartitionerV6(const std::vector<T>& data,
                                   int base_size = 1024,
                                   cudaStream_t cuda_stream = 0,
                                   int multiplier = 8,
                                   int thresholds = 3)
        : data_size(data.size()), 
          base_partition_size(base_size), 
          stream(cuda_stream),
          variance_block_multiplier(multiplier),
          num_thresholds(thresholds)
    {
        // Ensure num_thresholds is at least 1
        if (this->num_thresholds < 1) {
            this->num_thresholds = 1;
        }
        CUDA_CHECK(cudaMalloc(&d_data, data_size * sizeof(T)));
        CUDA_CHECK(cudaMemcpyAsync(d_data, data.data(), data_size * sizeof(T), 
                                  cudaMemcpyHostToDevice, stream));
    }
    
    ~GPUVariableLengthPartitionerV6() {
        if (d_data) CUDA_CHECK(cudaFree(d_data));
    }
    
    std::vector<PartitionInfo> partition() {
        if (data_size == 0) return std::vector<PartitionInfo>();
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        int sm_count = prop.multiProcessorCount;
        
        // MODIFICATION 5: Use member variable for variance block size calculation
        int variance_block_size = base_partition_size * variance_block_multiplier;
        int num_variance_blocks = (data_size + variance_block_size - 1) / variance_block_size;
        float* d_variances;
        float* d_variance_thresholds;
        
        CUDA_CHECK(cudaMalloc(&d_variances, num_variance_blocks * sizeof(float)));
        // MODIFICATION 6: Allocate memory for thresholds dynamically
        CUDA_CHECK(cudaMalloc(&d_variance_thresholds, num_thresholds * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_variances, 0, num_variance_blocks * sizeof(float), stream));
        
        int threads = 128;
        int blocks = min(num_variance_blocks, sm_count * 4);
        
        analyzeDataVarianceFast<T><<<blocks, threads, 0, stream>>>(
            d_data, data_size, variance_block_size, d_variances, num_variance_blocks);
        
        thrust::device_ptr<float> var_ptr(d_variances);
        if (num_variance_blocks > 1) {
           thrust::sort(var_ptr, var_ptr + num_variance_blocks);
        }
        
        // MODIFICATION 7: Calculate thresholds in a loop based on num_thresholds
        std::vector<float> h_thresholds(num_thresholds);
        for (int i = 0; i < num_thresholds; ++i) {
            long long idx = (long long)(i + 1) * num_variance_blocks / (num_thresholds + 1);
            if (idx >= num_variance_blocks) idx = num_variance_blocks - 1;
            if (idx < 0) idx = 0;
            // A Thrust device_ptr can be indexed like a regular pointer after sync
            h_thresholds[i] = (num_variance_blocks > 0) ? var_ptr[idx] : 0.0f;
        }
        
        CUDA_CHECK(cudaMemcpyAsync(d_variance_thresholds, h_thresholds.data(), 
                                  num_thresholds * sizeof(float), cudaMemcpyHostToDevice, stream));
        
        // MODIFICATION 8: Pre-calculate partition sizes for each bucket
        std::vector<int> h_partition_sizes_for_buckets(num_thresholds + 1);
        int min_partition_size_val = base_partition_size;
        for (int i = 0; i <= num_thresholds; ++i) {
            // Creates a geometric progression of sizes, e.g., base*4, base*2, base, base/2
            int shift = (num_thresholds / 2) - i;
            h_partition_sizes_for_buckets[i] = std::max(MIN_PARTITION_SIZE, base_partition_size << shift);
            if (h_partition_sizes_for_buckets[i] < min_partition_size_val) {
                min_partition_size_val = h_partition_sizes_for_buckets[i];
            }
        }
        
        int* d_partition_sizes_for_buckets;
        CUDA_CHECK(cudaMalloc(&d_partition_sizes_for_buckets, (num_thresholds + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(d_partition_sizes_for_buckets, h_partition_sizes_for_buckets.data(),
                                  (num_thresholds + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));

        // MODIFICATION 9: Estimate partitions based on the calculated smallest size
        int estimated_partitions = (min_partition_size_val > 0) ? (data_size / min_partition_size_val + 1) * 2 : data_size / MIN_PARTITION_SIZE;
        int* d_partition_starts;
        int* d_partition_ends;
        int* d_num_partitions;
        
        CUDA_CHECK(cudaMalloc(&d_partition_starts, estimated_partitions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_partition_ends, estimated_partitions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_num_partitions, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_num_partitions, 0, sizeof(int), stream));
        
        blocks = min((num_variance_blocks + threads - 1) / threads, sm_count * 2);

        // MODIFICATION 10: Call the updated kernel with new parameters
        createPartitionsFast<T><<<blocks, threads, 0, stream>>>(
            data_size, base_partition_size, d_variances, num_variance_blocks,
            d_partition_starts, d_partition_ends, d_num_partitions, 
            d_variance_thresholds, d_partition_sizes_for_buckets, 
            num_thresholds, variance_block_multiplier);
        
        int h_num_partitions;
        CUDA_CHECK(cudaMemcpyAsync(&h_num_partitions, d_num_partitions, sizeof(int), 
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Ensure h_num_partitions is not out of bounds
        if (h_num_partitions > estimated_partitions) {
            h_num_partitions = estimated_partitions;
        }

        if (h_num_partitions > 0) {
            thrust::device_ptr<int> starts_ptr(d_partition_starts);
            thrust::device_ptr<int> ends_ptr(d_partition_ends);
            thrust::sort_by_key(starts_ptr, starts_ptr + h_num_partitions, ends_ptr);
        }
        
        int* d_model_types;
        double* d_theta0;
        double* d_theta1;
        int* d_delta_bits;
        long long* d_max_errors;
        double* d_costs;
        
        CUDA_CHECK(cudaMalloc(&d_model_types, h_num_partitions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_theta0, h_num_partitions * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_theta1, h_num_partitions * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_delta_bits, h_num_partitions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_max_errors, h_num_partitions * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_costs, h_num_partitions * sizeof(double)));
        
        int threads_per_block = 256;
        int grid_size = h_num_partitions;

        size_t shared_mem_size = threads_per_block * sizeof(double);
        shared_mem_size = max(shared_mem_size, threads_per_block * sizeof(long long)); 
        if (grid_size > 0) {
            fitPartitionsBatched_Optimized<T><<<grid_size, threads_per_block, shared_mem_size, stream>>>(
                d_data,
                d_partition_starts,
                d_partition_ends,
                d_model_types,
                d_theta0,
                d_theta1,
                d_delta_bits,
                d_max_errors,
                d_costs,
                h_num_partitions
            );
        }
        
        std::vector<int> h_starts(h_num_partitions);
        std::vector<int> h_ends(h_num_partitions);
        std::vector<int> h_model_types(h_num_partitions);
        std::vector<double> h_theta0(h_num_partitions);
        std::vector<double> h_theta1(h_num_partitions);
        std::vector<int> h_delta_bits(h_num_partitions);
        std::vector<long long> h_max_errors(h_num_partitions);
        
        if (h_num_partitions > 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_starts.data(), d_partition_starts, 
                                    h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_ends.data(), d_partition_ends, 
                                    h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_model_types.data(), d_model_types, 
                                    h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_theta0.data(), d_theta0, 
                                    h_num_partitions * sizeof(double), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_theta1.data(), d_theta1, 
                                    h_num_partitions * sizeof(double), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_delta_bits.data(), d_delta_bits, 
                                    h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_max_errors.data(), d_max_errors, 
                                    h_num_partitions * sizeof(long long), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        std::vector<PartitionInfo> result;
        result.reserve(h_num_partitions);
        
        for (int i = 0; i < h_num_partitions; i++) {
            PartitionInfo info;
            info.start_idx = h_starts[i];
            info.end_idx = h_ends[i];
            info.model_type = h_model_types[i];
            info.model_params[0] = h_theta0[i];
            info.model_params[1] = h_theta1[i];
            info.model_params[2] = 0.0;
            info.model_params[3] = 0.0;
            info.delta_bits = h_delta_bits[i];
            info.delta_array_bit_offset = 0;
            info.error_bound = h_max_errors[i];
            result.push_back(info);
        }
        
        if (!result.empty()) {
            std::sort(result.begin(), result.end(), 
                     [](const PartitionInfo& a, const PartitionInfo& b) {
                         return a.start_idx < b.start_idx;
                     });
            
            result[0].start_idx = 0;
            result.back().end_idx = data_size;
            
            for (size_t i = 0; i < result.size() - 1; i++) {
                if (result[i].end_idx != result[i + 1].start_idx) {
                    result[i].end_idx = result[i + 1].start_idx;
                }
            }
        }
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_variances));
        CUDA_CHECK(cudaFree(d_variance_thresholds));
        CUDA_CHECK(cudaFree(d_partition_starts));
        CUDA_CHECK(cudaFree(d_partition_ends));
        CUDA_CHECK(cudaFree(d_num_partitions));
        CUDA_CHECK(cudaFree(d_model_types));
        CUDA_CHECK(cudaFree(d_theta0));
        CUDA_CHECK(cudaFree(d_theta1));
        CUDA_CHECK(cudaFree(d_delta_bits));
        CUDA_CHECK(cudaFree(d_max_errors));
        CUDA_CHECK(cudaFree(d_costs));
        // MODIFICATION 11: Free the new device array
        CUDA_CHECK(cudaFree(d_partition_sizes_for_buckets));
        
        return result;
    }
};

    // Helper function to align offset to a specific boundary
inline uint64_t alignOffset(uint64_t offset, uint64_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
}



// Main LeCoGPU class - UPDATED FOR SoA
template<typename T>
class LeCoGPU {
private:
    cudaStream_t main_cuda_stream;
    cudaStream_t compression_cuda_stream;
    cudaStream_t decompression_cuda_stream;

    struct PartitionCache {
        static const int CACHE_SIZE = 64;
        PartitionInfo entries_arr[CACHE_SIZE];
        int indices_arr[CACHE_SIZE]; 
        int lru_counters[CACHE_SIZE];
        int global_lru_counter;

        PartitionCache() : global_lru_counter(0) {
            std::fill(indices_arr, indices_arr + CACHE_SIZE, -1);
            std::fill(lru_counters, lru_counters + CACHE_SIZE, 0);
        }
        bool find(int cache_key, PartitionInfo& found_info) {
            for (int i = 0; i < CACHE_SIZE; i++) {
                if (indices_arr[i] == cache_key) {
                    found_info = entries_arr[i];
                    lru_counters[i] = ++global_lru_counter;
                    return true;
                }
            }
            return false;
        }
        void insert(int cache_key, const PartitionInfo& new_info) {
            int lru_target_idx = 0;
            int min_counter_val = lru_counters[0];
            for (int i = 0; i < CACHE_SIZE; i++) {
                if (indices_arr[i] == -1) { 
                    lru_target_idx = i; 
                    break; 
                }
                if (lru_counters[i] < min_counter_val) {
                    min_counter_val = lru_counters[i];
                    lru_target_idx = i;
                }
            }
            entries_arr[lru_target_idx] = new_info;
            indices_arr[lru_target_idx] = cache_key;
            lru_counters[lru_target_idx] = ++global_lru_counter;
        }
    };
    PartitionCache cpu_partition_cache;

    uint32_t calculateChecksum(const void* data_to_check, size_t size_of_data) {
        const uint8_t* byte_ptr = static_cast<const uint8_t*>(data_to_check);
        uint32_t csum = 0;
        #ifdef __SSE4_2__
        for (size_t i = 0; i < size_of_data; i++) 
            csum = _mm_crc32_u8(csum, byte_ptr[i]);
        #else
        for (size_t i = 0; i < size_of_data; i++) { 
            csum += byte_ptr[i]; 
            csum = (csum << 1) | (csum >> 31); 
        }
        #endif
        return csum;
    }

public:
    LeCoGPU() {
        CUDA_CHECK(cudaStreamCreate(&main_cuda_stream));
        CUDA_CHECK(cudaStreamCreate(&compression_cuda_stream));
        CUDA_CHECK(cudaStreamCreate(&decompression_cuda_stream));
    }
    
    ~LeCoGPU() {
        CUDA_CHECK(cudaStreamDestroy(main_cuda_stream));
        CUDA_CHECK(cudaStreamDestroy(compression_cuda_stream));
        CUDA_CHECK(cudaStreamDestroy(decompression_cuda_stream));
    }

    void analyzeRandomAccessPerformance(CompressedData<T>* compressed_data,
                                    const std::vector<int>& positions,
                                    int fixed_partition_size) {
        std::cout << "\n--- Random Access Performance Analysis ---" << std::endl;
        
        std::vector<int> sorted_positions = positions;
        std::sort(sorted_positions.begin(), sorted_positions.end());
        
        long long total_stride = 0;
        int sequential_count = 0;
        for (size_t i = 1; i < sorted_positions.size(); i++) {
            int stride = sorted_positions[i] - sorted_positions[i-1];
            total_stride += stride;
            if (stride == 1) sequential_count++;
        }
        
        double avg_stride = (sorted_positions.size() > 1) ? (double)total_stride / (sorted_positions.size() - 1) : 0.0;
        double sequential_ratio = (sorted_positions.size() > 1) ? (double)sequential_count / (sorted_positions.size() - 1) : 0.0;
        
        std::cout << "Query pattern analysis:" << std::endl;
        std::cout << "  - Average stride: " << avg_stride << std::endl;
        std::cout << "  - Sequential access ratio: " << sequential_ratio << std::endl;
        std::cout << "  - Total unique positions: " << positions.size() << std::endl;
        
        std::map<int, int> partition_counts;
        for (int pos : positions) {
            int partition = pos / fixed_partition_size;
            partition_counts[partition]++;
        }
        
        std::cout << "  - Unique partitions accessed: " << partition_counts.size() << std::endl;
        if (!partition_counts.empty()) {
            std::cout << "  - Queries per partition (avg): " << 
                (double)positions.size() / partition_counts.size() << std::endl;
        }
        
        size_t bitpacked_bytes_accessed = positions.size() * sizeof(T) * 2; 
        size_t preunpacked_bytes_accessed = positions.size() * sizeof(long long);
        
        std::cout << "\nMemory access analysis:" << std::endl;
        std::cout << "  - Bit-packed bytes accessed: ~" << bitpacked_bytes_accessed << std::endl;
        std::cout << "  - Pre-unpacked bytes accessed: " << preunpacked_bytes_accessed << std::endl;
        if (bitpacked_bytes_accessed > 0) {
            std::cout << "  - Memory access ratio: " << 
                (double)preunpacked_bytes_accessed / bitpacked_bytes_accessed << "x" << std::endl;
        }
    }
    
    // MODIFICATION 12: compress method signature updated
    CompressedData<T>* compress(const std::vector<T>& host_data_vec,
                                bool use_variable_partitioning,
                                long long* compressed_size_bytes_result,
                                bool use_gpu_partitioning = false,
                                const std::string& dataset_name = "",
                                int variance_block_multiplier = 8,
                                int num_thresholds = 3) {
        int num_elements = host_data_vec.size();
        if (num_elements == 0) {
            if(compressed_size_bytes_result) *compressed_size_bytes_result = 0;
            CompressedData<T>* result_empty = new CompressedData<T>();
            result_empty->num_partitions = 0; 
            result_empty->total_values = 0;
            result_empty->d_start_indices = nullptr;
            result_empty->d_end_indices = nullptr;
            result_empty->d_model_types = nullptr;
            result_empty->d_model_params = nullptr;
            result_empty->d_delta_bits = nullptr;
            result_empty->d_delta_array_bit_offsets = nullptr;
            result_empty->d_error_bounds = nullptr;
            result_empty->delta_array = nullptr;
            result_empty->d_plain_deltas = nullptr;
            result_empty->d_self = nullptr;
            return result_empty;
        }

        T* d_input_data;
        CUDA_CHECK(cudaMalloc(&d_input_data, num_elements * sizeof(T)));
        CUDA_CHECK(cudaMemcpyAsync(d_input_data, host_data_vec.data(), 
                                num_elements * sizeof(T),
                                cudaMemcpyHostToDevice, compression_cuda_stream));

        CompressedData<T>* result_compressed_data = new CompressedData<T>();
        std::vector<PartitionInfo> h_partition_infos;
        
        // MODIFICATION 13: Pass new parameters to the partitioner's constructor
        GPUVariableLengthPartitionerV6<T> gpu_partitioner(host_data_vec, 
                                                        2048,   // Base size
                                                        compression_cuda_stream,
                                                        variance_block_multiplier,
                                                        num_thresholds);
        h_partition_infos = gpu_partitioner.partition();

        
        if (h_partition_infos.empty() && num_elements > 0) {
            PartitionInfo p_def; 
            p_def.start_idx = 0; 
            p_def.end_idx = num_elements;
            p_def.model_type = MODEL_LINEAR; 
            std::fill(p_def.model_params, p_def.model_params + 4, 0.0);
            p_def.delta_bits = 0; 
            p_def.delta_array_bit_offset = 0; 
            p_def.error_bound = 0;
            h_partition_infos.push_back(p_def);
        }
        
        result_compressed_data->num_partitions = h_partition_infos.size();
        if (result_compressed_data->num_partitions == 0 && num_elements > 0) {
            CUDA_CHECK(cudaFree(d_input_data)); 
            delete result_compressed_data;
            if(compressed_size_bytes_result) *compressed_size_bytes_result = 0;
            return nullptr;
        }
        
        result_compressed_data->total_values = num_elements;
        result_compressed_data->d_plain_deltas = nullptr; 
        
        if (result_compressed_data->num_partitions == 0) {
            result_compressed_data->d_start_indices = nullptr;
            result_compressed_data->d_end_indices = nullptr;
            result_compressed_data->d_model_types = nullptr;
            result_compressed_data->d_model_params = nullptr;
            result_compressed_data->d_delta_bits = nullptr;
            result_compressed_data->d_delta_array_bit_offsets = nullptr;
            result_compressed_data->d_error_bounds = nullptr;
            result_compressed_data->delta_array = nullptr;
        } else {
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_start_indices,
                                result_compressed_data->num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_end_indices,
                                result_compressed_data->num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_model_types,
                                result_compressed_data->num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_model_params,
                                result_compressed_data->num_partitions * 4 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_delta_bits,
                                result_compressed_data->num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_delta_array_bit_offsets,
                                result_compressed_data->num_partitions * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_error_bounds,
                                result_compressed_data->num_partitions * sizeof(long long)));
            
            std::vector<int32_t> h_start_indices(result_compressed_data->num_partitions);
            std::vector<int32_t> h_end_indices(result_compressed_data->num_partitions);
            std::vector<int32_t> h_model_types(result_compressed_data->num_partitions);
            std::vector<double> h_model_params(result_compressed_data->num_partitions * 4);
            std::vector<int32_t> h_delta_bits(result_compressed_data->num_partitions);
            std::vector<int64_t> h_delta_array_bit_offsets(result_compressed_data->num_partitions);
            std::vector<long long> h_error_bounds(result_compressed_data->num_partitions);
            
            for (int i = 0; i < result_compressed_data->num_partitions; i++) {
                h_start_indices[i] = h_partition_infos[i].start_idx;
                h_end_indices[i] = h_partition_infos[i].end_idx;
                h_model_types[i] = h_partition_infos[i].model_type;
                for (int j = 0; j < 4; j++) {
                    h_model_params[i * 4 + j] = h_partition_infos[i].model_params[j];
                }
                h_delta_bits[i] = h_partition_infos[i].delta_bits;
                h_delta_array_bit_offsets[i] = h_partition_infos[i].delta_array_bit_offset;
                h_error_bounds[i] = h_partition_infos[i].error_bound;
            }
            
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_start_indices, h_start_indices.data(),
                                    result_compressed_data->num_partitions * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_end_indices, h_end_indices.data(),
                                    result_compressed_data->num_partitions * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_model_types, h_model_types.data(),
                                    result_compressed_data->num_partitions * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_model_params, h_model_params.data(),
                                    result_compressed_data->num_partitions * 4 * sizeof(double),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_delta_bits, h_delta_bits.data(),
                                    result_compressed_data->num_partitions * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_delta_array_bit_offsets, h_delta_array_bit_offsets.data(),
                                    result_compressed_data->num_partitions * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_error_bounds, h_error_bounds.data(),
                                    result_compressed_data->num_partitions * sizeof(long long),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
        }

        int64_t* d_total_bits;
        CUDA_CHECK(cudaMalloc(&d_total_bits, sizeof(int64_t)));
        CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), compression_cuda_stream));

        if (result_compressed_data->num_partitions > 0) {
            int block_size = std::min(256, ((h_partition_infos[0].end_idx - h_partition_infos[0].start_idx) + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
            block_size = std::max(block_size, WARP_SIZE);
            
            size_t shared_mem_size = (4 * sizeof(double) + sizeof(long long) + sizeof(bool)) * block_size;
            
            wprocessPartitionsKernel<T><<<result_compressed_data->num_partitions, block_size, 
                            shared_mem_size, compression_cuda_stream>>>(
                d_input_data,
                result_compressed_data->d_start_indices,
                result_compressed_data->d_end_indices,
                result_compressed_data->d_model_types,
                result_compressed_data->d_model_params,
                result_compressed_data->d_delta_bits,
                result_compressed_data->d_error_bounds,
                result_compressed_data->num_partitions,
                d_total_bits);
            CUDA_CHECK(cudaGetLastError());
            
            int offset_blocks = (result_compressed_data->num_partitions + 255) / 256;
            setBitOffsetsKernel<<<offset_blocks, 256, 0, compression_cuda_stream>>>(
                result_compressed_data->d_start_indices,
                result_compressed_data->d_end_indices,
                result_compressed_data->d_delta_bits,
                result_compressed_data->d_delta_array_bit_offsets,
                result_compressed_data->num_partitions);
            CUDA_CHECK(cudaGetLastError());
        }

        int64_t h_total_bits = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_total_bits, d_total_bits, sizeof(int64_t),
                                cudaMemcpyDeviceToHost, compression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(compression_cuda_stream));
        CUDA_CHECK(cudaFree(d_total_bits));

        size_t final_delta_array_words = (h_total_bits + 31) / 32;
        if (h_total_bits == 0) final_delta_array_words = 0;
        
        if (final_delta_array_words > 0) {
            CUDA_CHECK(cudaMalloc(&result_compressed_data->delta_array, 
                                final_delta_array_words * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemsetAsync(result_compressed_data->delta_array, 0, 
                                    final_delta_array_words * sizeof(uint32_t), 
                                    compression_cuda_stream));
        } else {
            result_compressed_data->delta_array = nullptr;
        }

        if (h_total_bits > 0 && result_compressed_data->num_partitions > 0 && num_elements > 0) {
            int pack_kernel_block_dim = MAX_BLOCK_SIZE;
            int pack_kernel_grid_dim = (num_elements + pack_kernel_block_dim - 1) / pack_kernel_block_dim;
            pack_kernel_grid_dim = std::min(pack_kernel_grid_dim, 65535);
            
            packDeltasKernelOptimized<T><<<pack_kernel_grid_dim, pack_kernel_block_dim, 0, 
                                        compression_cuda_stream>>>(
                d_input_data,
                result_compressed_data->d_start_indices,
                result_compressed_data->d_end_indices,
                result_compressed_data->d_model_types,
                result_compressed_data->d_model_params,
                result_compressed_data->d_delta_bits,
                result_compressed_data->d_delta_array_bit_offsets,
                result_compressed_data->num_partitions,
                result_compressed_data->delta_array);
            CUDA_CHECK(cudaGetLastError());
        }
        
        if(compressed_size_bytes_result) {
            long long final_model_size = 0;
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(int32_t); // start_indices
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(int32_t); // end_indices
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(int32_t); // model_types
            final_model_size += (long long)result_compressed_data->num_partitions * 4 * sizeof(double); // model_params
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(int32_t); // delta_bits
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(int64_t); // delta_array_bit_offsets
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(long long); // error_bounds
            final_model_size += sizeof(SerializedHeader); // header
            
            long long final_delta_size = (h_total_bits + 7) / 8;
            *compressed_size_bytes_result = final_model_size + final_delta_size;
        }
        
        CUDA_CHECK(cudaMalloc(&result_compressed_data->d_self, sizeof(CompressedData<T>)));
        CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_self, result_compressed_data, 
                                sizeof(CompressedData<T>), cudaMemcpyHostToDevice, 
                                compression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(compression_cuda_stream));

        CUDA_CHECK(cudaFree(d_input_data));
        
        return result_compressed_data;
    }
    
    
    void cleanup(CompressedData<T>* compressed_object_to_clean) {
        if (compressed_object_to_clean) {
            if (compressed_object_to_clean->d_start_indices) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_start_indices));
            if (compressed_object_to_clean->d_end_indices) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_end_indices));
            if (compressed_object_to_clean->d_model_types) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_model_types));
            if (compressed_object_to_clean->d_model_params) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_model_params));
            if (compressed_object_to_clean->d_delta_bits) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_delta_bits));
            if (compressed_object_to_clean->d_delta_array_bit_offsets) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_delta_array_bit_offsets));
            if (compressed_object_to_clean->d_error_bounds) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_error_bounds));
            if (compressed_object_to_clean->delta_array) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->delta_array));
            
            if (compressed_object_to_clean->d_plain_deltas)
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_plain_deltas));
                
            if (compressed_object_to_clean->d_self) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_self));
            delete compressed_object_to_clean;
        }
    }

        
    

    void decompressFullFile_OnTheFly_Optimized_V2(CompressedData<T>* compressed_data_input,
                                                std::vector<T>& output_decompressed_data) {
        if (!compressed_data_input || compressed_data_input->total_values == 0) {
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_self) {
            std::cerr << "Error: compressed_data_input->d_self is null." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        int total_elements = compressed_data_input->total_values;
        output_decompressed_data.resize(total_elements);
        
        T* d_output_ptr;
        CUDA_CHECK(cudaMalloc(&d_output_ptr, total_elements * sizeof(T)));
        
        int grid_size = compressed_data_input->num_partitions;
        int block_size = 256;
        
        if (grid_size > 0) {
            ::decompressFullFile_OnTheFly_Optimized_V2<T><<<grid_size, block_size, 0, 
                                                        decompression_cuda_stream>>>(
                compressed_data_input->d_self, d_output_ptr, total_elements);
            CUDA_CHECK(cudaGetLastError());
        }
        
        CUDA_CHECK(cudaMemcpyAsync(output_decompressed_data.data(), d_output_ptr,
                                total_elements * sizeof(T), cudaMemcpyDeviceToHost,
                                decompression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(decompression_cuda_stream));
        
        CUDA_CHECK(cudaFree(d_output_ptr));
    }

public:
    SerializedData* serializeGPU(CompressedData<T>* compressed_object_to_serialize) {
        if (!compressed_object_to_serialize) { 
            std::cerr << "Error: Null data to serialize." << std::endl; 
            return nullptr; 
        }
        
        if (compressed_object_to_serialize->delta_array == nullptr && 
            compressed_object_to_serialize->d_plain_deltas != nullptr) {
            std::cerr << "Error: Cannot serialize data that has been deserialized with pre-unpacking enabled. "
                    << "Re-serialization is not supported in this mode." << std::endl;
            return nullptr;
        }
        
        SerializedHeader file_header;
        memset(&file_header, 0, sizeof(SerializedHeader));
        file_header.magic = 0x4F43454C; 
        file_header.version = 5;
        file_header.total_values = compressed_object_to_serialize->total_values;
        file_header.num_partitions = compressed_object_to_serialize->num_partitions;
        
        uint64_t max_bit_offset_val = 0;
        if (compressed_object_to_serialize->num_partitions > 0) {
            int64_t last_bit_offset;
            int32_t last_delta_bits;
            int32_t last_start, last_end;
            int last_idx = compressed_object_to_serialize->num_partitions - 1;
            
            CUDA_CHECK(cudaMemcpy(&last_bit_offset, 
                compressed_object_to_serialize->d_delta_array_bit_offsets + last_idx,
                sizeof(int64_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_delta_bits, 
                compressed_object_to_serialize->d_delta_bits + last_idx,
                sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_start, 
                compressed_object_to_serialize->d_start_indices + last_idx,
                sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_end, 
                compressed_object_to_serialize->d_end_indices + last_idx,
                sizeof(int32_t), cudaMemcpyDeviceToHost));
            
            int seg_len = last_end - last_start;
            if (seg_len > 0) {
                max_bit_offset_val = last_bit_offset + (uint64_t)seg_len * last_delta_bits;
            }
        }
        
        uint64_t total_delta_bytes = (max_bit_offset_val + 7) / 8;
        
        uint64_t current_offset = sizeof(SerializedHeader);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.start_indices_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.end_indices_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.model_types_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.model_params_offset = current_offset;
        file_header.model_params_size_bytes = compressed_object_to_serialize->num_partitions * 4 * sizeof(double);
        current_offset += file_header.model_params_size_bytes;
        
        current_offset = alignOffset(current_offset, 8);
        file_header.delta_bits_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.delta_array_bit_offsets_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int64_t);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.error_bounds_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(long long);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.delta_array_offset = current_offset;
        file_header.delta_array_size_bytes = total_delta_bytes;
        
        file_header.data_type_size = sizeof(T);
        
        SerializedHeader temp_header_for_csum = file_header;
        temp_header_for_csum.header_checksum = 0;
        file_header.header_checksum = calculateChecksum(&temp_header_for_csum, sizeof(SerializedHeader));

        size_t final_total_size = current_offset + total_delta_bytes;
        final_total_size = alignOffset(final_total_size, 8);
        
        uint8_t* d_serialized_blob;
        CUDA_CHECK(cudaMalloc(&d_serialized_blob, final_total_size));
        CUDA_CHECK(cudaMemset(d_serialized_blob, 0, final_total_size));
        
        SerializedHeader* d_header;
        CUDA_CHECK(cudaMalloc(&d_header, sizeof(SerializedHeader)));
        CUDA_CHECK(cudaMemcpy(d_header, &file_header, sizeof(SerializedHeader), cudaMemcpyHostToDevice));
        
        int block_size = 256;
        int num_blocks = 8 + ((total_delta_bytes > 0) ? 32 : 0);
        
        if (compressed_object_to_serialize->num_partitions > 0 || total_delta_bytes > 0) {
            packToBlobKernelOptimized<<<num_blocks, block_size, 0, main_cuda_stream>>>(
                d_header,
                compressed_object_to_serialize->d_start_indices,
                compressed_object_to_serialize->d_end_indices,
                compressed_object_to_serialize->d_model_types,
                compressed_object_to_serialize->d_model_params,
                compressed_object_to_serialize->d_delta_bits,
                compressed_object_to_serialize->d_delta_array_bit_offsets,
                compressed_object_to_serialize->d_error_bounds,
                compressed_object_to_serialize->delta_array,
                compressed_object_to_serialize->num_partitions,
                total_delta_bytes,
                d_serialized_blob
            );
            CUDA_CHECK(cudaGetLastError());
        }
        
        SerializedData* output_serialized_obj = new SerializedData();
        try { 
            output_serialized_obj->data = new uint8_t[final_total_size]; 
            memset(output_serialized_obj->data, 0, final_total_size);
        }
        catch (const std::bad_alloc& e) { 
            CUDA_CHECK(cudaFree(d_serialized_blob));
            CUDA_CHECK(cudaFree(d_header));
            delete output_serialized_obj; 
            return nullptr; 
        }
        output_serialized_obj->size = final_total_size;
        
        CUDA_CHECK(cudaMemcpyAsync(output_serialized_obj->data, d_serialized_blob, 
                                final_total_size, cudaMemcpyDeviceToHost, main_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(main_cuda_stream));
        
        CUDA_CHECK(cudaFree(d_serialized_blob));
        CUDA_CHECK(cudaFree(d_header));
        
        return output_serialized_obj;
    }


    CompressedData<T>* deserializeGPU(const SerializedData* serialized_input_data, bool preUnpackDeltas = false) {
        if (!serialized_input_data || !serialized_input_data->data || 
            serialized_input_data->size < sizeof(SerializedHeader)) { 
            return nullptr; 
        }
        
        SerializedHeader read_header;
        memcpy(&read_header, serialized_input_data->data, sizeof(read_header));

        if (read_header.magic != 0x4F43454C || 
            (read_header.version != 4 && read_header.version != 5) ||
            read_header.data_type_size != sizeof(T)) {
            return nullptr; 
        }
        
        SerializedHeader temp_hdr_for_csum = read_header; 
        temp_hdr_for_csum.header_checksum = 0;
        if (calculateChecksum(&temp_hdr_for_csum, sizeof(SerializedHeader)) != read_header.header_checksum) { 
            return nullptr; 
        }

        uint8_t* d_serialized_blob;
        CUDA_CHECK(cudaMalloc(&d_serialized_blob, serialized_input_data->size));
        CUDA_CHECK(cudaMemcpyAsync(d_serialized_blob, serialized_input_data->data, 
                                serialized_input_data->size, cudaMemcpyHostToDevice, 
                                main_cuda_stream));
        
        CompressedData<T>* new_compressed_data = new CompressedData<T>();
        new_compressed_data->num_partitions = read_header.num_partitions;
        new_compressed_data->total_values = read_header.total_values;
        new_compressed_data->d_plain_deltas = nullptr;
        
        if (new_compressed_data->num_partitions > 0) {
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_start_indices, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_end_indices, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_model_types, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_model_params, 
                                read_header.num_partitions * 4 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_delta_bits, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_delta_array_bit_offsets, 
                                read_header.num_partitions * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_error_bounds, 
                                read_header.num_partitions * sizeof(long long)));
        } else {
            new_compressed_data->d_start_indices = nullptr;
            new_compressed_data->d_end_indices = nullptr;
            new_compressed_data->d_model_types = nullptr;
            new_compressed_data->d_model_params = nullptr;
            new_compressed_data->d_delta_bits = nullptr;
            new_compressed_data->d_delta_array_bit_offsets = nullptr;
            new_compressed_data->d_error_bounds = nullptr;
        }
        
        uint64_t num_delta_words = (read_header.delta_array_size_bytes + 3) / 4;
        if (num_delta_words > 0) {
            CUDA_CHECK(cudaMalloc(&new_compressed_data->delta_array, 
                                num_delta_words * sizeof(uint32_t)));
        } else { 
            new_compressed_data->delta_array = nullptr; 
        }
        
        if (new_compressed_data->num_partitions > 0 || num_delta_words > 0) {
            int block_size = 256;
            int grid_size = (new_compressed_data->num_partitions * 8 + num_delta_words + block_size - 1) / block_size;
            grid_size = min(grid_size, 65535);
            
            unpackFromBlobKernel<<<grid_size, block_size, 0, main_cuda_stream>>>(
                d_serialized_blob,
                new_compressed_data->num_partitions,
                read_header.delta_array_size_bytes,
                new_compressed_data->d_start_indices,
                new_compressed_data->d_end_indices,
                new_compressed_data->d_model_types,
                new_compressed_data->d_model_params,
                new_compressed_data->d_delta_bits,
                new_compressed_data->d_delta_array_bit_offsets,
                new_compressed_data->d_error_bounds,
                new_compressed_data->delta_array
            );
            CUDA_CHECK(cudaGetLastError());
        }
        
        if (preUnpackDeltas && new_compressed_data->total_values > 0 && new_compressed_data->delta_array != nullptr) {
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_plain_deltas, 
                                new_compressed_data->total_values * sizeof(long long)));
            
            int block_size = 256;
            int grid_size = (new_compressed_data->total_values + block_size - 1) / block_size;
            
            cudaDeviceProp prop;
            int device;
            CUDA_CHECK(cudaGetDevice(&device));
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            grid_size = std::min(grid_size, prop.multiProcessorCount * 32);
            
            unpackAllDeltasKernel<T><<<grid_size, block_size, 0, main_cuda_stream>>>(
                new_compressed_data->d_start_indices,
                new_compressed_data->d_end_indices,
                new_compressed_data->d_model_types,
                new_compressed_data->d_delta_bits,
                new_compressed_data->d_delta_array_bit_offsets,
                new_compressed_data->delta_array,
                new_compressed_data->d_plain_deltas,
                new_compressed_data->num_partitions,
                new_compressed_data->total_values
            );
            CUDA_CHECK(cudaGetLastError());
            
            CUDA_CHECK(cudaStreamSynchronize(main_cuda_stream));
            
            CUDA_CHECK(cudaFree(new_compressed_data->delta_array));
            new_compressed_data->delta_array = nullptr;
        }
        
        CUDA_CHECK(cudaMalloc(&new_compressed_data->d_self, sizeof(CompressedData<T>)));
        CUDA_CHECK(cudaMemcpyAsync(new_compressed_data->d_self, new_compressed_data, 
                                sizeof(CompressedData<T>), cudaMemcpyHostToDevice,
                                main_cuda_stream));
        
        CUDA_CHECK(cudaStreamSynchronize(main_cuda_stream));
        
        CUDA_CHECK(cudaFree(d_serialized_blob));
        
        return new_compressed_data;
    }
};

// Benchmark & File I/O Utilities
template<typename Func>
double benchmark(Func func_to_run, int num_iterations = 100) {
    if (num_iterations > 0) 
        for (int i = 0; i < std::min(5, num_iterations); i++) 
            func_to_run();
    auto timer_start = std::chrono::high_resolution_clock::now();
    if (num_iterations > 0) 
        for (int i = 0; i < num_iterations; i++) 
            func_to_run();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto timer_end = std::chrono::high_resolution_clock::now();
    if (num_iterations <= 0) return 0.0;
    return std::chrono::duration<double, std::milli>(timer_end - timer_start).count() / num_iterations;
}

template<typename T>
bool read_text_file(const std::string& in_filename, std::vector<T>& out_data_vec) {
    std::ifstream file_stream(in_filename);
    if (!file_stream.is_open()) { 
        std::cerr << "Error: Could not open text file " << in_filename << std::endl; 
        return false; 
    }
    out_data_vec.clear(); 
    std::string line_str;
    while (std::getline(file_stream, line_str)) {
        try {
            line_str.erase(0, line_str.find_first_not_of(" \t\n\r\f\v"));
            line_str.erase(line_str.find_last_not_of(" \t\n\r\f\v") + 1);
            if (line_str.empty()) continue;
            if (std::is_same<T, int>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoi(line_str)));
            else if (std::is_same<T, long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stol(line_str)));
            else if (std::is_same<T, long long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoll(line_str)));
            else if (std::is_same<T, unsigned int>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoul(line_str)));
            else if (std::is_same<T, unsigned long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoul(line_str)));
            else if (std::is_same<T, unsigned long long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoull(line_str)));
            else { 
                std::cerr << "Warning: Read_text_file unsupported integer type." << std::endl; 
            }
        } catch (const std::exception& e) { 
            std::cerr << "Warning: Parsing line '"<< line_str << "': " << e.what() << std::endl;
        }
    }
    file_stream.close();
    std::cout << "Successfully read " << out_data_vec.size() << " values from text file: " 
             << in_filename << std::endl;
    return true;
}

template<typename T>
bool read_binary_file(const std::string& in_filename, std::vector<T>& out_data_vec) {
    std::ifstream file_stream(in_filename, std::ios::binary | std::ios::ate);
    if (!file_stream.is_open()) { 
        std::cerr << "Error: Could not open binary file " << in_filename << std::endl; 
        return false; 
    }
    std::streampos stream_file_size = file_stream.tellg();
    if (stream_file_size < 0 || stream_file_size % sizeof(T) != 0) { 
        std::cerr << "Error: Binary file " << in_filename << " has invalid size." << std::endl; 
        file_stream.close(); 
        return false;
    }
    if (stream_file_size == 0) {
        out_data_vec.clear(); 
        file_stream.close(); 
        std::cout << "Read 0 values from empty binary file: " << in_filename << std::endl; 
        return true;
    }
    file_stream.seekg(0, std::ios::beg);
    size_t num_file_elements = static_cast<size_t>(stream_file_size) / sizeof(T);
    try {
        out_data_vec.resize(num_file_elements);
    } catch(const std::bad_alloc&){ 
        std::cerr << "Error: Malloc failed for binary data." << std::endl; 
        file_stream.close();
        return false;
    }
    file_stream.read(reinterpret_cast<char*>(out_data_vec.data()), num_file_elements * sizeof(T));
    bool read_success = file_stream.good() && 
                       (static_cast<size_t>(file_stream.gcount()) == num_file_elements * sizeof(T));
    file_stream.close();
    if(!read_success) {
        out_data_vec.clear(); 
        std::cerr << "Error reading binary file: " << in_filename << std::endl; 
        return false;
    }
    std::cout << "Successfully read " << out_data_vec.size() << " values from binary file: " 
             << in_filename << std::endl;
    return true;
}

// MODIFICATION 14: Test function signature updated
template<typename T>
void run_compression_test(const std::vector<T>& data_to_test, 
                          const std::string& data_type_string_name, 
                          const std::string& dataset_name = "",
                          int variance_block_multiplier = 8,
                          int num_thresholds = 3) {
    if (data_to_test.empty()) {
        std::cout << "Input data vector is empty. Skipping test for type " << data_type_string_name << "." << std::endl;
        return;
    }

    LeCoGPU<T> leco;

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "  Running Test for Data Type: " << data_type_string_name << " (" << sizeof(T) * 8 << "-bit)" << std::endl;
    std::cout << "  Dataset: " << dataset_name << std::endl;
    std::cout << "  Number of values: " << data_to_test.size() << std::endl;
    // MODIFICATION 15: Print new partitioning parameters
    std::cout << "  Partitioner Params: Multiplier=" << variance_block_multiplier 
              << ", Thresholds=" << num_thresholds << std::endl;
    std::cout << "======================================================================" << std::endl;

    long long compressed_size_bytes = 0;
    CompressedData<T>* compressed_data = nullptr;
    double compression_time_ms = 0;

    std::cout << "\n--- Compression Phase ---" << std::endl;
    compression_time_ms = benchmark([&]() {
        if (compressed_data) {
            leco.cleanup(compressed_data);
            compressed_data = nullptr;
        }
        // MODIFICATION 16: Pass new parameters to compress function
        compressed_data = leco.compress(data_to_test, true, &compressed_size_bytes, true, dataset_name, variance_block_multiplier, num_thresholds);
    }, 5); 

    if (!compressed_data || compressed_size_bytes == 0) {
        std::cerr << "Compression failed or resulted in zero size." << std::endl;
        return;
    }

    long long original_size_bytes = data_to_test.size() * sizeof(T);
    double compression_ratio = static_cast<double>(original_size_bytes) / compressed_size_bytes;
    double compression_throughput_mbs = (static_cast<double>(original_size_bytes) / (1024.0 * 1024.0)) / (compression_time_ms / 1000.0);

    printf("Original Size:      %lld bytes\n", original_size_bytes);
    printf("Compressed Size:    %lld bytes\n", compressed_size_bytes);
    printf("Compression Ratio:  %.2f : 1\n", compression_ratio);
    printf("Compression Time:   %.4f ms\n", compression_time_ms);
    printf("Compression Speed:  %.2f MB/s\n", compression_throughput_mbs);

    std::cout << "\n--- Decompression Phase (Standard On-the-Fly) ---" << std::endl;
    std::vector<T> decompressed_data;
    double decompression_time_ms = benchmark([&]() {
        leco.decompressFullFile_OnTheFly_Optimized_V2(compressed_data, decompressed_data);
    }, 100);

    double decompression_throughput_mbs = (static_cast<double>(original_size_bytes) / (1024.0 * 1024.0)) / (decompression_time_ms / 1000.0);
    printf("Decompression Time: %.4f ms\n", decompression_time_ms);
    printf("Decompression Speed:%.2f MB/s\n", decompression_throughput_mbs);

    bool verification_passed = (data_to_test.size() == decompressed_data.size());
    if (verification_passed) {
        size_t mismatches = 0;
        for (size_t i = 0; i < data_to_test.size(); ++i) {
            if (data_to_test[i] != decompressed_data[i]) {
                mismatches++;
            }
        }
        if (mismatches == 0) {
            printf("Verification:       SUCCESS\n");
        } else {
            verification_passed = false;
            printf("Verification:       FAILED (%zu mismatches)\n", mismatches);
        }
    } else {
        printf("Verification:       FAILED (size mismatch: original %zu vs decompressed %zu)\n", data_to_test.size(), decompressed_data.size());
    }

    std::cout << "\n--- Serialization / Deserialization Phase ---" << std::endl;
    SerializedData* serialized_blob = nullptr;
    double serialization_time_ms = benchmark([&]() {
        if(serialized_blob) delete serialized_blob;
        serialized_blob = leco.serializeGPU(compressed_data);
    }, 20);

    if (!serialized_blob || serialized_blob->size == 0) {
        std::cerr << "Serialization failed." << std::endl;
        leco.cleanup(compressed_data);
        return;
    }
    printf("Serialization Time: %.4f ms\n", serialization_time_ms);
    printf("Serialized Size:    %zu bytes\n", serialized_blob->size);

    CompressedData<T>* deserialized_data = nullptr;
    double deserialization_time_ms = benchmark([&]() {
        if(deserialized_data) leco.cleanup(deserialized_data);
        deserialized_data = leco.deserializeGPU(serialized_blob, false);
    }, 100);

    if (!deserialized_data) {
        std::cerr << "Deserialization failed." << std::endl;
        delete serialized_blob;
        leco.cleanup(compressed_data);
        return;
    }
    printf("Deserialization Time: %.4f ms\n", deserialization_time_ms);

    std::vector<T> decompressed_from_deserialized;
    leco.decompressFullFile_OnTheFly_Optimized_V2(deserialized_data, decompressed_from_deserialized);
    if (data_to_test == decompressed_from_deserialized) {
        printf("Verification (Post-S/D): SUCCESS\n");
    } else {
        printf("Verification (Post-S/D): FAILED\n");
    }

    std::cout << "\n--- Decompression Phase (High-Throughput Pre-Unpacked) ---" << std::endl;
    CompressedData<T>* deserialized_preunpacked_data = nullptr;
    double deserialization_preunpack_time_ms = benchmark([&]() {
        if(deserialized_preunpacked_data) leco.cleanup(deserialized_preunpacked_data);
        deserialized_preunpacked_data = leco.deserializeGPU(serialized_blob, true);
    }, 100);

    if (!deserialized_preunpacked_data) {
        std::cerr << "Deserialization with pre-unpacking failed." << std::endl;
        delete serialized_blob;
        leco.cleanup(compressed_data);
        leco.cleanup(deserialized_data);
        return;
    }
    printf("Deserialization Time (Pre-Unpack): %.4f ms\n", deserialization_preunpack_time_ms);
    
    std::vector<T> decompressed_preunpacked;
    double decompression_preunpacked_time_ms = benchmark([&]() {
        leco.decompressFullFile_OnTheFly_Optimized_V2(deserialized_preunpacked_data, decompressed_preunpacked);
    }, 100);
    
    double decompression_preunpacked_throughput_mbs = (static_cast<double>(original_size_bytes) / (1024.0 * 1024.0)) / (decompression_preunpacked_time_ms / 1000.0);
    printf("Decompression Time (Pre-Unpacked): %.4f ms\n", decompression_preunpacked_time_ms);
    printf("Decompression Speed (Pre-Unpacked):%.2f MB/s\n", decompression_preunpacked_throughput_mbs);

    if (data_to_test == decompressed_preunpacked) {
        printf("Verification (Pre-Unpacked):       SUCCESS\n");
    } else {
        printf("Verification (Pre-Unpacked):       FAILED\n");
    }

    std::cout << "\n--- Cleanup ---" << std::endl;
    leco.cleanup(compressed_data);
    leco.cleanup(deserialized_data);
    leco.cleanup(deserialized_preunpacked_data);
    if (serialized_blob) {
        delete serialized_blob;
    }
    std::cout << "All resources cleaned up." << std::endl;
    std::cout << "======================================================================\n" << std::endl;
}

// Main Function (only compiled when not in library mode)
#ifndef GLECO2_LIB_MODE
int main(int argc, char* argv[]) {
    srand(static_cast<unsigned int>(time(NULL)));

    // MODIFICATION 17: Variables for new command-line arguments
    std::string data_type_arg_str = "int";
    std::string file_type_flag_str = "";
    std::string input_filename_str = "";
    int variance_block_multiplier = 8; // Default value
    int num_thresholds = 3;            // Default value

    // MODIFICATION 18: Enhanced command-line parsing loop
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--text" || args[i] == "--binary") {
            if (i + 1 < args.size()) {
                file_type_flag_str = args[i];
                input_filename_str = args[++i];
            }
        } else if (args[i] == "--multiplier") {
            if (i + 1 < args.size()) {
                try {
                    variance_block_multiplier = std::stoi(args[++i]);
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid value for --multiplier." << std::endl;
                    return 1;
                }
            }
        } else if (args[i] == "--thresholds") {
            if (i + 1 < args.size()) {
                try {
                    num_thresholds = std::stoi(args[++i]);
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid value for --thresholds." << std::endl;
                    return 1;
                }
            }
        } else { // Assume it's the data type or filename
            if (i == 0) { // First positional argument is data type
                data_type_arg_str = args[i];
            } else if (input_filename_str.empty()) { // Second could be filename
                input_filename_str = args[i];
                if (file_type_flag_str.empty()) file_type_flag_str = "--text";
            }
        }
    }


    std::cout << "LeCo GPU Compression Test - Integer Focus" << std::endl;
    std::cout << "Selected Data Type: " << data_type_arg_str << std::endl;
    if (!input_filename_str.empty() && !file_type_flag_str.empty()) 
        std::cout << "Input File: " << input_filename_str << " (Type: " << file_type_flag_str << ")" << std::endl;
    else { 
        std::cout << "Using synthetic data." << std::endl; 
        input_filename_str = ""; 
        file_type_flag_str = ""; 
    }
    
    int cuda_device_count; 
    CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
    if (cuda_device_count == 0) { 
        std::cerr << "No CUDA devices found." << std::endl; 
        return 1; 
    }
    cudaDeviceProp cuda_dev_props; 
    CUDA_CHECK(cudaGetDeviceProperties(&cuda_dev_props, 0));
    std::cout << "Using GPU: " << cuda_dev_props.name << " (Compute: " << cuda_dev_props.major 
             << "." << cuda_dev_props.minor << ")" << std::endl;
    
    bool data_loaded_ok = false;

    std::string dataset_name;
    if (!input_filename_str.empty()) {
        size_t last_slash = input_filename_str.find_last_of("/\\");
        dataset_name = (last_slash != std::string::npos) ? 
                       input_filename_str.substr(last_slash + 1) : input_filename_str;
        
        size_t last_dot = dataset_name.find_last_of(".");
        if (last_dot != std::string::npos) {
            dataset_name = dataset_name.substr(0, last_dot);
        }
    } else {
        dataset_name = "synthetic_" + data_type_arg_str;
    }

    std::cout << "File name = " << dataset_name << std::endl;
    
    if (data_type_arg_str == "int") {
        std::vector<int> int_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") data_loaded_ok = read_text_file(input_filename_str, int_data_vec);
            else if (file_type_flag_str == "--binary") data_loaded_ok = read_binary_file(input_filename_str, int_data_vec);
        } else { 
            int_data_vec.resize(1000000); 
            for(size_t i=0; i<int_data_vec.size(); ++i) int_data_vec[i] = 1000 + static_cast<int>(i) * 5 + (rand() % 20 - 10); 
            data_loaded_ok=true; 
            std::cout << "Generated synthetic int data: " << int_data_vec.size() << std::endl;
        }
        // MODIFICATION 19: Pass new parameters to test function
        if (data_loaded_ok && !int_data_vec.empty()) run_compression_test(int_data_vec, "int", dataset_name, variance_block_multiplier, num_thresholds);
        else if(data_loaded_ok) std::cout << "Data empty for int." << std::endl;
        else if(!input_filename_str.empty()) std::cout << "Failed to load int data." << std::endl;

    } else if (data_type_arg_str == "long") {
        std::vector<long> long_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") data_loaded_ok = read_text_file(input_filename_str, long_data_vec);
            else if (file_type_flag_str == "--binary") data_loaded_ok = read_binary_file(input_filename_str, long_data_vec);
        } else { 
            long_data_vec.resize(1000000); 
            for(size_t i=0; i<long_data_vec.size(); ++i) long_data_vec[i] = 100000L + static_cast<long>(i) * 50L + (rand() % 100 - 50);
            data_loaded_ok=true; 
            std::cout << "Generated synthetic long data: " << long_data_vec.size() << std::endl;
        }
        if (data_loaded_ok && !long_data_vec.empty()) run_compression_test(long_data_vec, "long", dataset_name, variance_block_multiplier, num_thresholds);
        else if(data_loaded_ok) std::cout << "Data empty for long." << std::endl;
        else if(!input_filename_str.empty()) std::cout << "Failed to load long data." << std::endl;

    } else if (data_type_arg_str == "long_long" || data_type_arg_str == "longlong") {
        std::vector<long long> ll_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") data_loaded_ok = read_text_file(input_filename_str, ll_data_vec);
            else if (file_type_flag_str == "--binary") data_loaded_ok = read_binary_file(input_filename_str, ll_data_vec);
        } else { 
            std::cout << "Generating synthetic long long data..." << std::endl;
            ll_data_vec.resize(1000000);
            for (size_t i = 0; i < ll_data_vec.size(); i++) { 
                 ll_data_vec[i] = 1000000000LL + static_cast<long long>(i) * (500LL + (rand()%200 - 100)) + (rand() % 1000 - 500);
            }
            std::cout << "Synthetic data generated: " << ll_data_vec.size() << " long longs." << std::endl;
            data_loaded_ok = true;
        }
        if (data_loaded_ok && !ll_data_vec.empty()) run_compression_test(ll_data_vec, "long_long", dataset_name, variance_block_multiplier, num_thresholds);
        else if(data_loaded_ok) std::cout << "Data empty for long_long." << std::endl;
        else if(!input_filename_str.empty()) std::cout << "Failed to load long_long data." << std::endl;

    } else if (data_type_arg_str == "unsigned_long_long" || data_type_arg_str == "ull") {
        std::vector<unsigned long long> ull_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") data_loaded_ok = read_text_file(input_filename_str, ull_data_vec);
            else if (file_type_flag_str == "--binary") data_loaded_ok = read_binary_file(input_filename_str, ull_data_vec);
        } else { 
            std::cout << "Generating synthetic unsigned long long data..." << std::endl;
            ull_data_vec.resize(1000000);
            unsigned long long base_val = 1000000000000ULL;
            for (size_t i = 0; i < ull_data_vec.size(); i++) {
                ull_data_vec[i] = base_val + static_cast<unsigned long long>(i) * (500ULL + (rand()%200 - 100)) + (rand() % 1000 - 500);
            }
            std::cout << "Synthetic data generated: " << ull_data_vec.size() << " unsigned long longs." << std::endl;
            data_loaded_ok = true;
        }
        if (data_loaded_ok && !ull_data_vec.empty()) run_compression_test(ull_data_vec, "unsigned_long_long", dataset_name, variance_block_multiplier, num_thresholds);
        else if(data_loaded_ok) std::cout << "Data empty for unsigned_long_long." << std::endl;
        else if(!input_filename_str.empty()) std::cout << "Failed to load unsigned_long_long data." << std::endl;
        
    } else if (data_type_arg_str == "uint" || data_type_arg_str == "uint32" || data_type_arg_str == "unsigned_int") {
        std::vector<unsigned int> uint_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") data_loaded_ok = read_text_file(input_filename_str, uint_data_vec);
            else if (file_type_flag_str == "--binary") data_loaded_ok = read_binary_file(input_filename_str, uint_data_vec);
        } else { 
            std::cout << "Generating synthetic unsigned int data..." << std::endl;
            uint_data_vec.resize(1000000);
            for (size_t i = 0; i < uint_data_vec.size(); i++) {
                uint_data_vec[i] = 1000000U + static_cast<unsigned int>(i) * 50U + (rand() % 100);
            }
            std::cout << "Synthetic data generated: " << uint_data_vec.size() << " unsigned ints." << std::endl;
            data_loaded_ok = true;
        }
        if (data_loaded_ok && !uint_data_vec.empty()) run_compression_test(uint_data_vec, "unsigned_int", dataset_name, variance_block_multiplier, num_thresholds);
        
    } else {
        // MODIFICATION 20: Updated usage message
        std::cerr << "\nUsage: " << argv[0] << " <data_type> [--text/--binary FILENAME] [--multiplier M] [--thresholds T]" << std::endl;
        std::cerr << "Supported integer data types: int, long, long_long, ull, uint" << std::endl;
        std::cerr << "  --multiplier M: Sets the initial analysis block size to base_size * M (default: 8)." << std::endl;
        std::cerr << "  --thresholds T: Sets the number of variance thresholds to use for partitioning (default: 3)." << std::endl;
        std::cerr << "If no filename, synthetic data is used." << std::endl;
        return 1;
    }
    return 0;
}
#endif // GLECO2_LIB_MODE
