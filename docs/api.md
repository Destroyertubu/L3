# API Reference

## Compression API

### `compressData()`

Compress data using L3 learned compression.

```cpp
template<typename T>
CompressedDataL3<T>* compressData(
    const std::vector<T>& h_data,
    int partition_size = 2048,
    CompressionStats* stats = nullptr
);
```

**Parameters**:
- `h_data`: Input data vector
- `partition_size`: Elements per partition (default: 2048)
- `stats`: Optional compression statistics output

**Returns**: Pointer to compressed data structure

**Example**:
```cpp
std::vector<uint32_t> data = {1, 2, 3, 4, 5};
auto compressed = compressData(data, 1024);
```

---

### `decompressData()`

Decompress L3-compressed data.

```cpp
template<typename T>
int decompressData(
    const CompressedDataL3<T>* compressed,
    std::vector<T>& h_output,
    DecompressionStats* stats = nullptr
);
```

**Parameters**:
- `compressed`: Compressed data structure
- `h_output`: Output vector (resized automatically)
- `stats`: Optional decompression statistics

**Returns**: 0 on success, error code otherwise

---

### `freeCompressedData()`

Free memory allocated for compressed data.

```cpp
template<typename T>
void freeCompressedData(CompressedDataL3<T>* data);
```

## Random Access API

### `randomAccessDecompress()`

Device function for random element access.

```cpp
template<typename T>
__device__ T randomAccessDecompress(
    const CompressedDataL3<T>* compressed_data,
    int global_idx
);
```

**Parameters**:
- `compressed_data`: Pointer to compressed data on device
- `global_idx`: Element index to access

**Returns**: Decompressed value at index

**Usage** (in CUDA kernel):
```cuda
__global__ void myKernel(CompressedDataL3<uint32_t>* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t value = randomAccessDecompress(data, idx);
    // Use value...
}
```

## Query Processing API

### Predicate Pushdown

```cpp
template<typename T>
void evaluatePredicateOnPartitions(
    const CompressedDataL3<T>* compressed,
    T min_value,
    T max_value,
    std::vector<int>& valid_partitions
);
```

**Parameters**:
- `compressed`: Compressed data with partition bounds
- `min_value`: Minimum value in predicate range
- `max_value`: Maximum value in predicate range
- `valid_partitions`: Output vector of non-pruned partition indices

## Data Structures

### `CompressedDataL3<T>`

Main compressed data structure.

```cpp
template<typename T>
struct CompressedDataL3 {
    // Partition metadata
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    int32_t* d_delta_bits;
    double*  d_model_params;
    int64_t* d_delta_array_bit_offsets;

    // Compressed data
    uint32_t* delta_array;
    long long* d_plain_deltas;

    // Partition bounds
    T* d_partition_min_values;
    T* d_partition_max_values;

    // Metadata
    int num_elements;
    int num_partitions;
    size_t compressed_size_bytes;
};
```

### `CompressionStats`

Compression statistics structure.

```cpp
struct CompressionStats {
    double compression_time_ms;
    size_t original_size_bytes;
    size_t compressed_size_bytes;
    double compression_ratio;
    int num_partitions;
    double avg_delta_bits;
};
```

## Configuration

### `L3Config`

Configuration for compression behavior.

```cpp
struct L3Config {
    int partition_size;          // Elements per partition
    bool enable_partition_bounds; // Compute min/max per partition
    bool use_plain_deltas;       // Store deltas uncompressed
    int max_delta_bits;          // Maximum bits for delta encoding
};
```

## Error Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| -1 | Invalid parameter |
| -2 | CUDA error |
| -3 | Memory allocation failed |
| -4 | Compression failed |
| -5 | Decompression failed |
