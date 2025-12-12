/**
 * L3 String Compression API
 *
 * Public header for GPU-accelerated string compression using learned models.
 */

#ifndef L3_STRING_CODEC_HPP
#define L3_STRING_CODEC_HPP

#include "L3_string_format.hpp"
#include "L3_string_utils.hpp"
#include <vector>
#include <string>

/**
 * Compress a collection of strings using GPU-accelerated learned compression
 *
 * @param strings Input strings to compress
 * @param config Compression configuration (auto-configured if default)
 * @param stats Output statistics (optional, can be nullptr)
 * @return Pointer to compressed data structure (caller owns)
 */
CompressedStringData* compressStrings(
    const std::vector<std::string>& strings,
    StringCompressionConfig config = StringCompressionConfig(),
    StringCompressionStats* stats = nullptr);

/**
 * Decompress strings from compressed data
 *
 * @param compressed Compressed string data
 * @param config Compression configuration (must match compression config)
 * @param output Output vector for decompressed strings
 * @param stats Output statistics (optional, can be nullptr)
 * @return Number of strings decompressed
 */
int decompressStrings(
    const CompressedStringData* compressed,
    const StringCompressionConfig& config,
    std::vector<std::string>& output,
    StringDecompressionStats* stats = nullptr);

/**
 * Free compressed string data
 *
 * @param compressed Compressed data to free
 */
void freeCompressedStringData(CompressedStringData* compressed);

/**
 * Print compression statistics
 */
void printStringCompressionStats(const StringCompressionStats& stats);

/**
 * Print decompression statistics
 */
void printStringDecompressionStats(const StringDecompressionStats& stats);

#endif // L3_STRING_CODEC_HPP
