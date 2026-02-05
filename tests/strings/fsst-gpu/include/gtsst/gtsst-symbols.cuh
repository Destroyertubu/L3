#ifndef GTSST_SYMBOLS_CUH
#define GTSST_SYMBOLS_CUH
#include <fsst/fsst-lib.cuh>

namespace gtsst::symbols {
    struct TinySymbol {
        // Encodes (from least to most significant): code:8
        // Length is assumed to be stored in context (and escape = 255)
        uint8_t val;

        __host__ __device__ TinySymbol() : val(255) {
        }

        explicit TinySymbol(uint8_t code) {
            val = code;
        }

        __host__ __device__ [[nodiscard]] uint16_t code() const {
            return val;
        }
    };

    struct SmallSymbol {
        // Tries to encode same information as Symbol, but more space efficient for GPU (shared memory is limited :( )

        // We still need to store the original sample, in order to check for equality (we can have hash collisions)
        uint64_t val;

        // Same information as icl from Symbol (least to most significant), but slightly different:
        // ignoredBytes:3,code:8,length:4
        uint16_t icl;

        SmallSymbol() : icl((0b11111111) << 3 | 0b1111 << 11) {
            val = 0;
        }

        explicit SmallSymbol(fsst::Symbol s) {
            icl = 0;
            val = s.val.num;
            set_code_len(s.code(), s.length());
        }

        void set_code_len(uint16_t code, uint8_t len) {
            assert(len > 0);
            icl = (8 - len) & 0b111; // set ignored bytes
            icl |= (code & 0b11111111) << 3; // set code
            icl |= ((uint16_t)len) << 11;
        }

        __host__ __device__ [[nodiscard]] uint8_t length() const {
            return (uint8_t)(icl >> 11);
        }

        __host__ __device__ [[nodiscard]] uint16_t code() const {
            return (icl >> 3) & 0b11111111;
        }

        __host__ __device__ [[nodiscard]] uint8_t ignoredBytes() const {
            return (uint8_t)icl & 0b111;
        }

        __host__ __device__ [[nodiscard]] uint8_t first() const {
            assert(length() >= 1);
            return 0xFF & val;
        }

        __host__ __device__ [[nodiscard]] uint16_t first2() const {
            assert(length() >= 2);
            return 0xFFFF & val;
        }

        __host__ __device__ [[nodiscard]] size_t hash() const {
            size_t v = 0xFFFFFF & val;
            return FSST_HASH(v);
        } // hash on the next 3 bytes
    };

    // TODO: remove GPU symbol
    struct GPUSymbol {
        // TIM: Copy of Symbol, but for gpu and without introducing a dependency chain from hell
        // the byte sequence that this symbol stands for
        union {
            uint8_t str[fsst::Symbol::maxLength];

            // Note that this is little-endian with respect to the underlying uint8_ts. So if 'egal' is encoded, it will be
            // 0x65 0x67 0x75 0x6c, which translates to 0x6c756765 for num.
            // So [0x65, 0x67, 0x75, 0x6c, 0x00, 0x00, 0x00, 0x00] => 0x000000006c756765
            uint64_t num;
        } val{}; // usually we process it as a num(ber), as this is fast

        // icl = uint64_t ignoredBits:16,code:12,length:4,unused:32 -- but we avoid exposing this bit-field notation
        uint64_t icl{}; // use a single uint64_t to be sure "code" is accessed with one load and can be compared with one
                   // comparison

        __device__ __host__ GPUSymbol() {
            val.num = 0;
        }

        // __device__ explicit GPUSymbol(const uint8_t src[32][128+8], uint32_t i, uint32_t len) {
        //     val.num = 0;
        //     // len = min(len, Symbol::maxLength);
        //     // memcpy(val.str, input, len);
        //
        //     for (uint32_t j = 0; j < len; j++) {
        //         val.str[j] = src[threadIdx.x][i+j];
        //     }
        //
        //     set_code_len(FSST_CODE_MAX, len);
        // }

        __device__ __host__ explicit GPUSymbol(const uint64_t data, const uint8_t len) {
            val.num = data;
            set_code_len(FSST_CODE_MAX, len);
        }

        __device__ explicit GPUSymbol(const uint8_t* input, uint32_t len) {
#if KERNEL_MODIFICATION_SYMBOL_ALIGNED_READ == 1
            size_t offset = reinterpret_cast<uintptr_t>(input) % 8;
            const uint64_t* aligned_input = reinterpret_cast<const uint64_t*>(input - offset);

            uint64_t before = aligned_input[0];
            uint64_t after = aligned_input[1];

            val.num = (before >> (offset * 8)) | (after << ((8 - offset) * 8));
            val.num &= (1ULL << (len * 8)) - 1;
#else
            val.num = 0;
            // len = min(len, Symbol::maxLength);
            // memcpy(val.str, input, len);

            for (uint32_t i = 0; i < len; i++) {
                val.str[i] = input[i];
            }
#endif
            set_code_len(FSST_CODE_MAX, len);
        }

        __device__ __host__ void set_code_len(uint32_t code, uint32_t len) {
            icl = (len << 28) | (code << 16) | ((8 - len) * 8);
        }

        __device__ __host__ [[nodiscard]] uint32_t length() const {
            return (uint32_t)(icl >> 28);
        }

        __device__ __host__ [[nodiscard]] uint16_t code() const {
            return (icl >> 16) & FSST_CODE_MASK;
        }

        __device__ __host__ [[nodiscard]] uint32_t ignoredBits() const {
            return (uint32_t)icl;
        }

        __device__ __host__ [[nodiscard]] uint8_t first() const {
            assert(length() >= 1);
            return 0xFF & val.num;
        }

        __device__ __host__ [[nodiscard]] uint16_t first2() const {
            return 0xFFFF & val.num;
        }

        __device__ __host__ [[nodiscard]] uint8_t second() const {
            return (0xFF00 & val.num) >> 8;
        }

        __device__ __host__ [[nodiscard]] size_t hash() const {
            size_t v = 0xFFFFFF & val.num;
            return FSST_HASH(v);
        } // hash on the next 3 bytes
    };

    struct ComparableSmallSymbol {
        uint32_t val1 = 0, val2 = 0; // lsb, msb

        // Same information as icl from Symbol (least to most significant), but slightly different:
        // ignoredBytes:3,code:8,length:4
        uint16_t icl = 0;

        ComparableSmallSymbol() : icl((0b11111111) << 3 | 0b1111 << 11) {
        }

        explicit ComparableSmallSymbol(fsst::Symbol s) {
            icl = 0;
            const uint64_t val = s.val.num & (0xFFFFFFFFFFFFFFFF >> ignoredBytes() * 8);
            val1 = (uint32_t)val;
            val2 = (uint32_t)(val >> 32);
            set_code_len(s.code(), s.length());
        }

        __host__ __device__ void set_code_len(uint16_t code, uint8_t len) {
            assert(len > 0);
            icl = (8 - len) & 0b111; // set ignored bytes
            icl |= (code & 0b11111111) << 3; // set code
            icl |= ((uint16_t)len) << 11;
        }

        __host__ __device__ [[nodiscard]] uint8_t length() const {
            return (uint8_t)(icl >> 11);
        }

        __host__ __device__ [[nodiscard]] uint16_t code() const {
            return (icl >> 3) & 0b11111111;
        }

        __host__ __device__ [[nodiscard]] uint8_t ignoredBytes() const {
            return (uint8_t)icl & 0b111;
        }

        __host__ __device__ [[nodiscard]] size_t hash() const {
            size_t v = 0xFFFFFF & val1;
            return FSST_HASH(v);
        } // hash on the next 3 bytes

        __host__ __device__ [[nodiscard]] bool match(const fsst::Symbol s) const {
            const uint64_t relevant_val = s.val.num & (0xFFFFFFFFFFFFFFFF >> ignoredBytes() * 8);
            const bool content = ((uint64_t)val1 | (uint64_t)val2 << 32) == relevant_val;

            return content;
        }

        __host__ __device__ [[nodiscard]] bool match(const GPUSymbol s) const {
            const uint64_t relevant_val = s.val.num & (0xFFFFFFFFFFFFFFFF >> ignoredBytes() * 8);
            const bool content = ((uint64_t)val1 | (uint64_t)val2 << 32) == relevant_val;

            return content;
        }
    };

    __host__ __device__ inline uint8_t get_mask_equality(const uint8_t a, const uint8_t b) {
        return -(a == b);
    }

    // Returns val if b == check, otherwise 0
    __host__ __device__ inline uint8_t get_value_if_equal(const uint8_t b, const uint8_t check, const uint8_t val) {
        return get_mask_equality(b, check) & val;
    }

    /*
     * SymbolMatch is a way to encode 2 matches with one 32-bit number. This is useful for 2-byte matches,
     * where the first byte determines which SymbolMatch (or multiple) to use, and then this struct can check for two
     * matches with a single 32-bit memory fetch (like shared memory)
     */
    struct SymbolMatch {
        uint32_t val;

        SymbolMatch() : val(0) {
        }

        SymbolMatch(const uint8_t symbol1, const uint8_t code1, const uint8_t symbol2, const uint8_t code2) :
            val(symbol1 << 24 | code1 << 16 | symbol2 << 8 | code2) {
        }

        __host__ __device__ [[nodiscard]] uint8_t first_symbol() const {
            return val >> 24;
        }

        __host__ __device__ [[nodiscard]] uint8_t first_code() const {
            return val >> 16;
        }

        __host__ __device__ [[nodiscard]] uint8_t second_symbol() const {
            return val >> 8;
        }

        __host__ __device__ [[nodiscard]] uint8_t second_code() const {
            return val;
        }

        // Returns a code if the given symbol matches either of the two symbols, otherwise 0
        __host__ __device__ [[nodiscard]] uint8_t match(const uint8_t symbol) const {
            return get_value_if_equal(symbol, first_symbol(), first_code()) |
                get_value_if_equal(symbol, second_symbol(), second_code());
        }
    };
} // namespace gtsst::symbols

#endif // GTSST_SYMBOLS_CUH
