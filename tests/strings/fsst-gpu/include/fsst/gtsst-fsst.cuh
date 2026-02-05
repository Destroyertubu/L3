#ifndef GTSST_FSST_HPP
#define GTSST_FSST_HPP
#include <cstdint>
#include <fsst/fsst-lib.cuh>
#include <fsst/fsst.cuh>
#include <queue>
#include <unordered_set>

template <>
struct std::hash<gtsst::fsst::QSymbol> {
    size_t operator()(const gtsst::fsst::QSymbol& q) const noexcept {
        uint64_t k = q.symbol.val.num;
        const uint64_t m = 0xc6a4a7935bd1e995;
        const int r = 47;
        uint64_t h = 0x8445d61a4e774912 ^ (8 * m);
        k *= m;
        k ^= k >> r;
        k *= m;
        h ^= k;
        h *= m;
        h ^= h >> r;
        h *= m;
        h ^= h >> r;
        return h;
    }
};

namespace gtsst::fsst {
    template <typename Encoding>
        requires std::constructible_from<Encoding, SymbolTable&>
    struct EncodingTable final {
        // Used for encoding the decoding table
        uint16_t number_of_symbols{}; // amount of symbols in the map (max 255)
        uint16_t len_histogram[FSST_CODE_BITS]{}; // lenHisto[x] is the amount of symbols of byte-length (x+1) in this
                                                  // SymbolTable
        Symbol symbols[SymbolTable::maxSize+1]{};

        // Encoding data
        Encoding encoding_data;

        explicit EncodingTable(const SymbolTable& st) : encoding_data(st) {
            // Copy encoding data
            number_of_symbols = st.nSymbols;
            for (int i = 0; i < FSST_CODE_BITS; i++) {
                len_histogram[i] = st.lenHisto[i];
            }

            // Copy symbols for encoding
            for (int i = 0; i < st.nSymbols; i++) {
                symbols[i] = st.symbols[i];
            }
        }

        size_t export_table(uint8_t* dst) const {
            uint64_t version = (FSST_VERSION << 32) | // version is 24 bits, most significant byte is 0
                (((uint64_t)number_of_symbols) << 8) | FSST_ENDIAN_MARKER; // least significant byte is nonzero

            /* do not assume unaligned reads here */
            std::memcpy(dst, &version, 8);
            dst[8] = 0;
            for (int i = 0; i < 8; i++)
                dst[9 + i] = (uint8_t)len_histogram[i];
            uint32_t pos = 17;

            // emit only the used bytes of the symbols
            for (int i = 0; i < number_of_symbols; i++) {
                for (int j = 0; j < symbols[i].length(); j++) {
                    dst[pos] = symbols[i].val.num >> j * 8 & 0xFF; // serialize used symbol bytes
                    pos += 1;
                }
            }

            return pos; // length of what was serialized
        }
    };

    struct DecodingTable {
        fsst_decoder_t decoder;

        size_t import_table(const uint8_t* src);
    };

    size_t simple_make_sample(uint8_t* sample_buf, const uint8_t* src, size_t len);

    int compress_count_single(SymbolTable* st, Counters& counters, size_t sampleFrac, const uint8_t* line, size_t len);
    void make_table(SymbolTable* st, Counters& counters, size_t sampleFrac);

    template <typename Encoding>
        requires std::constructible_from<Encoding, SymbolTable&>
    EncodingTable<Encoding>* build_symbol_table(const uint8_t* sample_buf, size_t len) {
        auto* encoder = new Encoder();
        auto *st = new SymbolTable(), *bestTable = new SymbolTable();
        int bestGain = (int)-FSST_SAMPLEMAXSZ; // worst case (everything exception)
        size_t sampleFrac = 128;

        uint8_t bestCounters[512 * sizeof(uint16_t)];
        for (sampleFrac = 8; true; sampleFrac += 30) {
            memset(&encoder->counters, 0, sizeof(Counters));
            long gain = compress_count_single(st, encoder->counters, sampleFrac, sample_buf, len);
            if (gain >= bestGain) {
                // a new best solution!
                encoder->counters.backup1(bestCounters);
                *bestTable = *st;
                bestGain = gain;
            }
            if (sampleFrac >= 128)
                break; // we do 5 rounds (sampleFrac=8,38,68,98,128)
            make_table(st, encoder->counters, sampleFrac);
        }
        delete st;
        encoder->counters.restore1(bestCounters);
        make_table(bestTable, encoder->counters, sampleFrac);
        bestTable->finalize_simple_decreasing(); // renumber codes for more efficient compression

        // Ensure no padding symbol is returned
        uint8_t padding_lookup = bestTable->findLongestSymbol(Symbol(254, 0));
        assert(padding_lookup != 255 && padding_lookup != 254);

        // Create encoding table
        const auto table = new EncodingTable<Encoding>(*bestTable);

        // Done with creating table
        delete encoder;
        delete bestTable;
        return table;
    }
} // namespace gtsst::fsst

#endif // GTSST_FSST_HPP
