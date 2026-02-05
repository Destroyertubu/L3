#ifndef GTSST_ELL_TABLE_CUH
#define GTSST_ELL_TABLE_CUH

#include <gtsst/gtsst-match-table.cuh>
#include <gtsst/gtsst-symbols.cuh>

namespace gtsst::symbols {
    struct SparseShortCodes {
        static constexpr int K = fsst::SymbolTable::maxSameRowTwo;
        static constexpr int M = 255;
        TinySymbol values[K * M];
        uint8_t col_indices[K * M]{};

        SparseShortCodes() = default;

        explicit SparseShortCodes(TinySymbol shortCodes[65536]) {
            memset(col_indices, 255, K * M * sizeof(uint8_t));

            for (int a = 0; a < 256; a++) {
                int i = 0;
                for (int b = 0; b < 256; b++) {
                    // If this shortcodes is not an escape, add it to the ELL format
                    if (TinySymbol ts = shortCodes[a | b << 8]; ts.code() != 255) {
                        values[a * K + i] = ts;
                        col_indices[a * K + i] = b;
                        i += 1;
                    }
                }
            }
        }

        __host__ __device__ [[nodiscard]] TinySymbol lookup(uint8_t a, uint8_t b) const {
            if (b == 255)
                return {};

            for (int i = 0; i < fsst::SymbolTable::maxSameRowTwo; i++) {
                uint8_t indice = col_indices[a * fsst::SymbolTable::maxSameRowTwo + i];

                if (indice == b) {
                    return values[a * fsst::SymbolTable::maxSameRowTwo + i];
                }
            }

            return {};
        }

        [[nodiscard]] static size_t size() {
            return sizeof(TinySymbol) * M * K * 2;
        }
    };

    struct SmallSymbolTableData {
        TinySymbol singleCodes[256]; // 1-byte symbols
        SmallSymbol hashes[fsst::SymbolTable::maxHashUsage + 1]; // 3+-byte symbols (and one escape symbol)
        uint8_t hashTab[fsst::SymbolTable::hashTabSize]{}; // 3+-byte symbol hashes
        SparseShortCodes sparseShortCodes; // 2-byte symbols using ELL representation

        SmallSymbolTableData() = default;

        explicit SmallSymbolTableData(const fsst::SymbolTable& st) {
            TinySymbol shortCodes[65536];
            int hashIdx = 0;

            // Let all empty spots point to the escape symbol
            memset(hashTab, fsst::SymbolTable::maxHashUsage, sizeof(hashTab));
            hashes[fsst::SymbolTable::maxHashUsage] = SmallSymbol();

            // Convert symbols from old to new
            for (int i = 0; i < st.nSymbols; i++) {
                if (fsst::Symbol s = st.symbols[i]; s.length() == 1) {
                    singleCodes[s.val.num & 0xFF] = TinySymbol(s.code());
                } else if (s.length() == 2) {
                    shortCodes[s.val.num & 0xFFFF] = TinySymbol(s.code());
                } else {
                    auto small = SmallSymbol(s);
                    assert(small.hash() == s.hash());

                    hashes[hashIdx] = small;
                    hashTab[small.hash() & fsst::SymbolTable::hashTabSize - 1] = hashIdx;
                    hashIdx += 1;
                }
            }

            sparseShortCodes = SparseShortCodes(shortCodes);
        }

        // Returns code:8, len: 4, code=255=escape
        [[nodiscard]] uint16_t findLongestSymbolELL(const fsst::Symbol sym) const {
            size_t idx = sym.hash() & (fsst::SymbolTable::hashTabSize - 1);

            uint8_t hashIdx = hashTab[idx];
            SmallSymbol s = hashes[hashIdx];
            uint8_t ignoredBits = s.ignoredBytes() * 8;
            uint64_t num = sym.val.num & (0xFFFFFFFFFFFFFFFF >> ignoredBits);

            if (s.code() != fsst::Symbol::escape && s.val == num) {
                uint16_t codeAndLen = s.icl >> 3;
                return codeAndLen;
            }

            uint16_t twoByte = sparseShortCodes.lookup(sym.first(), sym.second()).code();
            if (twoByte != fsst::Symbol::escape)
                return twoByte | (2 << 8);

            uint16_t singleByte = singleCodes[sym.first()].code();
            return singleByte | (1 << 8);
        }
    };

    static_assert(std::is_constructible_v<SmallSymbolTableData, fsst::SymbolTable&>);
} // namespace gtsst::symbols

#endif // GTSST_ELL_TABLE_CUH
