#ifndef GTSST_TABLES_CUH
#define GTSST_TABLES_CUH
#include <gtsst/gtsst-symbols.cuh>

namespace gtsst::symbols {
    struct PlainSymbolTableData {
        SmallSymbol hashTab[fsst::SymbolTable::hashTabSize];
        uint8_t shortCodes[65536];
        uint8_t singleCodes[256];

        PlainSymbolTableData() = default;

        explicit PlainSymbolTableData(const fsst::SymbolTable& st) {
            for (int i = 0; i < fsst::SymbolTable::hashTabSize; i++) {
                hashTab[i] = SmallSymbol();
            }

            memset(shortCodes, fsst::Symbol::escape, sizeof(shortCodes));
            memset(singleCodes, fsst::Symbol::escape, sizeof(singleCodes));

            // Convert symbols from old to new
            for (int i = 0; i < st.nSymbols; i++) {
                fsst::Symbol s = st.symbols[i];
                if (s.length() == 1) {
                    singleCodes[s.first()] = s.code();
                } else if (s.length() == 2) {
                    shortCodes[s.first2()] = s.code();
                } else {
                    auto small = SmallSymbol(s);
                    assert(small.hash() == s.hash());

                    hashTab[small.hash() & fsst::SymbolTable::hashTabSize - 1] = small;
                }
            }
        }

        // Returns code:8, len: 4, code=255=escape
        [[nodiscard]] uint16_t findLongestSymbol(const fsst::Symbol sym) const {
            size_t idx = sym.hash() & (fsst::SymbolTable::hashTabSize - 1);

            SmallSymbol s = hashTab[idx];
            uint8_t ignoredBits = s.ignoredBytes() * 8;
            uint64_t num = sym.val.num & (0xFFFFFFFFFFFFFFFF >> ignoredBits);

            if (s.code() != fsst::Symbol::escape && s.val == num) {
                uint16_t codeAndLen = s.icl >> 3;
                return codeAndLen;
            }

            const uint16_t twoByte = shortCodes[sym.first2()];
            if (twoByte != fsst::Symbol::escape)
                return twoByte | (2 << 8);

            const uint16_t singleByte = singleCodes[sym.first()];
            return singleByte | (1 << 8);
        }
    };

  static_assert(std::is_constructible_v<PlainSymbolTableData, fsst::SymbolTable&>);
} // namespace gtsst::symbols

#endif // GTSST_TABLES_CUH
