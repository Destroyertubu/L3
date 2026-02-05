#ifndef GTSST_MATCH_TABLE_CUH
#define GTSST_MATCH_TABLE_CUH
#include "gtsst-symbols.cuh"
#include <fsst/fsst-lib.cuh>

namespace gtsst::symbols {
    /*
     * SymbolMatchTable uses SymbolMatch to support matching on 2-byte symbols, with the idea that a single 32-bit value
     * can support two symbols and two codes. This means you can do two matches on a single 32-bit load from shared
     * memory, and if the total amount of available rows is 32 there will be no conflicts.
     *
     * The idea is that to match a 2-byte symbol against all options is as follows:
     *  - Use the first symbol to pick a 'row'
     *  - Then request all SymbolMatch objects in a row (spaced 32 apart, so no conflicts* -use padding if needed-)
     *  - For every SymbolMatch, run the match and OR it against the current result
     *  - This will either result in a 0 (for no match) or a value
     *  - If 0, then make it 255 to indicate no match
     *
     *  * No conflicts because all rows are stored column-major, so threads will either query different banks or the
     * same address (resulting in a broadcast)
     */
    struct SymbolMatchTable {
        static constexpr uint8_t rows = fsst::SymbolTable::maxRowsTwo; // Should just be 32
        static constexpr uint8_t matchesPerRow = 4; // Basically the amount of lookups per symbol compare (and therefore
                                                    // also the amount of possible 2-byte symbols per row)

        SymbolMatch matches[rows * matchesPerRow];
        uint8_t row_indices[256]{}; // TODO: there might be potential to also use this to encode single byte codes.

        SymbolMatchTable() = default;

        explicit SymbolMatchTable(TinySymbol shortCodes[65536]) {
            // Mark all row_indices as escape, will set the right ones later
            memset(row_indices, 255, 256);

            assert(fsst::SymbolTable::maxSameRowTwo <=
                   matchesPerRow * 2); // Ensure this match table can cover all symbols in a row

            // Get rows with values
            uint16_t values[rows][matchesPerRow * 2] = {}; // Every match can match 2 symbols

            uint8_t usedRows = 0;
            for (uint16_t a = 0; a < 256; a++) {
                bool matches = false;
                int i = 0;

                for (uint16_t b = 0; b < 256; b++) {
                    // If there is a symbol, save it in temporary storage
                    if (TinySymbol ts = shortCodes[a | b << 8]; ts.code() != 255) {
                        assert(usedRows < rows);

                        matches = true;
                        values[usedRows][i] = b << 8 | ts.code() + 1;
                        i += 1;
                    }
                }

                assert(i <= matchesPerRow * 2);

                // If any 2-byte symbol is found in this row, save it
                if (matches) {
                    row_indices[a] = usedRows;
                    usedRows += 1;
                }
            }

            // And now construct all the match structs
            for (uint8_t row = 0; row < usedRows; row++) {
                for (int i = 0; i < matchesPerRow; i++) {
                    const uint16_t match1 = values[row][i * 2];
                    const uint16_t match2 = values[row][i * 2 + 1];

                    assert(((uint8_t)match1) != 0 ||
                           match1 >> 8 == 0); // Ensure that a non-zero symbol does not have a zero code
                    assert(((uint8_t)match2) != 0 ||
                           match2 >> 8 == 0); // Ensure that a non-zero symbol does not have a zero code

                    // Store in column major
                    matches[i * rows + row] = SymbolMatch(match1 >> 8, match1, match2 >> 8, match2);
                }
            }
        }

        __host__ __device__ [[nodiscard]] uint8_t lookup(const uint8_t a, const uint8_t b) const {
            // First do a row lookup
            const uint8_t row = row_indices[a];

            // If there is no row, then already return an escape since there are no 2-byte symbols starting with 'a'
            if (row == 255) {
                return 255;
            }

            assert(row < rows); // Ensure this is a valid row lookup

            // Then do matchesPerRow lookups, which contain 2 lookups each
            uint8_t result = 0;
            for (int i = 0; i < matchesPerRow; i++) {
                SymbolMatch match = matches[rows * i + row];

                assert(result == 0 |
                       match.match(b) ==
                           0); // Check there are no two matches, so either there is match yet or there is no new match

                result |= match.match(b);
            }

            return result - 1;
        }
    };

    struct SmallSymbolMatchTableData {
        TinySymbol singleCodes[256]; // 1-byte symbols
        ComparableSmallSymbol hashes[fsst::SymbolTable::maxHashUsage + 1]; // 3+-byte symbols (and one escape symbol)
        uint8_t hashTab[fsst::SymbolTable::hashTabSize]{}; // 3+-byte symbol hashes
        SymbolMatchTable matchTable; // 2-byte symbols using match table representation

        SmallSymbolMatchTableData() = default;

        explicit SmallSymbolMatchTableData(const fsst::SymbolTable& st) {
            TinySymbol shortCodes[65536];
            int hashIdx = 0;

            // Let all empty spots point to the escape symbol
            memset(hashTab, fsst::SymbolTable::maxHashUsage, sizeof(hashTab));
            hashes[fsst::SymbolTable::maxHashUsage] = ComparableSmallSymbol();

            // Convert symbols from old to new
            for (int i = 0; i < st.nSymbols; i++) {
                fsst::Symbol s = st.symbols[i];

                assert(s.code() != fsst::Symbol::ignore);

                if (s.length() == 1) {
                    singleCodes[s.val.num & 0xFF] = TinySymbol(s.code());
                } else if (s.length() == 2) {
                    shortCodes[s.val.num & 0xFFFF] = TinySymbol(s.code());
                } else {
                    auto small = ComparableSmallSymbol(s);
                    assert(small.hash() == s.hash());

                    hashes[hashIdx] = small;
                    hashTab[small.hash() & fsst::SymbolTable::hashTabSize - 1] = hashIdx;
                    hashIdx += 1;
                }
            }

            matchTable = SymbolMatchTable(shortCodes);
        }

        // Returns code:8, len: 4, code=255=escape
        [[nodiscard]] uint16_t findLongestSymbol(const fsst::Symbol sym) const {
            const size_t idx = sym.hash() & (fsst::SymbolTable::hashTabSize - 1);

            const uint8_t hashIdx = hashTab[idx];
            const ComparableSmallSymbol s = hashes[hashIdx];
            const uint16_t twoByte = matchTable.lookup(sym.first(), sym.second());
            const uint16_t singleByte = singleCodes[sym.first()].code();

            if (s.code() != fsst::Symbol::escape && s.match(sym)) {
                const uint16_t codeAndLen = s.icl >> 3;
                return codeAndLen;
            }

            if (twoByte != fsst::Symbol::escape)
                return twoByte | (2 << 8);

            return singleByte | (1 << 8);
        }

        __host__ __device__ [[nodiscard]] uint16_t findLongestSymbol(const GPUSymbol sym) const {
            size_t idx = sym.hash() & fsst::SymbolTable::hashTabSize - 1;

            const uint8_t hashIdx = hashTab[idx];
            const ComparableSmallSymbol s = hashes[hashIdx];
            const uint16_t twoByte = matchTable.lookup(sym.first(), sym.second());
            const uint16_t singleByte = singleCodes[sym.first()].code();

            if (s.code() != fsst::Symbol::escape && s.match(sym)) {
                const uint16_t codeAndLen = s.icl >> 3;
                return codeAndLen;
            }

            if (twoByte != fsst::Symbol::escape)
                return twoByte | (2 << 8);

            // TODO: remove debug catch
            if (singleByte == fsst::Symbol::ignore) {
                uint8_t x = sym.first();
                idx += x;
                idx -= x;
                assert(false);
            }

            return singleByte | (1 << 8);
        }
    };

    static_assert(std::is_constructible_v<SmallSymbolMatchTableData, fsst::SymbolTable&>);
} // namespace gtsst::symbols

#endif // GTSST_MATCH_TABLE_CUH
