#ifndef FSST_LIB_HPP
#define FSST_LIB_HPP
#include "fsst.cuh"


#include <cassert>
#include <cstring>
#include <memory>

namespace gtsst::fsst {
#define FSST_ENDIAN_MARKER ((uint64_t)1)
#define FSST_VERSION_20190218 20190218
#define FSST_VERSION_20240131 20240131
#define FSST_VERSION ((uint64_t)FSST_VERSION_20240131)

#define FSST_SAMPLELINE ((size_t)512)
    static_assert(FSST_SAMPLELINE == 512); // From Tim: required to make gpu_sampling work (fix that function if you want to change this value)

    // "symbols" are character sequences (up to 8 bytes)
    // A symbol is compressed into a "code" of, in principle, one byte. But, we added an exception mechanism:
    // byte 255 followed by byte X represents the single-byte symbol X. Its code is 256+X.

    // we represent codes in uint16_t (not uint8_t). 12 bits code (of which 10 are used), 4 bits length
#define FSST_LEN_BITS 12
#define FSST_CODE_BITS 9
#define FSST_CODE_BASE 256UL /* first 256 codes [0,255] are pseudo codes: escaped bytes */
#define FSST_CODE_MAX                                                                                                  \
    (1UL << FSST_CODE_BITS) /* all bits set: indicating a symbol that has not been assigned a code yet */ // TIM NOTE:
    // 512 = 0b 10
    // 0000 0000


#define FSST_CODE_MASK                                                                                                 \
    (FSST_CODE_MAX -                                                                                                   \
     1UL) /* all bits set: indicating a symbol that has not been assigned a code yet */ // TIM NOTE: 511 = 0b 01 1111
    // 1111


    inline uint64_t fsst_unaligned_load(uint8_t const* V) {
        uint64_t Ret;
        memcpy(&Ret, V, sizeof(uint64_t)); // compiler will generate efficient code (unaligned load, where possible)
        return Ret;
    }

    struct Symbol {
        static const unsigned maxLength = 8; // TIM NOTE: 8 => 3
        static constexpr uint8_t escape = FSST_ESC;
        static constexpr uint8_t ignore = escape - 1;
        static constexpr uint8_t skip = escape - 2;

        // the byte sequence that this symbol stands for
        union {
            char str[maxLength];
            uint64_t num;
        } val; // usually we process it as a num(ber), as this is fast

        // icl = uint64_t ignoredBits:16,code:12,length:4,unused:32 -- but we avoid exposing this bit-field notation
        uint64_t icl; // use a single uint64_t to be sure "code" is accessed with one load and can be compared with one
        // comparison

        Symbol() : icl(0) {
            val.num = 0;
        }

        explicit Symbol(uint8_t c, uint16_t code) : icl((1 << 28) | (code << 16) | 56) {
            val.num = c;
        } // single-char symbol
        explicit Symbol(const char* begin, const char* end) : Symbol(begin, (uint32_t)(end - begin)) {
        }

        explicit Symbol(const uint8_t* begin, const uint8_t* end) :
            Symbol((const char*)begin, (uint32_t)(end - begin)) {
        }

        explicit Symbol(const char* input, uint32_t len) {
            val.num = 0;
            if (len >= 8) {
                len = 8;
                memcpy(val.str, input, 8);
            } else {
                memcpy(val.str, input, len);
            }
            set_code_len(FSST_CODE_MAX, len);
        }

        // TIM NOTE: added to have uchar instead of char, doesnt really matter I think?
        explicit Symbol(const uint8_t* input, uint32_t len) : Symbol((const char*)input, len) {
        }

        void set_code_len(uint32_t code, uint32_t len) {
            icl = (len << 28) | (code << 16) | ((8 - len) * 8);
        }

        uint32_t length() const {
            return (uint32_t)(icl >> 28);
        }

        uint16_t code() const {
            return (icl >> 16) & FSST_CODE_MASK;
        }

        uint32_t ignoredBits() const {
            return (uint32_t)icl;
        }

        uint8_t first() const {
            assert(length() >= 1);
            return 0xFF & val.num;
        }

        uint16_t first2() const {
            /*assert( length() >= 2);*/
            return 0xFFFF & val.num;
        }

        uint8_t second() const {
            /*assert( length() >= 2);*/
            return (0xFF00 & val.num) >> 8;
        }

#define FSST_HASH_LOG2SIZE 10
#define FSST_HASH_PRIME 2971215073LL
#define FSST_SHIFT 15
#define FSST_HASH(w) (((w) * FSST_HASH_PRIME) ^ (((w) * FSST_HASH_PRIME) >> FSST_SHIFT))

        size_t hash() const {
            size_t v = 0xFFFFFF & val.num;
            return FSST_HASH(v);
        } // hash on the next 3 bytes
    };

    // Symbol that can be put in a queue, ordered on gain
    struct QSymbol {
        Symbol symbol;
        mutable uint32_t gain; // mutable because gain value should be ignored in find() on unordered_set of QSymbols
        bool operator==(const QSymbol& other) const {
            return symbol.val.num == other.symbol.val.num && symbol.length() == other.symbol.length();
        }
    };

    // we construct FSST symbol tables using a random sample of about 16KB (1<<14)
#define FSST_SAMPLETARGET (1 << 14)
#define FSST_SAMPLEMAXSZ ((long)2 * FSST_SAMPLETARGET)
    static_assert(FSST_SAMPLETARGET % FSST_SAMPLELINE == 0);  // From Tim: required to make gpu_sampling work (fix that function if you want to change this value)

    // two phases of compression, before and after optimize():
    //
    // (1) to encode values we probe (and maintain) three datastructures:
    // - uint16_t byteCodes[256] array at the position of the next byte  (s.length==1)
    // - uint16_t shortCodes[65536] array at the position of the next twobyte pattern (s.length==2)
    // - Symbol hashtable[1024] (keyed by the next three bytes, ie for s.length>2),
    // this search will yield a uint16_t code, it points into Symbol symbols[]. You always find a hit, because the first 256
    // codes are pseudo codes representing a single byte these will become escapes)
    //
    // (2) when we finished looking for the best symbol table we call optimize() to reshape it:
    // - it renumbers the codes by length (first symbols of length 2,3,4,5,6,7,8; then 1 (starting from byteLim are symbols
    // of length 1)
    //   length 2 codes for which no longer suffix symbol exists (< suffixLim) come first among the 2-byte codes
    //   (allows shortcut during compression)
    // - for each two-byte combination, in all unused slots of shortCodes[], it enters the byteCode[] of the symbol
    // corresponding
    //   to the first byte (if such a single-byte symbol exists). This allows us to just probe the next two bytes (if there
    //   is only one byte left in the string, there is still a terminator-byte added during compression) in shortCodes[].
    //   That is, byteCodes[] and its codepath is no longer required. This makes compression faster. The reason we use
    //   byteCodes[] during symbolTable construction is that adding a new code/symbol is expensive (you have to touch
    //   shortCodes[] in 256 places). This optimization was hence added to make symbolTable construction faster.
    //
    // this final layout allows for the fastest compression code, only currently present in compressBulk

    // in the hash table, the icl field contains (low-to-high) ignoredBits:16,code:12,length:4
#define FSST_ICL_FREE                                                                                                  \
    (((uint32_t) 15 << 28) |                                                                                                      \
     (((uint32_t)FSST_CODE_MASK) << 16)) // high bits of icl (len=8,code=FSST_CODE_MASK) indicates free bucket


    // ignoredBits is (8-length)*8, which is the amount of high bits to zero in the input word before comparing with the
    // hashtable key
    //             ..it could of course be computed from len during lookup, but storing it precomputed in some loose
    //             bits is faster
    //
    // the gain field is only used in the symbol queue that sorts symbols on gain

    struct SymbolTable {
        static const uint32_t hashTabSize = 1 << FSST_HASH_LOG2SIZE; // smallest size that incurs no precision loss
        static const uint8_t maxSize = Symbol::skip; // 255 is reserved for escape, 254 for padding, 253 to map padding to something else
        static const uint8_t maxSameRowTwo = 8;
        static const uint8_t maxRowsTwo = 32; // 32 because then there are no bank conflicts
        // TIM NOTE: added this to limit size of rows in shortCodes (for sparse matrix later)
        static const uint8_t maxHashUsage = 127; // TIM NOTE: added this to limit size of hashes in hashTab

        // lookup table using the next two bytes (65536 codes), or just the next single byte
        uint16_t shortCodes[65536]; // contains code for 2-byte symbol, otherwise code for pseudo byte (escaped byte)

        // lookup table (only used during symbolTable construction, not during normal text compression)
        uint16_t byteCodes[256]; // contains code for every 1-byte symbol, otherwise code for pseudo byte (escaped byte)

        // 'symbols' is the current symbol  table symbol[code].symbol is the max 8-byte 'symbol' for single-byte 'code'
        Symbol symbols[FSST_CODE_MAX];
        // x in [0,255]: pseudo symbols representing escaped byte x; x in [FSST_CODE_BASE=256,256+nSymbols]: real
        // symbols

        // replicate long symbols in hashTab (avoid indirection).
        Symbol hashTab[hashTabSize]; // used for all symbols of 3 and more bytes

        uint16_t nSymbols; // amount of symbols in the map (max 255)
        uint16_t suffixLim; // codes higher than this do not have a longer suffix
        uint16_t terminator; // code of 1-byte symbol, that can be used as a terminator during compression
        bool zeroTerminated;
        // whether we are expecting zero-terminated strings (we then also produce zero-terminated compressed strings)
        uint16_t
        lenHisto[FSST_CODE_BITS]; // lenHisto[x] is the amount of symbols of byte-length (x+1) in this SymbolTable

        // TIM NOTE: added max row, to add a mechanism that only allows X amount (maxSameRowTwo) of the same row entries
        // in the shortcodes matrix
        uint8_t maxRow[256];
        uint8_t maxHash;
        uint8_t rows_used = 0;

        SymbolTable() : nSymbols(0), suffixLim(FSST_CODE_MAX), terminator(0), zeroTerminated(false) {
            // stuff done once at startup
            for (uint32_t i = 0; i < 256; i++) {
                symbols[i] = Symbol(i, i | (1 << FSST_LEN_BITS)); // pseudo symbols
            }
            Symbol unused = Symbol((uint8_t)0, FSST_CODE_MASK); // single-char symbol, exception code
            for (uint32_t i = 256; i < FSST_CODE_MAX; i++) {
                symbols[i] = unused; // we start with all symbols unused
            }
            // empty hash table
            Symbol s;
            s.val.num = 0;
            s.icl = FSST_ICL_FREE; // marks empty in hashtab
            for (uint32_t i = 0; i < hashTabSize; i++)
                hashTab[i] = s;

            // fill byteCodes[] with the pseudo code all bytes (escaped bytes)
            for (uint32_t i = 0; i < 256; i++)
                byteCodes[i] = (1 << FSST_LEN_BITS) | i;

            // fill shortCodes[] with the pseudo code for the first byte of each two-byte pattern
            for (uint32_t i = 0; i < 65536; i++)
                shortCodes[i] = (1 << FSST_LEN_BITS) | (i & 255);

            memset(lenHisto, 0, sizeof(lenHisto)); // all unused
            memset(maxRow, 0, sizeof(maxRow)); // all unused
            maxHash = 0;
        }

        void clear() {
            // clear a symbolTable with minimal effort (only erase the used positions in it)
            memset(lenHisto, 0, sizeof(lenHisto)); // all unused
            memset(maxRow, 0, sizeof(maxRow)); // unused
            maxHash = 0;
            rows_used = 0;
            for (uint32_t i = FSST_CODE_BASE; i < FSST_CODE_BASE + nSymbols; i++) {
                if (symbols[i].length() == 1) {
                    uint16_t val = symbols[i].first();
                    byteCodes[val] = (1 << FSST_LEN_BITS) | val;
                } else if (symbols[i].length() == 2) {
                    uint16_t val = symbols[i].first2();
                    shortCodes[val] = (1 << FSST_LEN_BITS) | (val & 255);
                } else {
                    uint32_t idx = symbols[i].hash() & (hashTabSize - 1);
                    hashTab[idx].val.num = 0;
                    hashTab[idx].icl = FSST_ICL_FREE; // marks empty in hashtab
                }
            }
            nSymbols = 0; // no need to clean symbols[] as no symbols are used
        }

        bool hashInsert(Symbol s) {
            if (maxHash >= SymbolTable::maxHashUsage)
                return false;
            uint32_t idx = s.hash() & (hashTabSize - 1);
            bool taken = (hashTab[idx].icl < FSST_ICL_FREE);
            if (taken)
                return false; // collision in hash table
            hashTab[idx].icl = s.icl;
            hashTab[idx].val.num =
                s.val.num & (0xFFFFFFFFFFFFFFFF >> (uint8_t)s.icl); // TIM NOTE: (uint8_t) s.icl = ignoredBits
            maxHash += 1;
            return true;
        }

        bool add(Symbol s, bool final = false) {
            assert(FSST_CODE_BASE + nSymbols < FSST_CODE_MAX);
            uint32_t len = s.length();
            s.set_code_len(FSST_CODE_BASE + nSymbols, len);
            assert(s.code() != Symbol::ignore);
            if (len == 1) {
                byteCodes[s.first()] = FSST_CODE_BASE + nSymbols + (1 << FSST_LEN_BITS); // len=1 (<<FSST_LEN_BITS)
            } else if (len == 2) {
                // TIM NOTE: limit amount of entries on same row (for sparse matrix format tests)
                if (final && maxRow[s.first()] >= maxSameRowTwo) {
                    return false;
                }

                // TIM NOTE: limit amount of rows used for matching table
                if (final && rows_used >= maxRowsTwo) {
                    return false;
                }

                // Increment row counter for matching table
                if (maxRow[s.first()] == 0) {
                    rows_used += 1;
                }

                maxRow[s.first()] += 1;

                shortCodes[s.first2()] = FSST_CODE_BASE + nSymbols + (2 << FSST_LEN_BITS); // len=2 (<<FSST_LEN_BITS)
            } else if (!hashInsert(s)) {
                return false;
            }
            symbols[FSST_CODE_BASE + nSymbols++] = s;
            lenHisto[len - 1]++;
            return true;
        }

        /// Find longest expansion, return code (= position in symbol table)
        uint16_t findLongestSymbol(Symbol s) const {
            size_t idx = s.hash() & (hashTabSize - 1);

            // TIM NOTE: made this unreadable stuff more like pseudocode in paper
            Symbol hs = hashTab[idx];
            uint8_t ignoredBits = (uint8_t)hs.icl;
            uint64_t num = s.val.num & (0xFFFFFFFFFFFFFFFF >> ignoredBits);

            /*
             * TIM NOTE: for some reason there is hs.icl <= s.icl as hit condition here.
             * I assume it replaces the hs.code != 511 here
             * icl (least to most significant) = [16 ignoredbits; 12 code; 4 len; unused 00s]
             *
             * Three cases for len:
             *    Equal: => continue to code cases
             *    HS < S: hit has less bytes than S, so not equal but will still => TRUE (but will fail on strcomp)
             *    HS > S: hit has more bytes than S, so not equal => FALSE
             *
             * Three cases for code:
             *    Equal: => ignoredBits also equal because of equal len, so TRUE
             *    Context: S will always have FSST_CODE_MAX = 512
             *    HS will always have FSST_CODE_MAX or lower, so HS <= S is always true
             *
             * In other words, this is just a length check?
             * strcomp will probably catch cache miss then..
             * To confirm:
             * free icl = 1111 0001 11111111 00000000 00000000
             * num = 0
             *
             * So hit miss has length of 15, so hit miss ALWAYS has more bytes than a regular symbol.
             * So the hs.icl <= s.icl checks for hit miss with length, then confirms equality with num strcomp
             */

            if (hs.icl <= s.icl && hs.val.num == num) {
                uint16_t codeAndLen = hs.icl >> 16; // TIM NOTE: drop ignoredBits
                return codeAndLen & FSST_CODE_MASK; // matched a long symbol // TIM NOTE: drop length, only get code
            }

            if (s.length() >= 2) {
                uint16_t code = shortCodes[s.first2()] & FSST_CODE_MASK;
                if (code >= FSST_CODE_BASE)
                    return code;
            }
            return byteCodes[s.first()] & FSST_CODE_MASK;
        }

        uint16_t findLongestSymbol(const uint8_t* cur, const uint8_t* end) const {
            return findLongestSymbol(Symbol(cur, end)); // represent the string as a temporary symbol
        }

        // rationale for finalize:
        // - during symbol table construction, we may create more than 256 codes, but bring it down to max 255 in the
        // last makeTable()
        //   consequently we needed more than 8 bits during symbol table contruction, but can simplify the codes to
        //   single bytes in finalize() (this feature is in fact lo longer used, but could still be exploited: symbol
        //   construction creates no more than 255 symbols in each pass)
        // - we not only reduce the amount of codes to <255, but also *reorder* the symbols and renumber their codes,
        // for higher compression perf.
        //   we renumber codes so they are grouped by length, to allow optimized scalar string compression (byteLim and
        //   suffixLim optimizations).
        // - we make the use of byteCode[] no longer necessary by inserting single-byte codes in the free spots of
        // shortCodes[]
        //   Using shortCodes[] only makes compression faster. When creating the symbolTable, however, using
        //   shortCodes[] for the single-byte symbols is slow, as each insert touches 256 positions in it. This
        //   optimization was added when optimizing symbolTable construction time.
        //
        // In all, we change the layout and coding, as follows..
        //
        // before finalize():
        // - The real symbols are symbols[256..256+nSymbols>. As we may have nSymbols > 255
        // - The first 256 codes are pseudo symbols (all escaped bytes)
        //
        // after finalize():
        // - table layout is symbols[0..nSymbols>, with nSymbols < 256.
        // - Real codes are [0,nSymbols>. 8-th bit not set.
        // - Escapes in shortCodes have the 8th bit set (value: 256+255=511). 255 because the code to be emitted is the
        // escape byte 255
        // - symbols are grouped by length: 2,3,4,5,6,7,8, then 1 (single-byte codes last)
        // the two-byte codes are split in two sections:
        // - first section contains codes for symbols for which there is no longer symbol (no suffix). It allows an
        // early-out during compression
        //
        // finally, shortCodes[] is modified to also encode all single-byte symbols (hence byteCodes[] is not required
        // on a critical path anymore).
        //
        void finalize(uint8_t zeroTerminated) {
            assert(nSymbols <= 255);
            uint8_t newCode[256], rsum[8], byteLim = nSymbols - (lenHisto[0] - zeroTerminated);

            // compute running sum of code lengths (starting offsets for each length)
            rsum[0] = byteLim; // 1-byte codes are highest
            rsum[1] = zeroTerminated;
            for (uint32_t i = 1; i < 7; i++)
                rsum[i + 1] = rsum[i] + lenHisto[i];

            // determine the new code for each symbol, ordered by length (and splitting 2byte symbols into two classes
            // around suffixLim)
            suffixLim = rsum[1];
            symbols[newCode[0] = 0] = symbols[256]; // keep symbol 0 in place (for zeroTerminated cases only)

            for (uint32_t i = zeroTerminated, j = rsum[2]; i < nSymbols; i++) {
                Symbol s1 = symbols[FSST_CODE_BASE + i];
                uint32_t len = s1.length(), opt = (len == 2) * nSymbols;
                if (opt) {
                    uint16_t first2 = s1.first2();
                    for (uint32_t k = 0; k < opt; k++) {
                        Symbol s2 = symbols[FSST_CODE_BASE + k];
                        if (k != i && s2.length() > 1 && first2 == s2.first2()) // test if symbol k is a suffix of s
                            opt = 0;
                    }
                    newCode[i] = opt ? suffixLim++ : --j; // symbols without a larger suffix have a code < suffixLim
                } else
                    newCode[i] = rsum[len - 1]++;
                s1.set_code_len(newCode[i], len);
                symbols[newCode[i]] = s1;
            }

            // TIM NOTE: after this point symbols[0, ..] is just the symbols in their order (why?)

            // renumber the codes in byteCodes[]
            for (uint32_t i = 0; i < 256; i++)
                if ((byteCodes[i] & FSST_CODE_MASK) >= FSST_CODE_BASE) // TIM NOTE: not escaped
                    byteCodes[i] = newCode[(uint8_t)byteCodes[i]] + (1 << FSST_LEN_BITS);
                    // TIM NOTE: encode newCode together with length?
                else
                    byteCodes[i] = 511 + (1 << FSST_LEN_BITS); // TIM NOTE:

            // renumber the codes in shortCodes[]
            for (uint32_t i = 0; i < 65536; i++)
                if ((shortCodes[i] & FSST_CODE_MASK) >= FSST_CODE_BASE) // not escaped
                    shortCodes[i] = newCode[(uint8_t)shortCodes[i]] + (shortCodes[i] & (15 << FSST_LEN_BITS));
                else
                    shortCodes[i] = byteCodes[i & 0xFF]; // TIM NOTE: Take value of single char too

            /* TIM NOTE: example access (for short codes 4 = 't')
             *    shortCodes[symbols[4].val.num & 0xFFFF] & 0b11111111 == 4 (code)
             *    shortCodes[symbols[4].val.num & 0xFFFF] >> 12 & 0b1111 == 1 (len)
             *
             *    So basically shortCodes[a][b] = shortCodes[a | b << 8] = shortCodes[a + b * 256]
             */

            // replace the symbols in the hash table
            for (uint32_t i = 0; i < hashTabSize; i++)
                if (hashTab[i].icl < FSST_ICL_FREE)
                    hashTab[i] = symbols[newCode[(uint8_t)hashTab[i].code()]];
        }

        void finalize_simple_decreasing() {
            assert(nSymbols <= SymbolTable::maxSize + 1);
            uint8_t newCode[256], rsum[8];

            rsum[7] = 0;
            for (int i = 7; i > 0; i--) {
                rsum[i - 1] = rsum[i] + lenHisto[i];
            }

            for (uint32_t i = 0; i < nSymbols; i++) {
                Symbol s1 = symbols[FSST_CODE_BASE + i];
                uint32_t len = s1.length();
                newCode[i] = rsum[len - 1]++;
                s1.set_code_len(newCode[i], len);
                symbols[newCode[i]] = s1;
            }

            // renumber the codes in byteCodes[]
            for (uint32_t i = 0; i < 256; i++)
                if ((byteCodes[i] & FSST_CODE_MASK) >= FSST_CODE_BASE) // TIM NOTE: not escaped
                    byteCodes[i] = newCode[(uint8_t)byteCodes[i]] + (1 << FSST_LEN_BITS);
                    // TIM NOTE: encode newCode together with length?
                else
                    byteCodes[i] = 511 + (1 << FSST_LEN_BITS); // TIM NOTE:

            // renumber the codes in shortCodes[]
            for (uint32_t i = 0; i < 65536; i++)
                if ((shortCodes[i] & FSST_CODE_MASK) >= FSST_CODE_BASE) // not escaped
                    shortCodes[i] = newCode[(uint8_t)shortCodes[i]] + (shortCodes[i] & (15 << FSST_LEN_BITS));
                else
                    shortCodes[i] = byteCodes[i & 0xFF]; // TIM NOTE: Take value of single char too

            // replace the symbols in the hash table
            for (uint32_t i = 0; i < hashTabSize; i++)
                if (hashTab[i].icl < FSST_ICL_FREE)
                    hashTab[i] = symbols[newCode[(uint8_t)hashTab[i].code()]];
        }
    };

#ifdef NONOPT_FSST
    struct Counters {
        uint16_t count1[FSST_CODE_MAX]; // array to count frequency of symbols as they occur in the sample
        uint16_t count2[FSST_CODE_MAX]
                       [FSST_CODE_MAX]; // array to count subsequent combinations of two symbols in the sample

        void count1Set(uint32_t pos1, uint16_t val) {
            count1[pos1] = val;
        }
        void count1Inc(uint32_t pos1) {
            count1[pos1]++;
        }
        void count2Inc(uint32_t pos1, uint32_t pos2) {
            count2[pos1][pos2]++;
        }
        uint32_t count1GetNext(uint32_t& pos1) {
            return count1[pos1];
        }
        uint32_t count2GetNext(uint32_t pos1, uint32_t& pos2) {
            return count2[pos1][pos2];
        }
        void backup1(uint8_t* buf) {
            memcpy(buf, count1, FSST_CODE_MAX * sizeof(uint16_t));
        }
        void restore1(uint8_t* buf) {
            memcpy(count1, buf, FSST_CODE_MAX * sizeof(uint16_t));
        }
    };
#else
    // we keep two counters count1[pos] and count2[pos1][pos2] of resp 16 and 12-bits. Both are split into two columns
    // for performance reasons first reason is to make the column we update the most during symbolTable construction
    // (the low bits) thinner, thus reducing CPU cache pressure. second reason is that when scanning the array, after
    // seeing a 64-bits 0 in the high bits column, we can quickly skip over many codes (15 or 7)
    struct Counters {
        // high arrays come before low arrays, because our GetNext() methods may overrun their 64-bits reads a few bytes
        uint8_t count1High[FSST_CODE_MAX]; // array to count frequency of symbols as they occur in the sample (16-bits)
        uint8_t count1Low[FSST_CODE_MAX]; // it is split in a low and high byte: cnt = count1High*256 + count1Low
        uint8_t count2High[FSST_CODE_MAX][FSST_CODE_MAX / 2];
        // array to count subsequent combinations of two symbols in the sample (12-bits: 8-bits low, 4-bits high)
        uint8_t count2Low[FSST_CODE_MAX][FSST_CODE_MAX];
        // its value is (count2High*256+count2Low) -- but high is 4-bits (we put two numbers in one, hence /2)
        // 385KB  -- but hot area likely just 10 + 30*4 = 130 cache lines (=8KB)

        void count1Set(uint32_t pos1, uint16_t val) {
            count1Low[pos1] = val & 255;
            count1High[pos1] = val >> 8;
        }

        void count1Inc(uint32_t pos1) {
            if (!count1Low[pos1]++)
                // increment high early (when low==0, not when low==255). This means (high > 0) <=> (cnt > 0)
                count1High
                    [pos1]++; //(0,0)->(1,1)->..->(255,1)->(0,1)->(1,2)->(2,2)->(3,2)..(255,2)->(0,2)->(1,3)->(2,3)...
        }

        void count2Inc(uint32_t pos1, uint32_t pos2) {
            if (!count2Low[pos1][pos2]++)
                // increment high early (when low==0, not when low==255). This means (high > 0) <=> (cnt > 0)
                // inc 4-bits high counter with 1<<0 (1) or 1<<4 (16) -- depending on whether pos2 is even or odd,
                // repectively
                count2High[pos1][(pos2) >> 1] += 1 << (((pos2) & 1) << 2);
            // we take our chances with overflow.. (4K maxval, on a 8K sample)
        }

        uint32_t count1GetNext(uint32_t& pos1) {
            // note: we will advance pos1 to the next nonzero counter in register range
            // read 16-bits single symbol counter, split into two 8-bits numbers (count1Low, count1High), while skipping
            // over zeros
            uint64_t high =
                fsst_unaligned_load(&count1High[pos1]); // note: this reads 8 subsequent counters [pos1..pos1+7]

            uint32_t zero = high ? (__builtin_ctzl(high) >> 3) : 7UL; // number of zero bytes
            high = (high >> (zero << 3)) & 255; // advance to nonzero counter
            if (((pos1 += zero) >= FSST_CODE_MAX) || !high) // SKIP! advance pos2
                return 0; // all zero

            uint32_t low = count1Low[pos1];
            if (low)
                high--; // high is incremented early and low late, so decrement high (unless low==0)
            return (uint32_t)((high << 8) + low);
        }

        uint32_t count2GetNext(uint32_t pos1, uint32_t& pos2) {
            // note: we will advance pos2 to the next nonzero counter in register range
            // read 12-bits pairwise symbol counter, split into low 8-bits and high 4-bits number while skipping over
            // zeros
            uint64_t high = fsst_unaligned_load(&count2High[pos1][pos2 >> 1]);
            // note: this reads 16 subsequent counters [pos2..pos2+15]
            high >>= ((pos2 & 1) << 2); // odd pos2: ignore the lowest 4 bits & we see only 15 counters

            uint32_t zero =
                high ? (__builtin_ctzl(high) >> 2) : (15UL - (pos2 & 1UL)); // number of zero 4-bits counters
            high = (high >> (zero << 2)) & 15; // advance to nonzero counter
            if (((pos2 += zero) >= FSST_CODE_MAX) || !high) // SKIP! advance pos2
                return 0UL; // all zero

            uint32_t low = count2Low[pos1][pos2];
            if (low)
                high--; // high is incremented early and low late, so decrement high (unless low==0)
            return (uint32_t)((high << 8) + low);
        }

        void backup1(uint8_t* buf) {
            memcpy(buf, count1High, FSST_CODE_MAX);
            memcpy(buf + FSST_CODE_MAX, count1Low, FSST_CODE_MAX);
        }

        void restore1(uint8_t* buf) {
            memcpy(count1High, buf, FSST_CODE_MAX);
            memcpy(count1Low, buf + FSST_CODE_MAX, FSST_CODE_MAX);
        }
    };
#endif


#define FSST_BUFSZ (3 << 19) // 768KB

    // an encoder is a symbolmap plus some bufferspace, needed during map construction as well as compression
    struct Encoder {
        std::shared_ptr<SymbolTable> symbolTable;

        // symbols, plus metadata and data structures for quick compression (shortCode,hashTab, etc)
        union {
            Counters counters; // for counting symbol occurences during map construction
        };
    };
} // namespace gtsst::fsst

#endif // FSST_LIB_HPP
