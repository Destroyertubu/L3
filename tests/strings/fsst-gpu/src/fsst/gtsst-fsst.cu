#include <fsst/fsst-lib.cuh>
#include <fsst/gtsst-fsst.cuh>
#include <gtsst/gtsst.cuh>

namespace gtsst::fsst {
    inline bool is_escape_code(uint16_t pos) {
        return pos < FSST_CODE_BASE;
    }

    int compress_count_single(SymbolTable* st, Counters& counters, size_t sampleFrac, const uint8_t* line,
                              const size_t len) {
        int gain = 0;

        const uint8_t *cur = line, *start = cur;
        const uint8_t* end = cur + len;

        // TODO: maybe some other way to skip data partially?

        if (cur < end) {
            uint16_t code2 = 255, code1 = st->findLongestSymbol(cur, end);
            cur += st->symbols[code1].length();
            gain += (int)(st->symbols[code1].length() - (1 + is_escape_code(code1)));
            while (true) {
                // count single symbol (i.e. an option is not extending it)
                counters.count1Inc(code1);

                // as an alternative, consider just using the next byte..
                if (st->symbols[code1].length() != 1) // .. but do not count single byte symbols doubly
                    counters.count1Inc(*start);

                if (cur == end) {
                    break;
                }

                // now match a new symbol
                start = cur;
                if (cur < end - 7) {
                    uint64_t word = fsst_unaligned_load(cur);
                    size_t code = word & 0xFFFFFF;
                    size_t idx = FSST_HASH(code) & (SymbolTable::hashTabSize - 1);
                    Symbol s = st->hashTab[idx];
                    code2 = st->shortCodes[word & 0xFFFF] & FSST_CODE_MASK;
                    word &= (0xFFFFFFFFFFFFFFFF >> (uint8_t)s.icl);
                    if ((s.icl < FSST_ICL_FREE) & (s.val.num == word)) {
                        code2 = s.code();
                        cur += s.length();
                    } else if (code2 >= FSST_CODE_BASE) {
                        cur += 2;
                    } else {
                        code2 = st->byteCodes[word & 0xFF] & FSST_CODE_MASK;
                        cur += 1;
                    }
                } else {
                    code2 = st->findLongestSymbol(cur, end);
                    cur += st->symbols[code2].length();
                }

                // compute compressed output size
                gain += ((int)(cur - start)) - (1 + is_escape_code(code2));

                if (sampleFrac < 128) {
                    // no need to count pairs in final round // TIM NOTE: why?
                    // consider the symbol that is the concatenation of the two last symbols
                    counters.count2Inc(code1, code2);

                    // as an alternative, consider just extending with the next byte..
                    if ((cur - start) > 1) // ..but do not count single byte extensions doubly
                        counters.count2Inc(code1, *start);
                }
                code1 = code2;
            }
        }
        return gain;
    }

    inline Symbol concat(Symbol a, Symbol b) {
        Symbol s;
        uint32_t length = a.length() + b.length();
        if (length > Symbol::maxLength) {
            length = Symbol::maxLength;

            // TIM NOTE: ADDED FOR CLARITY
            b.set_code_len(FSST_CODE_MASK, length - a.length());
            b.val.num = b.val.num & ((1 << b.length() * 8) - 1);
        }
        s.set_code_len(FSST_CODE_MASK, length);
        s.val.num = (b.val.num << (8 * a.length())) | a.val.num;
        return s;
    }

    void make_table(SymbolTable* st, Counters& counters, size_t sampleFrac) {
        // hashmap of c (needed because we can generate duplicate candidates)
        std::unordered_set<QSymbol> cands;

        // artificially make terminater the most frequent symbol so it gets included
        uint16_t terminator = st->nSymbols ? FSST_CODE_BASE : st->terminator;
        counters.count1Set(terminator, 65535);

        auto addOrInc = [&](std::unordered_set<QSymbol>& cands, Symbol s, uint64_t count) {
            // TIM NOTE: REMOVED FOR NOW
            // if (count < (5*sampleFrac)/128) return; // improves both compression speed (less candidates), but also
            // quality!!
            QSymbol q;
            q.symbol = s;
            q.gain = count * s.length();
            auto it = cands.find(q);
            if (it != cands.end()) {
                q.gain += (*it).gain;
                cands.erase(*it);
            }
            cands.insert(q);
        };

        // add candidate symbols based on counted frequency
        for (uint32_t pos1 = 0; pos1 < FSST_CODE_BASE + (size_t)st->nSymbols; pos1++) {
            uint32_t cnt1 = counters.count1GetNext(pos1); // may advance pos1!!
            if (!cnt1)
                continue;

            // TIM NOTE: PUT IT FROM 8 to 2, seems more natural and works better
            // heuristic: promoting single-byte symbols (*8) helps reduce exception rates and increases [de]compression
            // speed
            Symbol s1 = st->symbols[pos1];
            addOrInc(cands, s1, ((s1.length() == 1) ? 2LL : 1LL) * cnt1);

            // TIM NOTE: REMOVED FOR NOW
            // if (sampleFrac >= 128 || // last round we do not create new (combined) symbols
            //     s1.length() == Symbol::maxLength || // symbol cannot be extended
            //     s1.val.str[0] == st->terminator) { // multi-byte symbols cannot contain the terminator byte
            //    continue;
            // }
            for (uint32_t pos2 = 0; pos2 < FSST_CODE_BASE + (size_t)st->nSymbols; pos2++) {
                uint32_t cnt2 = counters.count2GetNext(pos1, pos2); // may advance pos2!!
                if (!cnt2)
                    continue;

                // create a new symbol
                Symbol s2 = st->symbols[pos2];
                Symbol s3 = concat(s1, s2);
                if (s2.val.str[0] != st->terminator) // multi-byte symbols cannot contain the terminator byte
                    addOrInc(cands, s3, cnt2);
            }
        }

        // insert candidates into priority queue (by gain)
        auto cmpGn = [](const QSymbol& q1, const QSymbol& q2) {
            return (q1.gain < q2.gain) || (q1.gain == q2.gain && q1.symbol.val.num > q2.symbol.val.num);
        };
        std::priority_queue<QSymbol, std::vector<QSymbol>, decltype(cmpGn)> pq(cmpGn);
        for (auto& q : cands)
            pq.push(q);

        // Create new symbol map using best candidates
        st->clear();
        while (st->nSymbols < SymbolTable::maxSize && !pq.empty()) {
            QSymbol q = pq.top();
            pq.pop();
            st->add(q.symbol, sampleFrac == 128);
        }

        // If this is the final iteration, make sure to add a character that maps padding (0xFE, to something else)
        if (sampleFrac == 128) {
            st->add(Symbol(Symbol::ignore, 0), true);
        }
    };

    size_t simple_make_sample(uint8_t* sample_buf, const uint8_t* src, const size_t len) {
        size_t sampleLen = 0;

        if (len < FSST_SAMPLETARGET) {
            memcpy(sample_buf, src, len);
            sampleLen = len;
        } else {
            size_t sampleRnd = FSST_HASH(4637947);
            const uint8_t* sampleLim = sample_buf + FSST_SAMPLETARGET;
            size_t chunks = 1 + (len - 1) / FSST_SAMPLELINE;

            while (sample_buf < sampleLim) {
                // Choose a random chunk of data
                sampleRnd = FSST_HASH(sampleRnd);
                size_t chunk = FSST_SAMPLELINE * (sampleRnd % chunks);

                // Add it to the sample
                size_t chunkLen = std::min(len - chunk, FSST_SAMPLELINE);
                memcpy(sample_buf, src + chunk, chunkLen);
                sample_buf += chunkLen;
                sampleLen += chunkLen;
            }
        }

        return sampleLen;
    }

    size_t DecodingTable::import_table(const uint8_t* src) {
        uint64_t version = 0;
        uint32_t code, pos = 17;
        uint8_t lenHisto[8];

        // version field (first 8 bytes) is now there just for future-proofness, unused still (skipped)
        memcpy(&version, src, 8);
        if ((version >> 32) != FSST_VERSION)
            return 0;
        decoder.zeroTerminated = src[8] & 1;
        memcpy(lenHisto, src + 9, 8);

        // in case of zero-terminated, first symbol is "" (zero always, may be overwritten)
        decoder.len[0] = 1;
        decoder.symbol[0] = 0;

        // we use lenHisto[0] as 1-byte symbol run length (at the end)
        code = decoder.zeroTerminated;
        if (decoder.zeroTerminated)
            lenHisto[0]--; // if zeroTerminated, then symbol "" aka 1-byte code=0, is not stored at the end

        // now get all symbols from the buffer
        for (int l = 7; l >= 0; l--) {
            for (uint32_t i = 0; i < lenHisto[l]; i++, code++) {
                decoder.len[code] = l + 1;
                decoder.symbol[code] = 0;
                for (uint32_t j = 0; j < decoder.len[code]; j++)
                    ((uint8_t*)&decoder.symbol[code])[j] = src[pos++]; // note this enforces 'little endian' symbols
            }
        }
        if (decoder.zeroTerminated)
            lenHisto[0]++;

        // fill unused symbols with text "corrupt". Gives a chance to detect corrupted code sequences (if there are
        // unused symbols).
        while (code < 255) {
            decoder.symbol[code] = FSST_CORRUPT;
            decoder.len[code++] = 8;
        }
        return pos;
    }
} // namespace gtsst::fsst