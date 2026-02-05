#include <compressors/shared.cuh>
#include <iostream>

namespace gtsst::compressors {
    size_t seq_decompress(const fsst::DecodingTable& dec, const uint8_t* src, uint8_t* dst, const uint32_t len, bool use_skip) {
        const uint8_t* src_lim = src + len;
        int i = 0;
        // int escapes = 0;

        while (src < src_lim) {
            uint8_t cur = *src;
            src += 1;

            // if escape, write next char to ouput
            if (cur == fsst::Symbol::escape) {
                dst[i++] = *src;
                src += 1;
                // escapes += 1;
                continue;
            }

            // if skip, do nothing :)
            if (use_skip && cur == fsst::Symbol::skip) {
                continue;
            }

            uint8_t sym_len = dec.decoder.len[cur];

            // Output symbol
            for (int j = 0; j < sym_len; j++) {
                dst[i++] = dec.decoder.symbol[cur] >> j * 8 & 0xFF;
            }
        }

        // std::cout << "escapes: " << escapes << std::endl;
        return i;
    }
} // namespace gtsst::compressors
