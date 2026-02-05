#ifndef FSST_H
#define FSST_H

#ifdef _MSC_VER
#define __restrict__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#define __ORDER_LITTLE_ENDIAN__ 2
#include <intrin.h>
static inline int __builtin_ctzl(unsigned long long x) {
    unsigned long ret;
    _BitScanForward64(&ret, x);
    return (int)ret;
}
#endif

#ifdef __cplusplus
#define FSST_FALLTHROUGH [[fallthrough]]
#include <cstring>
extern "C" {
#else
#define FSST_FALLTHROUGH
#endif

#include <stddef.h>

/* A compressed string is simply a string of 1-byte codes; except for code 255, which is followed by an uncompressed byte. */
#define FSST_ESC 255
#define FSST_CORRUPT 32774747032022883 /* 7-byte number in little endian containing "corrupt" */

/* Data structure needed for compressing strings - use fsst_duplicate() to create thread-local copies. Use fsst_destroy() to free. */
typedef void* fsst_encoder_t; /* opaque type - it wraps around a rather large (~900KB) C++ object */

/* Data structure needed for decompressing strings - read-only and thus can be shared between multiple decompressing threads. */
typedef struct {
   unsigned long long version;     /* version id */
   unsigned char zeroTerminated;   /* terminator is a single-byte code that does not appear in longer symbols */
   unsigned char len[255];         /* len[x] is the byte-length of the symbol x (1 < len[x] <= 8). */
   unsigned long long symbol[255]; /* symbol[x] contains in LITTLE_ENDIAN the bytesequence that code x represents (0 <= x < 255). */
} fsst_decoder_t;

/* Calibrate a FSST symboltable from a batch of strings (it is best to provide at least 16KB of data). */
fsst_encoder_t*
fsst_create(
   size_t n,         /* IN: number of strings in batch to sample from. */
   const size_t lenIn[],   /* IN: byte-lengths of the inputs */
   const unsigned char *strIn[],  /* IN: string start pointers. */
   int zeroTerminated       /* IN: whether input strings are zero-terminated. If so, encoded strings are as well (i.e. symbol[0]=""). */
);

/* Create another encoder instance, necessary to do multi-threaded encoding using the same symbol table. */
fsst_encoder_t*
fsst_duplicate(
   fsst_encoder_t *encoder  /* IN: the symbol table to duplicate. */
);

#define FSST_MAXHEADER (8+1+8+2048+1) /* maxlen of deserialized fsst header, produced/consumed by fsst_export() resp. fsst_import() */

/* Space-efficient symbol table serialization (smaller than sizeof(fsst_decoder_t) - by saving on the unused bytes in symbols of len < 8). */
unsigned int                /* OUT: number of bytes written in buf, at most sizeof(fsst_decoder_t) */
fsst_export(
   fsst_encoder_t *encoder, /* IN: the symbol table to dump. */
   unsigned char *buf       /* OUT: pointer to a byte-buffer where to serialize this symbol table. */
);

/* Deallocate encoder. */
void
fsst_destroy(fsst_encoder_t*);

/* Return a decoder structure from serialized format (typically used in a block-, file- or row-group header). */
unsigned int                /* OUT: number of bytes consumed in buf (0 on failure). */
fsst_import(
   fsst_decoder_t *decoder, /* IN: this symbol table will be overwritten. */
   const unsigned char *buf       /* OUT: pointer to a byte-buffer where fsst_export() serialized this symbol table. */
);

/* Return a decoder structure from an encoder. */
fsst_decoder_t
fsst_decoder(
   fsst_encoder_t *encoder
);

/* Compress a batch of strings (on AVX512 machines best performance is obtained by compressing more than 32KB of string volume). */
/* The output buffer must be large; at least "conservative space" (7+2*inputlength) for the first string for something to happen. */
size_t                      /* OUT: the number of compressed strings (<=n) that fit the output buffer. */
fsst_compress(
   fsst_encoder_t *encoder, /* IN: encoder obtained from fsst_create(). */
   size_t nstrings,         /* IN: number of strings in batch to compress. */
   const size_t lenIn[],          /* IN: byte-lengths of the inputs */
   const unsigned char *strIn[],  /* IN: input string start pointers. */
   size_t outsize,          /* IN: byte-length of output buffer. */
   unsigned char *output,   /* OUT: memory buffer to put the compressed strings in (one after the other). */
   size_t lenOut[],         /* OUT: byte-lengths of the compressed strings. */
   unsigned char *strOut[]  /* OUT: output string start pointers. Will all point into [output,output+size). */
);

/* Decompress a single string, inlined for speed. */
inline size_t /* OUT: bytesize of the decompressed string. If > size, the decoded output is truncated to size. */
fsst_decompress(
   const fsst_decoder_t *decoder,  /* IN: use this symbol table for compression. */
   size_t lenIn,             /* IN: byte-length of compressed string. */
   const unsigned char *strIn,     /* IN: compressed string. */
   size_t size,              /* IN: byte-length of output buffer. */
   unsigned char *output     /* OUT: memory buffer to put the decompressed string in. */
) {
   unsigned char* len = const_cast<unsigned char*>(decoder->len);
   unsigned char* strOut = output;
   unsigned long long* symbol = const_cast<unsigned long long*>(decoder->symbol);
   size_t code, posOut = 0, posIn = 0;
#ifndef FSST_MUST_ALIGN /* defining on platforms that require aligned memory access may help their performance */
#define FSST_UNALIGNED_STORE(dst,src) memcpy((unsigned long long*) (dst), &(src), sizeof(unsigned long long))
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
   while (posOut+32 <= size && posIn+4 <= lenIn) {
      unsigned int nextBlock, escapeMask;
      memcpy(&nextBlock, strIn+posIn, sizeof(unsigned int));
      escapeMask = (nextBlock&0x80808080u)&((((~nextBlock)&0x7F7F7F7Fu)+0x7F7F7F7Fu)^0x80808080u);
      if (escapeMask == 0) {
         code = strIn[posIn++]; FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); posOut += len[code];
         code = strIn[posIn++]; FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); posOut += len[code];
         code = strIn[posIn++]; FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); posOut += len[code];
         code = strIn[posIn++]; FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); posOut += len[code];
     } else {
         unsigned long firstEscapePos=__builtin_ctzl((unsigned long long) escapeMask)>>3;
         switch(firstEscapePos) { /* Duff's device */
         case 3: code = strIn[posIn++]; FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); posOut += len[code];
                 // fall through
         case 2: code = strIn[posIn++]; FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); posOut += len[code];
                 // fall through
         case 1: code = strIn[posIn++]; FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); posOut += len[code];
                 // fall through
         case 0: posIn+=2; strOut[posOut++] = strIn[posIn-1]; /* decompress an escaped byte */
         }
      }
   }
   if (posOut+24 <= size) { // handle the possibly 3 last bytes without a loop
      if (posIn+2 <= lenIn) {
	 strOut[posOut] = strIn[posIn+1];
         if (strIn[posIn] != FSST_ESC) {
            code = strIn[posIn++]; FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); posOut += len[code];
            if (strIn[posIn] != FSST_ESC) {
               code = strIn[posIn++]; FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); posOut += len[code];
            } else {
               posIn += 2; strOut[posOut++] = strIn[posIn-1];
            }
         } else {
            posIn += 2; posOut++;
         }
      }
      if (posIn < lenIn) { // last code cannot be an escape
         code = strIn[posIn++]; FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); posOut += len[code];
      }
   }
#else
   while (posOut+8 <= size && posIn < lenIn)
      if ((code = strIn[posIn++]) < FSST_ESC) { /* symbol compressed as code? */
         FSST_UNALIGNED_STORE(strOut+posOut, symbol[code]); /* unaligned memory write */
         posOut += len[code];
      } else {
         strOut[posOut] = strIn[posIn]; /* decompress an escaped byte */
         posIn++; posOut++;
      }
#endif
#endif
   while (posIn < lenIn)
      if ((code = strIn[posIn++]) < FSST_ESC) {
         size_t posWrite = posOut, endWrite = posOut + len[code];
         unsigned char* symbolPointer = reinterpret_cast<unsigned char*>(&symbol[code]) - posWrite;
         if ((posOut = endWrite) > size) endWrite = size;
         for(; posWrite < endWrite; posWrite++)  /* only write if there is room */
            strOut[posWrite] = symbolPointer[posWrite];
      } else {
         if (posOut < size) strOut[posOut] = strIn[posIn]; /* idem */
         posIn++; posOut++;
      }
   if (posOut >= size && (decoder->zeroTerminated&1)) strOut[size-1] = 0;
   return posOut; /* full size of decompressed string (could be >size, then the actually decompressed part) */
}

#ifdef __cplusplus
}
#endif
#endif //FSST_H
