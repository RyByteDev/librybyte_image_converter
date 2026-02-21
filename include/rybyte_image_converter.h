/* Generated with cbindgen:0.29.2 */

/* Automatically generated, do not edit */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum FfiFormat {
  Png = 0,
  Jpeg = 1,
  Gif = 2,
  Bmp = 3,
  Ico = 4,
  Tiff = 5,
  WebP = 6,
  Pnm = 7,
  Tga = 8,
  Dds = 9,
  Hdr = 10,
  Farbfeld = 11,
} FfiFormat;

typedef enum FfiErrorCode {
  Success = 0,
  InvalidParameter = 1,
  ImageError = 2,
  IoError = 3,
  EncodingError = 4,
  Unsupported = 5,
  DdsError = 6,
  JpegError = 7,
  Unknown = 255,
} FfiErrorCode;

typedef struct Arc_Mutex_ImageConverter Arc_Mutex_ImageConverter;

typedef struct FfiImageConverter {
  struct Arc_Mutex_ImageConverter;
} FfiImageConverter;

typedef struct FfiParallelConfig {
  unsigned int tile_size;
  unsigned char enable_tiling;
  unsigned int min_size_for_parallel;
  uintptr_t thread_count;
} FfiParallelConfig;

typedef struct FfiJpegSettings {
  unsigned char quality;
  unsigned char progressive;
  unsigned char subsampling;
  unsigned char optimization;
  unsigned char arithmetic_coding;
  unsigned char smoothing;
  unsigned char trellis_optimization;
  unsigned char progressive_scans;
} FfiJpegSettings;

typedef struct FfiPngSettings {
  unsigned char compression_level;
  unsigned char compression_type;
  unsigned char filter;
  unsigned char interlaced;
  unsigned char strip_metadata;
  unsigned char bit_depth;
} FfiPngSettings;

typedef struct FfiResizeSettings {
  unsigned int width;
  unsigned int height;
  unsigned char keep_aspect_ratio;
  unsigned char filter;
  unsigned char only_shrink;
} FfiResizeSettings;

typedef struct FfiImageInfo {
  unsigned int width;
  unsigned int height;
  unsigned int color_type;
} FfiImageInfo;

typedef struct FfiWebPSettings {
  float quality;
  unsigned char lossless;
  unsigned char method;
  unsigned char preset;
  unsigned char threading;
  uintptr_t target_size;
} FfiWebPSettings;

typedef struct FfiTiffSettings {
  unsigned char compression;
  unsigned char jpeg_quality;
  unsigned char predictor;
} FfiTiffSettings;

typedef struct FfiIcoSettings {
  unsigned int max_dimension;
  unsigned char generate_multiple_sizes;
  unsigned int sizes[8];
  uintptr_t num_sizes;
} FfiIcoSettings;

typedef struct FfiDdsSettings {
  unsigned char format;
  unsigned char generate_mipmaps;
  unsigned int mipmap_count;
  unsigned char is_cubemap;
} FfiDdsSettings;

typedef struct FfiGifSettings {
  unsigned short palette_size;
  unsigned char dithering;
  unsigned char quality;
} FfiGifSettings;

const char *ffi_get_last_error(void);

struct FfiImageConverter *ffi_image_converter_new(enum FfiFormat format);

void ffi_image_converter_free(struct FfiImageConverter *handle);

enum FfiErrorCode ffi_image_converter_set_parallel_config(struct FfiImageConverter *handle,
                                                          struct FfiParallelConfig config);

enum FfiErrorCode ffi_image_converter_set_jpeg_settings(struct FfiImageConverter *handle,
                                                        struct FfiJpegSettings settings);

enum FfiErrorCode ffi_image_converter_set_png_settings(struct FfiImageConverter *handle,
                                                       struct FfiPngSettings settings);

enum FfiErrorCode ffi_image_converter_set_resize(struct FfiImageConverter *handle,
                                                 struct FfiResizeSettings settings);

enum FfiErrorCode ffi_image_converter_strip_metadata(struct FfiImageConverter *handle,
                                                     unsigned char strip);

enum FfiErrorCode ffi_image_converter_enable_optimization(struct FfiImageConverter *handle,
                                                          unsigned char enable);

enum FfiErrorCode ffi_image_converter_convert_memory(struct FfiImageConverter *handle,
                                                     const uint8_t *input_ptr,
                                                     uintptr_t input_len,
                                                     uint8_t **out_ptr,
                                                     uintptr_t *out_len);

enum FfiErrorCode ffi_image_converter_convert_batch(struct FfiImageConverter *handle,
                                                    const uint8_t *const *input_ptrs,
                                                    const uintptr_t *input_lens,
                                                    uintptr_t count,
                                                    uint8_t **out_ptrs,
                                                    uintptr_t *out_lens,
                                                    enum FfiErrorCode *out_errors);

void ric_free_buffer(uint8_t *ptr, uintptr_t len);

enum FfiErrorCode ffi_image_converter_get_image_info(const uint8_t *input_ptr,
                                                     uintptr_t input_len,
                                                     struct FfiImageInfo *out_info);

enum FfiErrorCode ffi_image_converter_set_webp_settings(struct FfiImageConverter *handle,
                                                        struct FfiWebPSettings settings);

enum FfiErrorCode ffi_image_converter_set_tiff_settings(struct FfiImageConverter *handle,
                                                        struct FfiTiffSettings settings);

enum FfiErrorCode ffi_image_converter_set_ico_settings(struct FfiImageConverter *handle,
                                                       struct FfiIcoSettings settings);

enum FfiErrorCode ffi_image_converter_set_dds_settings(struct FfiImageConverter *handle,
                                                       struct FfiDdsSettings settings);

enum FfiErrorCode ffi_image_converter_set_gif_settings(struct FfiImageConverter *handle,
                                                       struct FfiGifSettings settings);
