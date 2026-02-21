use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_uchar, c_uint, c_ushort};
use std::ptr;
use std::slice;
use std::sync::{Arc, Mutex};
use crate::ConversionError;

use crate::{
    ImageConverter,
    Format,
    ImageColorType,
    JpegSettings,
    JpegSubsampling,
    JpegOptimization,
    PngSettings,
    PngCompressionType,
    PngFilterType,
    ResizeSettings,
    ResizeFilter,
    WebPSettings,
    WebPPreset,
    TiffSettings,
    TiffCompression,
    IcoSettings,
    GifSettings,
    DdsSettings,
    DdsFormat,
    ParallelConfig,
};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(err: impl Into<String>) {
    let s = CString::new(err.into()).unwrap_or_else(|_| CString::new("unknown").unwrap());
    LAST_ERROR.with(|c| {
        *c.borrow_mut() = Some(s);
    })
}

#[no_mangle]
pub extern "C" fn ffi_get_last_error() -> *const c_char {
    LAST_ERROR.with(|c| match &*c.borrow() {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    })
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum FfiErrorCode {
    Success = 0,
    InvalidParameter = 1,
    ImageError = 2,
    IoError = 3,
    EncodingError = 4,
    Unsupported = 5,
    DdsError = 6,
    JpegError = 7,
    Unknown = 255,
}

impl From<&crate::ConversionError> for FfiErrorCode {
    fn from(e: &crate::ConversionError) -> Self {
        match e {
            ConversionError::InvalidFormat => FfiErrorCode::InvalidParameter,
            ConversionError::InvalidParameter(_) => FfiErrorCode::InvalidParameter,
            ConversionError::ImageError(_) => FfiErrorCode::ImageError,
            ConversionError::IoError(_) => FfiErrorCode::IoError,
            ConversionError::EncodingError(_) => FfiErrorCode::EncodingError,
            ConversionError::UnsupportedOperation(_) => FfiErrorCode::Unsupported,
            ConversionError::DdsError(_) => FfiErrorCode::DdsError,
            ConversionError::JpegError(_) => FfiErrorCode::JpegError,
            _ => FfiErrorCode::Unknown,
        }
    }
}

#[repr(C)]
pub struct FfiImageConverter {
    inner: Arc<Mutex<ImageConverter>>,
}

unsafe impl Send for FfiImageConverter {}
unsafe impl Sync for FfiImageConverter {}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum FfiFormat {
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
}

impl From<FfiFormat> for Format {
    fn from(f: FfiFormat) -> Format {
        match f {
            FfiFormat::Png => Format::Png,
            FfiFormat::Jpeg => Format::Jpeg,
            FfiFormat::Gif => Format::Gif,
            FfiFormat::Bmp => Format::Bmp,
            FfiFormat::Ico => Format::Ico,
            FfiFormat::Tiff => Format::Tiff,
            FfiFormat::WebP => Format::WebP,
            FfiFormat::Pnm => Format::Pnm,
            FfiFormat::Tga => Format::Tga,
            FfiFormat::Dds => Format::Dds,
            FfiFormat::Hdr => Format::Hdr,
            FfiFormat::Farbfeld => Format::Farbfeld,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiJpegSettings {
    pub quality: c_uchar,
    pub progressive: c_uchar,
    pub subsampling: c_uchar,
    pub optimization: c_uchar,
    pub arithmetic_coding: c_uchar,
    pub smoothing: c_uchar,
    pub trellis_optimization: c_uchar,
    pub progressive_scans: c_uchar,
}

impl From<FfiJpegSettings> for JpegSettings {
    fn from(s: FfiJpegSettings) -> Self {
        let subs = match s.subsampling {
            0 => JpegSubsampling::None,
            1 => JpegSubsampling::Ratio422,
            2 => JpegSubsampling::Ratio420,
            3 => JpegSubsampling::Ratio440,
            _ => JpegSubsampling::Ratio420,
        };
        let opt = match s.optimization {
            0 => JpegOptimization::Speed,
            1 => JpegOptimization::Balanced,
            2 => JpegOptimization::Size,
            3 => JpegOptimization::Quality,
            _ => JpegOptimization::Balanced,
        };

        JpegSettings {
            quality: s.quality,
            progressive: s.progressive != 0,
            subsampling: subs,
            optimization: opt,
            arithmetic_coding: s.arithmetic_coding != 0,
            smoothing: s.smoothing,
            trellis_optimization: s.trellis_optimization != 0,
            progressive_scans: s.progressive_scans,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiPngSettings {
    pub compression_level: c_uchar,
    pub compression_type: c_uchar,
    pub filter: c_uchar,
    pub interlaced: c_uchar,
    pub strip_metadata: c_uchar,
    pub bit_depth: c_uchar,
}

impl From<FfiPngSettings> for PngSettings {
    fn from(s: FfiPngSettings) -> Self {
        let ctype = match s.compression_type {
            0 => PngCompressionType::Fast,
            1 => PngCompressionType::Default,
            2 => PngCompressionType::Best,
            3 => PngCompressionType::Huffman,
            _ => PngCompressionType::Default,
        };

        let filter = match s.filter {
            0 => PngFilterType::NoFilter,
            1 => PngFilterType::Sub,
            2 => PngFilterType::Up,
            3 => PngFilterType::Avg,
            4 => PngFilterType::Paeth,
            5 => PngFilterType::Adaptive,
            _ => PngFilterType::Sub,
        };

        PngSettings {
            compression_level: s.compression_level,
            compression_type: ctype,
            filter,
            interlaced: s.interlaced != 0,
            strip_metadata: s.strip_metadata != 0,
            bit_depth: s.bit_depth,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiResizeSettings {
    pub width: c_uint,
    pub height: c_uint,
    pub keep_aspect_ratio: c_uchar,
    pub filter: c_uchar,
    pub only_shrink: c_uchar,
}

impl From<FfiResizeSettings> for ResizeSettings {
    fn from(s: FfiResizeSettings) -> Self {
        let filter = match s.filter {
            0 => ResizeFilter::Nearest,
            1 => ResizeFilter::Triangle,
            2 => ResizeFilter::CatmullRom,
            3 => ResizeFilter::Gaussian,
            4 => ResizeFilter::Lanczos3,
            _ => ResizeFilter::Lanczos3,
        };

        ResizeSettings {
            width: s.width,
            height: s.height,
            keep_aspect_ratio: s.keep_aspect_ratio != 0,
            filter,
            only_shrink: s.only_shrink != 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiParallelConfig {
    pub tile_size: c_uint,
    pub enable_tiling: c_uchar,
    pub min_size_for_parallel: c_uint,
    pub thread_count: usize,
}

impl From<FfiParallelConfig> for ParallelConfig {
    fn from(c: FfiParallelConfig) -> Self {
        ParallelConfig {
            tile_size: c.tile_size,
            enable_tiling: c.enable_tiling != 0,
            min_size_for_parallel: c.min_size_for_parallel,
            thread_count: c.thread_count,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiImageInfo {
    pub width: c_uint,
    pub height: c_uint,
    pub color_type: c_uint,
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_new(format: FfiFormat) -> *mut FfiImageConverter {
    let rust_format: Format = format.into();
    let conv = ImageConverter::new(rust_format);
    
    Box::into_raw(Box::new(FfiImageConverter {
        inner: Arc::new(Mutex::new(conv)),
    }))
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_free(handle: *mut FfiImageConverter) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_set_parallel_config(
    handle: *mut FfiImageConverter,
    config: FfiParallelConfig,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.set_parallel_config(config.into());
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_set_jpeg_settings(
    handle: *mut FfiImageConverter,
    settings: FfiJpegSettings,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.compression.jpeg = settings.into();
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_set_png_settings(
    handle: *mut FfiImageConverter,
    settings: FfiPngSettings,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.compression.png = settings.into();
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_set_resize(
    handle: *mut FfiImageConverter,
    settings: FfiResizeSettings,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.resize = Some(settings.into());
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_strip_metadata(
    handle: *mut FfiImageConverter,
    strip: c_uchar,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.set_strip_metadata(strip != 0);
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_enable_optimization(
    handle: *mut FfiImageConverter,
    enable: c_uchar,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.set_enable_optimization(enable != 0);
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_convert_memory(
    handle: *mut FfiImageConverter,
    input_ptr: *const u8,
    input_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> FfiErrorCode {
    if handle.is_null() || input_ptr.is_null() || input_len == 0 || out_ptr.is_null() || out_len.is_null() {
        set_last_error("Invalid parameters");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        let input_slice = slice::from_raw_parts(input_ptr, input_len);

        let conv = match (*handle).inner.lock() {
            Ok(c) => c,
            Err(_) => {
                set_last_error("Failed to lock converter");
                return FfiErrorCode::Unknown;
            }
        };

        match conv.convert(input_slice) {
            Ok(mut vec) => {
                vec.shrink_to_fit();
                let size = vec.len();

                if size == 0 {
                    *out_ptr = ptr::null_mut();
                    *out_len = 0;
                    return FfiErrorCode::Success;
                }

                let mem = vec.as_mut_ptr();
                std::mem::forget(vec);

                *out_ptr = mem;
                *out_len = size;

                FfiErrorCode::Success
            }
            Err(e) => {
                set_last_error(format!("{:?}", e));
                FfiErrorCode::from(&e)
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_convert_batch(
    handle: *mut FfiImageConverter,
    input_ptrs: *const *const u8,
    input_lens: *const usize,
    count: usize,
    out_ptrs: *mut *mut u8,
    out_lens: *mut usize,
    out_errors: *mut FfiErrorCode,
) -> FfiErrorCode {
    if handle.is_null() || input_ptrs.is_null() || input_lens.is_null() 
        || count == 0 || out_ptrs.is_null() || out_lens.is_null() {
        set_last_error("Invalid parameters");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        let input_ptr_slice = slice::from_raw_parts(input_ptrs, count);
        let input_len_slice = slice::from_raw_parts(input_lens, count);
        
        let inputs: Vec<&[u8]> = input_ptr_slice.iter()
            .zip(input_len_slice.iter())
            .map(|(&ptr, &len)| slice::from_raw_parts(ptr, len))
            .collect();

        let conv = match (*handle).inner.lock() {
            Ok(c) => c,
            Err(_) => {
                set_last_error("Failed to lock converter");
                return FfiErrorCode::Unknown;
            }
        };

        let results = conv.convert_batch(&inputs);

        let out_ptr_slice = slice::from_raw_parts_mut(out_ptrs, count);
        let out_len_slice = slice::from_raw_parts_mut(out_lens, count);
        let mut out_err_slice = if !out_errors.is_null() {
            Some(slice::from_raw_parts_mut(out_errors, count))
        } else {
            None
        };

        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(mut vec) => {
                    vec.shrink_to_fit();
                    let size = vec.len();
                    
                    if size == 0 {
                        out_ptr_slice[i] = ptr::null_mut();
                        out_len_slice[i] = 0;
                    } else {
                        let mem = vec.as_mut_ptr();
                        std::mem::forget(vec);
                        out_ptr_slice[i] = mem;
                        out_len_slice[i] = size;
                    }
                    
                    if let Some(ref mut errs) = out_err_slice {
                        errs[i] = FfiErrorCode::Success;
                    }
                }
                Err(e) => {
                    out_ptr_slice[i] = ptr::null_mut();
                    out_len_slice[i] = 0;
                    
                    if let Some(ref mut errs) = out_err_slice {
                        errs[i] = FfiErrorCode::from(&e);
                    }
                }
            }
        }

        FfiErrorCode::Success
    }
}

#[no_mangle]
pub extern "C" fn ric_free_buffer(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        unsafe {
            Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_get_image_info(
    input_ptr: *const u8,
    input_len: usize,
    out_info: *mut FfiImageInfo,
) -> FfiErrorCode {
    if input_ptr.is_null() || input_len == 0 || out_info.is_null() {
        set_last_error("Invalid parameters");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        let input_slice = slice::from_raw_parts(input_ptr, input_len);
        match ImageConverter::get_image_info(input_slice) {
            Ok(info) => {
                let color = match info.color_type {
                    ImageColorType::Grayscale => 0u32,
                    ImageColorType::GrayscaleAlpha => 1u32,
                    ImageColorType::Rgb => 2u32,
                    ImageColorType::Rgba => 3u32,
                    ImageColorType::Grayscale16 => 4u32,
                    ImageColorType::GrayscaleAlpha16 => 5u32,
                    ImageColorType::Rgb16 => 6u32,
                    ImageColorType::Rgba16 => 7u32,
                    ImageColorType::Rgb32F => 8u32,
                    ImageColorType::Rgba32F => 9u32,
                    _ => 0xFFFFFFFFu32,
                };

                (*out_info).width = info.width;
                (*out_info).height = info.height;
                (*out_info).color_type = color;
                FfiErrorCode::Success
            }
            Err(e) => {
                set_last_error(format!("{:?}", e));
                FfiErrorCode::from(&e)
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiWebPSettings {
    pub quality: f32,
    pub lossless: c_uchar,
    pub method: c_uchar,
    pub preset: c_uchar,
    pub threading: c_uchar,
    pub target_size: usize,
}

impl From<FfiWebPSettings> for WebPSettings {
    fn from(s: FfiWebPSettings) -> Self {
        let preset = match s.preset {
            0 => WebPPreset::Default,
            1 => WebPPreset::Picture,
            2 => WebPPreset::Photo,
            3 => WebPPreset::Drawing,
            4 => WebPPreset::Icon,
            5 => WebPPreset::Text,
            _ => WebPPreset::Default,
        };

        WebPSettings {
            quality: s.quality,
            lossless: s.lossless != 0,
            method: s.method,
            preset,
            threading: s.threading != 0,
            target_size: s.target_size,
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_set_webp_settings(
    handle: *mut FfiImageConverter,
    settings: FfiWebPSettings,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.compression.webp = settings.into();
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiTiffSettings {
    pub compression: c_uchar,
    pub jpeg_quality: c_uchar,
    pub predictor: c_uchar,
}

impl From<FfiTiffSettings> for TiffSettings {
    fn from(s: FfiTiffSettings) -> Self {
        let compression = match s.compression {
            0 => TiffCompression::None,
            1 => TiffCompression::Lzw,
            2 => TiffCompression::Deflate,
            3 => TiffCompression::PackBits,
            4 => TiffCompression::Jpeg,
            _ => TiffCompression::Lzw,
        };

        TiffSettings {
            compression,
            jpeg_quality: s.jpeg_quality,
            predictor: s.predictor != 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_set_tiff_settings(
    handle: *mut FfiImageConverter,
    settings: FfiTiffSettings,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.compression.tiff = settings.into();
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiIcoSettings {
    pub max_dimension: c_uint,
    pub generate_multiple_sizes: c_uchar,
    pub sizes: [c_uint; 8],
    pub num_sizes: usize,
}

impl From<FfiIcoSettings> for IcoSettings {
    fn from(s: FfiIcoSettings) -> Self {
        IcoSettings {
            max_dimension: s.max_dimension,
            generate_multiple_sizes: s.generate_multiple_sizes != 0,
            sizes: s.sizes,
            num_sizes: s.num_sizes,
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_set_ico_settings(
    handle: *mut FfiImageConverter,
    settings: FfiIcoSettings,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.compression.ico = settings.into();
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiDdsSettings {
    pub format: c_uchar,
    pub generate_mipmaps: c_uchar,
    pub mipmap_count: c_uint,
    pub is_cubemap: c_uchar,
}

impl From<FfiDdsSettings> for DdsSettings {
    fn from(s: FfiDdsSettings) -> Self {
        let format = match s.format {
            0 => DdsFormat::RGBA8,
            1 => DdsFormat::BC1,
            2 => DdsFormat::BC2,
            3 => DdsFormat::BC3,
            4 => DdsFormat::BC4,
            5 => DdsFormat::BC5,
            6 => DdsFormat::BC6H,
            7 => DdsFormat::BC7,
            _ => DdsFormat::RGBA8,
        };

        DdsSettings {
            format,
            generate_mipmaps: s.generate_mipmaps != 0,
            mipmap_count: s.mipmap_count,
            is_cubemap: s.is_cubemap != 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_set_dds_settings(
    handle: *mut FfiImageConverter,
    settings: FfiDdsSettings,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.compression.dds = settings.into();
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiGifSettings {
    pub palette_size: c_ushort,
    pub dithering: c_uchar,
    pub quality: c_uchar,
}

impl From<FfiGifSettings> for GifSettings {
    fn from(s: FfiGifSettings) -> Self {
        GifSettings {
            palette_size: s.palette_size,
            dithering: s.dithering != 0,
            quality: s.quality,
        }
    }
}

#[no_mangle]
pub extern "C" fn ffi_image_converter_set_gif_settings(
    handle: *mut FfiImageConverter,
    settings: FfiGifSettings,
) -> FfiErrorCode {
    if handle.is_null() {
        set_last_error("Null converter handle");
        return FfiErrorCode::InvalidParameter;
    }

    unsafe {
        if let Ok(mut conv) = (*handle).inner.lock() {
            conv.compression.gif = settings.into();
            FfiErrorCode::Success
        } else {
            set_last_error("Failed to lock converter");
            FfiErrorCode::Unknown
        }
    }
}