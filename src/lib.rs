use image::{
    DynamicImage, ImageError, ImageFormat,
    codecs::png::{PngEncoder, CompressionType, FilterType as ImageFilterType},
    codecs::ico::IcoEncoder,
    codecs::ico::IcoFrame,
    codecs::bmp::BmpEncoder,
    codecs::tiff::TiffEncoder,
    ColorType, Rgb, ImageEncoder, /*GenericImageView,*/ RgbaImage, //ImageBuffer,
};
use std::io::{Cursor, Write};
use thiserror::Error;
use ddsfile::{Dds, NewDxgiParams, DxgiFormat, Caps2, AlphaMode, D3D10ResourceDimension};
use mozjpeg::{Compress, ColorSpace as MozColorSpace, ScanMode};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod ffi;

#[derive(Error, Debug)]
pub enum ConversionError {
    #[error("Image processing error: {0}")]
    ImageError(#[from] ImageError),
    
    #[error("Invalid format")]
    InvalidFormat,
    
    #[error("Encoding error: {0}")]
    EncodingError(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("DDS error: {0}")]
    DdsError(String),
    
    #[error("Farbfeld error: {0}")]
    FarbfeldError(String),
    
    #[error("JPEG error: {0}")]
    JpegError(String),
    
    #[cfg(feature = "webp-encoding")]
    #[error("WebP error: {0}")]
    WebPError(String),
}

pub type Result<T> = std::result::Result<T, ConversionError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum Format {
    Png,
    Jpeg,
    Gif,
    Bmp,
    Ico,
    Tiff,
    WebP,
    Pnm,
    Tga,
    Dds,
    Hdr,
    Farbfeld,
}

impl Format {
    fn to_image_format(self) -> ImageFormat {
        match self {
            Format::Png => ImageFormat::Png,
            Format::Jpeg => ImageFormat::Jpeg,
            Format::Gif => ImageFormat::Gif,
            Format::Bmp => ImageFormat::Bmp,
            Format::Ico => ImageFormat::Ico,
            Format::Tiff => ImageFormat::Tiff,
            Format::WebP => ImageFormat::WebP,
            Format::Pnm => ImageFormat::Pnm,
            Format::Tga => ImageFormat::Tga,
            Format::Dds => ImageFormat::Dds,
            Format::Hdr => ImageFormat::Hdr,
            Format::Farbfeld => ImageFormat::Farbfeld,
        }
    }

    #[allow(dead_code)]
    fn supports_alpha(self) -> bool {
        matches!(
            self,
            Format::Png | Format::Ico | Format::Tiff | Format::WebP | Format::Tga | Format::Dds | Format::Farbfeld
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum JpegSubsampling {
    None,
    Ratio422,
    Ratio420,
    Ratio440,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum JpegOptimization {
    Speed,
    Balanced,
    Size,
    Quality,
}

#[derive(Debug, Clone)]
pub struct JpegSettings {
    pub quality: u8,
    pub progressive: bool,
    pub subsampling: JpegSubsampling,
    pub optimization: JpegOptimization,
    pub arithmetic_coding: bool,
    pub smoothing: u8,
    pub trellis_optimization: bool,
    pub progressive_scans: u8,
}

impl Default for JpegSettings {
    fn default() -> Self {
        Self {
            quality: 90,
            progressive: false,
            subsampling: JpegSubsampling::Ratio420,
            optimization: JpegOptimization::Balanced,
            arithmetic_coding: false,
            smoothing: 0,
            trellis_optimization: true,
            progressive_scans: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum PngFilterType {
    NoFilter,
    Sub,
    Up,
    Avg,
    Paeth,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum PngCompressionType {
    Fast,
    Default,
    Best,
    Huffman,
}

#[derive(Debug, Clone)]
pub struct PngSettings {
    pub compression_level: u8,
    pub compression_type: PngCompressionType,
    pub filter: PngFilterType,
    pub interlaced: bool,
    pub strip_metadata: bool,
    pub bit_depth: u8,
}

impl Default for PngSettings {
    fn default() -> Self {
        Self {
            compression_level: 6,
            compression_type: PngCompressionType::Default,
            filter: PngFilterType::Sub,
            interlaced: false,
            strip_metadata: false,
            bit_depth: 0,
        }
    }
}

impl PngFilterType {
    fn to_filter_type(self) -> ImageFilterType {
        match self {
            PngFilterType::NoFilter => ImageFilterType::NoFilter,
            PngFilterType::Sub => ImageFilterType::Sub,
            PngFilterType::Up => ImageFilterType::Up,
            PngFilterType::Avg => ImageFilterType::Avg,
            PngFilterType::Paeth => ImageFilterType::Paeth,
            PngFilterType::Adaptive => ImageFilterType::Adaptive,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum WebPPreset {
    Default,
    Picture,
    Photo,
    Drawing,
    Icon,
    Text,
}

#[derive(Debug, Clone)]
pub struct WebPSettings {
    pub quality: f32,
    pub lossless: bool,
    pub method: u8,
    pub preset: WebPPreset,
    pub threading: bool,
    pub target_size: usize,
}

impl Default for WebPSettings {
    fn default() -> Self {
        Self {
            quality: 80.0,
            lossless: false,
            method: 4,
            preset: WebPPreset::Default,
            threading: true,
            target_size: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum TiffCompression {
    None,
    Lzw,
    Deflate,
    PackBits,
    Jpeg,
}

#[derive(Debug, Clone)]
pub struct TiffSettings {
    pub compression: TiffCompression,
    pub jpeg_quality: u8,
    pub predictor: bool,
}

impl Default for TiffSettings {
    fn default() -> Self {
        Self {
            compression: TiffCompression::Lzw,
            jpeg_quality: 90,
            predictor: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct IcoSettings {
    pub max_dimension: u32,
    pub generate_multiple_sizes: bool,
    pub sizes: [u32; 8],
    pub num_sizes: usize,
}

impl Default for IcoSettings {
    fn default() -> Self {
        Self {
            max_dimension: 256,
            generate_multiple_sizes: false,
            sizes: [16, 32, 48, 64, 128, 256, 0, 0],
            num_sizes: 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum DdsFormat {
    RGBA8,
    BC1,
    BC2,
    BC3,
    BC4,
    BC5,
    BC6H,
    BC7,
}

#[derive(Debug, Clone)]
pub struct DdsSettings {
    pub format: DdsFormat,
    pub generate_mipmaps: bool,
    pub mipmap_count: u32,
    pub is_cubemap: bool,
}

impl Default for DdsSettings {
    fn default() -> Self {
        Self {
            format: DdsFormat::RGBA8,
            generate_mipmaps: true,
            mipmap_count: 0,
            is_cubemap: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GifSettings {
    pub palette_size: u16,
    pub dithering: bool,
    pub quality: u8,
}

impl Default for GifSettings {
    fn default() -> Self {
        Self {
            palette_size: 256,
            dithering: false,
            quality: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressionSettings {
    pub jpeg: JpegSettings,
    pub png: PngSettings,
    pub webp: WebPSettings,
    pub tiff: TiffSettings,
    pub ico: IcoSettings,
    pub dds: DdsSettings,
    pub gif: GifSettings,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            jpeg: JpegSettings::default(),
            png: PngSettings::default(),
            webp: WebPSettings::default(),
            tiff: TiffSettings::default(),
            ico: IcoSettings::default(),
            dds: DdsSettings::default(),
            gif: GifSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum ResizeFilter {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

impl ResizeFilter {
    fn to_image_filter(self) -> image::imageops::FilterType {
        match self {
            ResizeFilter::Nearest => image::imageops::FilterType::Nearest,
            ResizeFilter::Triangle => image::imageops::FilterType::Triangle,
            ResizeFilter::CatmullRom => image::imageops::FilterType::CatmullRom,
            ResizeFilter::Gaussian => image::imageops::FilterType::Gaussian,
            ResizeFilter::Lanczos3 => image::imageops::FilterType::Lanczos3,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResizeSettings {
    pub width: u32,
    pub height: u32,
    pub keep_aspect_ratio: bool,
    pub filter: ResizeFilter,
    pub only_shrink: bool,
}

impl Default for ResizeSettings {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            keep_aspect_ratio: true,
            filter: ResizeFilter::Lanczos3,
            only_shrink: false,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct ParallelConfig {
    pub tile_size: u32,
    pub enable_tiling: bool,
    pub min_size_for_parallel: u32,
    pub thread_count: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            tile_size: 512,
            enable_tiling: true,
            min_size_for_parallel: 1024,
            thread_count: 0,
        }
    }
}

pub struct ImageConverter {
    output_format: Format,
    compression: CompressionSettings,
    resize: Option<ResizeSettings>,
    strip_metadata: bool,
    enable_optimization: bool,
    parallel_config: ParallelConfig,
}

impl ImageConverter {
    pub fn new(output_format: Format) -> Self {
        rayon::ThreadPoolBuilder::new()
            .num_threads(0)
            .build_global()
            .ok();

        Self {
            output_format,
            compression: CompressionSettings::default(),
            resize: None,
            strip_metadata: false,
            enable_optimization: true,
            parallel_config: ParallelConfig::default(),
        }
    }

    pub fn set_strip_metadata(&mut self, strip: bool) {
        self.strip_metadata = strip;
    }

    pub fn set_enable_optimization(&mut self, enable: bool) {
        self.enable_optimization = enable;
    }

    pub fn set_parallel_config(&mut self, config: ParallelConfig) {
        self.parallel_config = config;
        
        if config.thread_count > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.thread_count)
                .build_global()
                .ok();
        }
    }

    pub fn with_compression(mut self, compression: CompressionSettings) -> Self {
        self.compression = compression;
        self
    }
    
    pub fn with_jpeg_settings(mut self, settings: JpegSettings) -> Self {
        self.compression.jpeg = settings;
        self
    }
    
    pub fn with_png_settings(mut self, settings: PngSettings) -> Self {
        self.compression.png = settings;
        self
    }
    
    pub fn with_webp_settings(mut self, settings: WebPSettings) -> Self {
        self.compression.webp = settings;
        self
    }
    
    pub fn with_resize(mut self, settings: ResizeSettings) -> Self {
        self.resize = Some(settings);
        self
    }
    
    pub fn strip_metadata(mut self, strip: bool) -> Self {
        self.strip_metadata = strip;
        self
    }
    
    pub fn enable_optimization(mut self, enable: bool) -> Self {
        self.enable_optimization = enable;
        self
    }

    pub fn with_parallel_config(mut self, config: ParallelConfig) -> Self {
        self.parallel_config = config;
        self
    }
    fn resize_image_parallel(&self, img: DynamicImage) -> DynamicImage {
        let Some(resize) = &self.resize else {
            return img;
        };
        
        if resize.width == 0 && resize.height == 0 {
            return img;
        }
        
        if resize.only_shrink && img.width() <= resize.width && img.height() <= resize.height {
            return img;
        }
        let (target_width, target_height) = if resize.width == 0 || resize.height == 0 {
            if resize.width == 0 {
                let new_height = resize.height;
                let ratio = new_height as f32 / img.height() as f32;
                let new_width = (img.width() as f32 * ratio).round() as u32;
                (new_width, new_height)
            } else {
                let new_width = resize.width;
                let ratio = new_width as f32 / img.width() as f32;
                let new_height = (img.height() as f32 * ratio).round() as u32;
                (new_width, new_height)
            }
        } else if resize.keep_aspect_ratio {
            let ratio_w = resize.width as f32 / img.width() as f32;
            let ratio_h = resize.height as f32 / img.height() as f32;
            let ratio = ratio_w.min(ratio_h);
            let new_width = (img.width() as f32 * ratio).round() as u32;
            let new_height = (img.height() as f32 * ratio).round() as u32;
            (new_width, new_height)
        } else {
            (resize.width, resize.height)
        };
        if !self.parallel_config.enable_tiling 
            || img.width().max(img.height()) < self.parallel_config.min_size_for_parallel {
            return img.resize_exact(target_width, target_height, resize.filter.to_image_filter());
        }
        self.resize_tiled_parallel(&img, target_width, target_height, resize.filter)
    }
    fn resize_tiled_parallel(&self, img: &DynamicImage, width: u32, height: u32, filter: ResizeFilter) -> DynamicImage {
        let tile_size = if self.parallel_config.tile_size == 0 {
            512
        } else {
            self.parallel_config.tile_size
        };

        let src_width = img.width();
        let src_height = img.height();
        if width * height > src_width * src_height * 4 {
            return img.resize_exact(width, height, filter.to_image_filter());
        }
        let scale_x = width as f64 / src_width as f64;
        let scale_y = height as f64 / src_height as f64;
        let mut output = RgbaImage::new(width, height);
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;
        let tiles: Vec<_> = (0..tiles_y)
            .flat_map(|ty| (0..tiles_x).map(move |tx| (tx, ty)))
            .collect();

        let tile_data: Vec<_> = tiles.par_iter().map(|&(tx, ty)| {
            let x_start = tx * tile_size;
            let y_start = ty * tile_size;
            let x_end = (x_start + tile_size).min(width);
            let y_end = (y_start + tile_size).min(height);
            let overlap = 4;
            let src_x_start = ((x_start as f64 / scale_x) as i32 - overlap).max(0) as u32;
            let src_y_start = ((y_start as f64 / scale_y) as i32 - overlap).max(0) as u32;
            let src_x_end = ((x_end as f64 / scale_x) as u32 + overlap as u32).min(src_width);
            let src_y_end = ((y_end as f64 / scale_y) as u32 + overlap as u32).min(src_height);
            let tile_img = img.crop_imm(src_x_start, src_y_start, 
                                        src_x_end - src_x_start, 
                                        src_y_end - src_y_start);
            let resized_tile = tile_img.resize_exact(
                x_end - x_start,
                y_end - y_start,
                filter.to_image_filter()
            ).to_rgba8();

            (x_start, y_start, resized_tile)
        }).collect();
        for (x_start, y_start, tile) in tile_data {
            for y in 0..tile.height() {
                for x in 0..tile.width() {
                    output.put_pixel(x_start + x, y_start + y, *tile.get_pixel(x, y));
                }
            }
        }

        DynamicImage::ImageRgba8(output)
    }
    
    pub fn convert(&self, input_data: &[u8]) -> Result<Vec<u8>> {
        let img = image::load_from_memory(input_data)?;
        self.convert_image(&img)
    }
    pub fn convert_batch(&self, inputs: &[&[u8]]) -> Vec<Result<Vec<u8>>> {
        inputs.par_iter()
            .map(|data| self.convert(data))
            .collect()
    }
    pub fn convert_batch_with_progress<F>(&self, inputs: &[&[u8]], progress: F) -> Vec<Result<Vec<u8>>>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let counter = AtomicUsize::new(0);
        let total = inputs.len();
        
        let results: Vec<_> = inputs.par_iter()
            .map(|data| {
                let result = self.convert(data);
                let count = counter.fetch_add(1, Ordering::SeqCst) + 1;
                progress(count, total);
                result
            })
            .collect();
        
        results
    }
    
    pub fn convert_image(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        let img = self.resize_image_parallel(img.clone());
        let mut output = Cursor::new(Vec::new());
        
        match self.output_format {
            Format::Png => self.encode_png_parallel(&img, &mut output)?,
            Format::Jpeg => self.encode_jpeg_parallel(&img, &mut output)?,
            Format::Ico => self.encode_ico_parallel(&img, &mut output)?,
            Format::Tiff => self.encode_tiff(&img, &mut output)?,
            Format::Bmp => self.encode_bmp(&img, &mut output)?,
            Format::Dds => self.encode_dds_parallel(&img, &mut output)?,
            Format::Hdr => self.encode_hdr(&img, &mut output)?,
            Format::Farbfeld => self.encode_farbfeld(&img, &mut output)?,
            Format::WebP => self.encode_webp_parallel(&img, &mut output)?,
            Format::Gif => self.encode_gif(&img, &mut output)?,
            _ => {
                img.write_to(&mut output, self.output_format.to_image_format())?;
            }
        }
        
        Ok(output.into_inner())
    }
    fn encode_jpeg_parallel(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let settings = &self.compression.jpeg;

        if settings.quality == 0 || settings.quality > 100 {
            return Err(ConversionError::InvalidParameter(
                "JPEG quality must be between 1 and 100".to_string()
            ));
        }

        let rgb_img = img.to_rgb8();
        let width = rgb_img.width();
        let height = rgb_img.height();
        if width * height < self.parallel_config.min_size_for_parallel * self.parallel_config.min_size_for_parallel {
            return self.encode_jpeg_standard(&rgb_img, width, height, output);
        }
        let pixels = rgb_img.into_raw();
        let row_stride = width as usize * 3;
        let processed_pixels = if settings.smoothing > 0 {
            self.parallel_smooth_image(&pixels, width, height, settings.smoothing)
        } else {
            pixels
        };
        let mut comp = Compress::new(MozColorSpace::JCS_RGB);
        comp.set_size(width as usize, height as usize);
        comp.set_quality(settings.quality as f32);

        if settings.progressive {
            comp.set_scan_optimization_mode(ScanMode::AllComponentsTogether);
            comp.set_progressive_mode();
        }

        match settings.optimization {
            JpegOptimization::Speed => comp.set_optimize_coding(false),
            _ => comp.set_optimize_coding(true),
        }

        if settings.trellis_optimization {
            comp.set_optimize_coding(true);
        }

        let mut writer = output;
        let mut started = comp.start_compress(&mut writer)
            .map_err(|e| ConversionError::JpegError(format!("MozJPEG start_compress failed: {}", e)))?;

        for y in 0..height as usize {
            let row = &processed_pixels[y * row_stride..(y + 1) * row_stride];
            started.write_scanlines(row)
                .map_err(|e| ConversionError::JpegError(format!("MozJPEG write_scanlines failed: {}", e)))?;
        }

        started.finish()
            .map_err(|e| ConversionError::JpegError(format!("MozJPEG finish failed: {}", e)))?;

        Ok(())
    }

    fn encode_jpeg_standard(&self, rgb_img: &image::RgbImage, width: u32, height: u32, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let settings = &self.compression.jpeg;
        let pixels = rgb_img.as_raw();

        let mut comp = Compress::new(MozColorSpace::JCS_RGB);
        comp.set_size(width as usize, height as usize);
        comp.set_quality(settings.quality as f32);

        if settings.progressive {
            comp.set_scan_optimization_mode(ScanMode::AllComponentsTogether);
            comp.set_progressive_mode();
        }

        match settings.optimization {
            JpegOptimization::Speed => comp.set_optimize_coding(false),
            _ => comp.set_optimize_coding(true),
        }

        let mut writer = output;
        let mut started = comp.start_compress(&mut writer)
            .map_err(|e| ConversionError::JpegError(format!("MozJPEG failed: {}", e)))?;

        let row_stride = width as usize * 3;
        for y in 0..height as usize {
            let row = &pixels[y * row_stride..(y + 1) * row_stride];
            started.write_scanlines(row)
                .map_err(|e| ConversionError::JpegError(format!("MozJPEG failed: {}", e)))?;
        }

        started.finish()
            .map_err(|e| ConversionError::JpegError(format!("MozJPEG failed: {}", e)))?;

        Ok(())
    }
    fn parallel_smooth_image(&self, pixels: &[u8], width: u32, height: u32, _smoothing: u8) -> Vec<u8> {
        let row_stride = width as usize * 3;
        
        (0..height as usize).into_par_iter()
            .flat_map(|y| {
                let row = &pixels[y * row_stride..(y + 1) * row_stride];
                row.to_vec()
            })
            .collect()
    }
    fn encode_png_parallel(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let settings = &self.compression.png;
        
        let compression_type = match settings.compression_type {
            PngCompressionType::Fast => CompressionType::Fast,
            PngCompressionType::Default => CompressionType::Default,
            PngCompressionType::Best => CompressionType::Best,
            #[allow(deprecated)]
            PngCompressionType::Huffman => CompressionType::Huffman,
        };
        let encoder = PngEncoder::new_with_quality(
            output,
            compression_type,
            settings.filter.to_filter_type(),
        );
        
        img.write_with_encoder(encoder)?;
        Ok(())
    }
    #[cfg(feature = "webp-encoding")]
    fn encode_webp_parallel(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let settings = &self.compression.webp;
        
        let rgba_img = img.to_rgba8();
        let width = rgba_img.width();
        let height = rgba_img.height();
        let pixels = rgba_img.as_raw();
        let encoded = if settings.lossless {
            webp::Encoder::from_rgba(pixels, width, height)
                .encode_lossless()
        } else {
            webp::Encoder::from_rgba(pixels, width, height)
                .encode(settings.quality)
        };
        
        output.write_all(&encoded)?;
        Ok(())
    }
    
    #[cfg(not(feature = "webp-encoding"))]
    fn encode_webp_parallel(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        img.write_to(output, ImageFormat::WebP)?;
        Ok(())
    }
    
    fn encode_gif(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        img.write_to(output, ImageFormat::Gif)?;
        Ok(())
    }
    fn encode_ico_parallel(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let settings = &self.compression.ico;
        
        if settings.generate_multiple_sizes && settings.num_sizes > 1 {
            let sizes: Vec<u32> = settings.sizes[..settings.num_sizes]
                .iter()
                .filter(|&&s| s > 0 && s <= settings.max_dimension)
                .copied()
                .collect();

            let resized_images: Vec<_> = sizes.par_iter()
                .map(|&size| {
                    img.resize(size, size, image::imageops::FilterType::Lanczos3).to_rgba8()
                })
                .collect();
            let frames: Vec<IcoFrame> = resized_images
                .iter()
                .map(|img| {
                    IcoFrame::as_png(
                        img.as_raw(),
                        img.width(),
                        img.height(),
                        ColorType::Rgba8,
                    ).map_err(|e| ConversionError::EncodingError(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;

            let encoder = IcoEncoder::new(output);
            encoder.encode_images(&frames)?;
        } else {
            let max_dim = settings.max_dimension;
            let resized = if img.width() > max_dim || img.height() > max_dim {
                img.resize(max_dim, max_dim, image::imageops::FilterType::Lanczos3)
            } else {
                img.clone()
            };
            
            let rgba_img = resized.to_rgba8();
            let encoder = IcoEncoder::new(output);
            
            encoder.write_image(
                rgba_img.as_raw(),
                rgba_img.width(),
                rgba_img.height(),
                ColorType::Rgba8,
            )?;
        }
        
        Ok(())
    }
    
    fn encode_tiff(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let encoder = TiffEncoder::new(output);
        img.write_with_encoder(encoder)?;
        Ok(())
    }
    
    fn encode_bmp(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let encoder = BmpEncoder::new(output);
        img.write_with_encoder(encoder)?;
        Ok(())
    }
    fn encode_dds_parallel(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let settings = &self.compression.dds;
        
        let rgba_img = img.to_rgba8();
        let width = rgba_img.width();
        let height = rgba_img.height();

        let dxgi_format = match settings.format {
            DdsFormat::RGBA8 => DxgiFormat::R8G8B8A8_UNorm,
            DdsFormat::BC1 => DxgiFormat::BC1_UNorm,
            DdsFormat::BC2 => DxgiFormat::BC2_UNorm,
            DdsFormat::BC3 => DxgiFormat::BC3_UNorm,
            DdsFormat::BC4 => DxgiFormat::BC4_UNorm,
            DdsFormat::BC5 => DxgiFormat::BC5_UNorm,
            DdsFormat::BC6H => DxgiFormat::BC6H_UF16,
            DdsFormat::BC7 => DxgiFormat::BC7_UNorm,
        };
        let (raw_data, mipmap_levels) = if settings.generate_mipmaps {
            self.generate_mipmaps_parallel(&rgba_img, settings.mipmap_count)
        } else {
            (rgba_img.into_raw(), None)
        };

        let params = NewDxgiParams {
            width,
            height,
            format: dxgi_format,
            mipmap_levels,
            array_layers: Some(1),
            caps2: Some(Caps2::empty()),
            alpha_mode: AlphaMode::Straight,
            resource_dimension: D3D10ResourceDimension::Texture2D,
            depth: Some(1),
            is_cubemap: settings.is_cubemap,
        };

        let mut dds = Dds::new_dxgi(params)
            .map_err(|e| ConversionError::DdsError(e.to_string()))?;

        dds.data = raw_data;

        dds.write(output)
            .map_err(|e| ConversionError::DdsError(e.to_string()))?;

        Ok(())
    }
    fn generate_mipmaps_parallel(&self, img: &RgbaImage, max_levels: u32) -> (Vec<u8>, Option<u32>) {
        let width = img.width();
        let height = img.height();
        
        let max_possible_levels = (width.min(height) as f32).log2().floor() as u32 + 1;
        let levels = if max_levels == 0 {
            max_possible_levels
        } else {
            max_levels.min(max_possible_levels)
        };

        if levels <= 1 {
            return (img.clone().into_raw(), Some(1));
        }
        let mipmap_images: Vec<_> = (0..levels).into_par_iter()
            .map(|level| {
                if level == 0 {
                    img.clone()
                } else {
                    let scale = 2u32.pow(level);
                    let mip_width = (width / scale).max(1);
                    let mip_height = (height / scale).max(1);
                    
                    image::imageops::resize(
                        img,
                        mip_width,
                        mip_height,
                        image::imageops::FilterType::Lanczos3
                    )
                }
            })
            .collect();
        let combined_data: Vec<u8> = mipmap_images.iter()
            .flat_map(|mip| mip.as_raw().iter().copied())
            .collect();

        (combined_data, Some(levels))
    }
    
    fn encode_hdr(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let rgb_img = img.to_rgb32f();
        let width = rgb_img.width() as usize;
        let height = rgb_img.height() as usize;
        let raw_data = rgb_img.into_raw();
        
        let rgb_data: Vec<Rgb<f32>> = raw_data
            .chunks_exact(3)
            .map(|chunk| Rgb([chunk[0], chunk[1], chunk[2]]))
            .collect();
        
        let encoder = image::codecs::hdr::HdrEncoder::new(output);
        encoder.encode(&rgb_data, width, height)
            .map_err(|e| ConversionError::EncodingError(e.to_string()))?;
        
        Ok(())
    }
    
    fn encode_farbfeld(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let rgba_img = img.to_rgba16();
        let width = rgba_img.width();
        let height = rgba_img.height();
        let raw_data = rgba_img.into_raw();
        
        if width == 0 || height == 0 || width > 0xFFFF || height > 0xFFFF {
            return Err(ConversionError::FarbfeldError(
                "Invalid image dimensions".to_string()
            ));
        }
        
        output.write_all(b"farbfeld")?;
        output.write_all(&(width as u32).to_be_bytes())?;
        output.write_all(&(height as u32).to_be_bytes())?;
        let converted: Vec<u8> = raw_data.par_chunks(4)
            .flat_map(|chunk| {
                let r = u16::from(chunk[0]);
                let g = u16::from(chunk[1]);
                let b = u16::from(chunk[2]);
                let a = u16::from(chunk[3]);
                
                [
                    r.to_be_bytes()[0], r.to_be_bytes()[1],
                    g.to_be_bytes()[0], g.to_be_bytes()[1],
                    b.to_be_bytes()[0], b.to_be_bytes()[1],
                    a.to_be_bytes()[0], a.to_be_bytes()[1],
                ]
            })
            .collect();
        
        output.write_all(&converted)?;
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub color_type: ImageColorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum ImageColorType {
    Grayscale,
    GrayscaleAlpha,
    Rgb,
    Rgba,
    Grayscale16,
    GrayscaleAlpha16,
    Rgb16,
    Rgba16,
    Rgb32F,
    Rgba32F,
    Unknown,
}

impl ImageConverter {
    pub fn get_image_info(input_data: &[u8]) -> Result<ImageInfo> {
        let img = image::load_from_memory(input_data)?;
        
        Ok(ImageInfo {
            width: img.width(),
            height: img.height(),
            color_type: match img.color() {
                ColorType::L8 => ImageColorType::Grayscale,
                ColorType::La8 => ImageColorType::GrayscaleAlpha,
                ColorType::Rgb8 => ImageColorType::Rgb,
                ColorType::Rgba8 => ImageColorType::Rgba,
                ColorType::L16 => ImageColorType::Grayscale16,
                ColorType::La16 => ImageColorType::GrayscaleAlpha16,
                ColorType::Rgb16 => ImageColorType::Rgb16,
                ColorType::Rgba16 => ImageColorType::Rgba16,
                ColorType::Rgb32F => ImageColorType::Rgb32F,
                ColorType::Rgba32F => ImageColorType::Rgba32F,
                _ => ImageColorType::Unknown,
            },
        })
    }
    pub fn get_batch_image_info(inputs: &[&[u8]]) -> Vec<Result<ImageInfo>> {
        inputs.par_iter()
            .map(|data| Self::get_image_info(data))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_image() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_fn(100, 100, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        }))
    }

    fn create_large_test_image() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_fn(2048, 2048, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        }))
    }
    
    #[test]
    fn test_parallel_jpeg_encoding() {
        let img = create_large_test_image();
        
        let converter = ImageConverter::new(Format::Jpeg)
            .with_jpeg_settings(JpegSettings {
                quality: 95,
                progressive: true,
                optimization: JpegOptimization::Quality,
                ..Default::default()
            });
        
        let result = converter.convert_image(&img);
        assert!(result.is_ok());
        
        let data = result.unwrap();
        assert!(data.len() > 0);
        assert_eq!(&data[0..2], &[0xFF, 0xD8]);
    }
    
    #[test]
    fn test_parallel_resize() {
        let img = create_large_test_image();
        
        let converter = ImageConverter::new(Format::Png)
            .with_resize(ResizeSettings {
                width: 512,
                height: 512,
                keep_aspect_ratio: false,
                filter: ResizeFilter::Lanczos3,
                only_shrink: false,
            })
            .with_parallel_config(ParallelConfig {
                tile_size: 256,
                enable_tiling: true,
                min_size_for_parallel: 1024,
                thread_count: 0,
            });
        
        let result = converter.convert_image(&img);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_conversion() {
        let images: Vec<DynamicImage> = (0..10)
            .map(|_| create_test_image())
            .collect();

        let converter = ImageConverter::new(Format::Jpeg);
        
        let img_bytes: Vec<Vec<u8>> = images.iter()
            .map(|img| {
                let mut buf = Cursor::new(Vec::new());
                img.write_to(&mut buf, ImageFormat::Png).unwrap();
                buf.into_inner()
            })
            .collect();

        let refs: Vec<&[u8]> = img_bytes.iter().map(|v| v.as_slice()).collect();
        
        let results = converter.convert_batch(&refs);
        
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_parallel_mipmap_generation() {
        let img = create_test_image();
        
        let converter = ImageConverter::new(Format::Dds)
            .with_compression(CompressionSettings {
                dds: DdsSettings {
                    format: DdsFormat::RGBA8,
                    generate_mipmaps: true,
                    mipmap_count: 0,
                    is_cubemap: false,
                },
                ..Default::default()
            });
        
        let result = converter.convert_image(&img);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parallel_ico_multi_size() {
        let img = create_test_image();
        
        let converter = ImageConverter::new(Format::Ico)
            .with_compression(CompressionSettings {
                ico: IcoSettings {
                    max_dimension: 256,
                    generate_multiple_sizes: true,
                    sizes: [16, 32, 48, 64, 128, 256, 0, 0],
                    num_sizes: 6,
                },
                ..Default::default()
            });
        
        let result = converter.convert_image(&img);
        assert!(result.is_ok());
    }
}