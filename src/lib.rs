use image::{
    DynamicImage, ImageError, ImageFormat,
    codecs::jpeg::JpegEncoder,
    codecs::png::{PngEncoder, CompressionType, FilterType},
    codecs::ico::IcoEncoder,
    codecs::bmp::BmpEncoder,
    codecs::tiff::TiffEncoder,
    ColorType, Rgb, ImageEncoder,
};
use std::io::{Cursor, Write};
use thiserror::Error;
use ddsfile::{Dds, NewDxgiParams, DxgiFormat, Caps2, AlphaMode, D3D10ResourceDimension};

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
    
    fn supports_alpha(self) -> bool {
        matches!(
            self,
            Format::Png | Format::Ico | Format::Tiff | Format::WebP | Format::Tga | Format::Dds | Format::Farbfeld
        )
    }
}

#[derive(Debug, Clone)]
pub struct CompressionSettings {
    pub jpeg_quality: u8,
    pub png_compression: u8,
    pub png_filter: PngFilterType,
    pub ico_settings: IcoSettings,
    pub dds_mipmaps: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum PngFilterType {
    NoFilter,
    Sub,
    Up,
    Avg,
    Paeth,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct IcoSettings {
    pub max_dimension: u32,
    pub generate_multiple_sizes: bool,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            jpeg_quality: 90,
            png_compression: 6,
            png_filter: PngFilterType::Sub,
            ico_settings: IcoSettings {
                max_dimension: 256,
                generate_multiple_sizes: false,
            },
            dds_mipmaps: true,
        }
    }
}

impl PngFilterType {
    fn to_filter_type(self) -> FilterType {
        match self {
            PngFilterType::NoFilter => FilterType::NoFilter,
            PngFilterType::Sub => FilterType::Sub,
            PngFilterType::Up => FilterType::Up,
            PngFilterType::Avg => FilterType::Avg,
            PngFilterType::Paeth => FilterType::Paeth,
        }
    }
}

impl Default for IcoSettings {
    fn default() -> Self {
        Self {
            max_dimension: 256,
            generate_multiple_sizes: false,
        }
    }
}

pub struct ImageConverter {
    output_format: Format,
    compression: CompressionSettings,
}

impl ImageConverter {
    pub fn new(output_format: Format) -> Self {
        Self {
            output_format,
            compression: CompressionSettings::default(),
        }
    }
    
    pub fn with_compression(mut self, compression: CompressionSettings) -> Self {
        self.compression = compression;
        self
    }
    
    pub fn with_jpeg_quality(mut self, quality: u8) -> Result<Self> {
        if quality == 0 || quality > 100 {
            return Err(ConversionError::InvalidParameter(
                "JPEG quality must be between 1 and 100".to_string()
            ));
        }
        self.compression.jpeg_quality = quality;
        Ok(self)
    }
    
    pub fn with_png_compression(mut self, level: u8) -> Result<Self> {
        if level > 9 {
            return Err(ConversionError::InvalidParameter(
                "PNG compression level must be between 0 and 9".to_string()
            ));
        }
        self.compression.png_compression = level;
        Ok(self)
    }
    
    pub fn with_ico_settings(mut self, settings: IcoSettings) -> Self {
        self.compression.ico_settings = settings;
        self
    }
    
    pub fn with_dds_mipmaps(mut self, mipmaps: bool) -> Self {
        self.compression.dds_mipmaps = mipmaps;
        self
    }
    
    pub fn convert(&self, input_data: &[u8]) -> Result<Vec<u8>> {
        let img = image::load_from_memory(input_data)?;
        self.convert_image(&img)
    }
    
    pub fn convert_image(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        let mut output = Cursor::new(Vec::new());
        
        match self.output_format {
            Format::Png => {
                self.encode_png(img, &mut output)?;
            }
            Format::Jpeg => {
                self.encode_jpeg(img, &mut output)?;
            }
            Format::Ico => {
                self.encode_ico(img, &mut output)?;
            }
            Format::Tiff => {
                self.encode_tiff(img, &mut output)?;
            }
            Format::Bmp => {
                self.encode_bmp(img, &mut output)?;
            }
            Format::Dds => {
                self.encode_dds(img, &mut output)?;
            }
            Format::Hdr => {
                self.encode_hdr(img, &mut output)?;
            }
            Format::Farbfeld => {
                self.encode_farbfeld(img, &mut output)?;
            }
            _ => {
                img.write_to(&mut output, self.output_format.to_image_format())?;
            }
        }
        
        Ok(output.into_inner())
    }
    
    fn encode_png(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let encoder = PngEncoder::new_with_quality(
            output,
            match self.compression.png_compression {
                0..=1 => CompressionType::Fast,
                2..=5 => CompressionType::Default,
                _ => CompressionType::Best,
            },
            self.compression.png_filter.to_filter_type(),
        );
        
        img.write_with_encoder(encoder)?;
        Ok(())
    }
    
    fn encode_jpeg(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let mut encoder = JpegEncoder::new_with_quality(
            output,
            self.compression.jpeg_quality,
        );
        
        let rgb_img = img.to_rgb8();
        encoder.encode(
            rgb_img.as_raw(),
            rgb_img.width(),
            rgb_img.height(),
            ColorType::Rgb8,
        )?;
        Ok(())
    }
    
    fn encode_ico(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let settings = &self.compression.ico_settings;
        
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
    
    fn encode_dds(&self, img: &DynamicImage, output: &mut Cursor<Vec<u8>>) -> Result<()> {
        let rgba_img = img.to_rgba8();
        let width = rgba_img.width();
        let height = rgba_img.height();
        let raw_data = rgba_img.into_raw();

        let params = NewDxgiParams {
            width,
            height,
            format: DxgiFormat::R8G8B8A8_UNorm,
            mipmap_levels: if self.compression.dds_mipmaps { Some(0) } else { None },
            array_layers: Some(1),
            caps2: Some(Caps2::empty()),
            alpha_mode: AlphaMode::Straight,
            resource_dimension: D3D10ResourceDimension::Texture2D,
            depth: Some(1),
            is_cubemap: false,
        };

        let mut dds = Dds::new_dxgi(params)
            .map_err(|e| ConversionError::DdsError(e.to_string()))?;

        dds.data = raw_data;

        dds.write(output)
            .map_err(|e| ConversionError::DdsError(e.to_string()))?;

        Ok(())
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
        
        for chunk in raw_data.chunks_exact(4) {
            let r = u16::from(chunk[0]);
            let g = u16::from(chunk[1]);
            let b = u16::from(chunk[2]);
            let a = u16::from(chunk[3]);
            
            output.write_all(&r.to_be_bytes())?;
            output.write_all(&g.to_be_bytes())?;
            output.write_all(&b.to_be_bytes())?;
            output.write_all(&a.to_be_bytes())?;
        }
        
        Ok(())
    }
    
    pub fn convert_with_format(
        input_data: &[u8],
        output_format: Format,
        compression: Option<CompressionSettings>,
    ) -> Result<Vec<u8>> {
        let mut converter = Self::new(output_format);
        if let Some(comp) = compression {
            converter = converter.with_compression(comp);
        }
        converter.convert(input_data)
    }
    
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

#[repr(C)]
pub struct RawPixels {
    pub data: *mut u8,
    pub len: usize,
    pub capacity: usize,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub color_type: ImageColorType,
}

impl RawPixels {
    pub fn new(data: Vec<u8>, width: u32, height: u32, channels: u32, color_type: ImageColorType) -> Self {
        let len = data.len();
        let capacity = data.capacity();
        let data_ptr = data.as_ptr() as *mut u8;
        std::mem::forget(data);
        
        Self {
            data: data_ptr,
            len,
            capacity,
            width,
            height,
            channels,
            color_type,
        }
    }
}

pub fn free_raw_pixels(pixels: RawPixels) {
    if !pixels.data.is_null() {
        unsafe {
            Vec::from_raw_parts(pixels.data, pixels.len, pixels.capacity);
        }
    }
}

pub fn to_bitmap(input_data: &[u8]) -> Result<Vec<u8>> {
    let converter = ImageConverter::new(Format::Bmp);
    converter.convert(input_data)
}

pub fn get_raw_pixels(input_data: &[u8]) -> Result<RawPixels> {
    let img = image::load_from_memory(input_data)?;
    
    match img.color() {
        ColorType::L8 | ColorType::L16 => {
            let gray_img = img.to_luma8();
            let width = gray_img.width();
            let height = gray_img.height();
            let data = gray_img.into_raw();
            
            let color_type = match img.color() {
                ColorType::L8 => ImageColorType::Grayscale,
                ColorType::L16 => ImageColorType::Grayscale16,
                _ => ImageColorType::Grayscale,
            };
            
            Ok(RawPixels::new(data, width, height, 1, color_type))
        }
        ColorType::La8 | ColorType::La16 => {
            let gray_alpha_img = img.to_luma_alpha8();
            let width = gray_alpha_img.width();
            let height = gray_alpha_img.height();
            let data = gray_alpha_img.into_raw();
            
            let color_type = match img.color() {
                ColorType::La8 => ImageColorType::GrayscaleAlpha,
                ColorType::La16 => ImageColorType::GrayscaleAlpha16,
                _ => ImageColorType::GrayscaleAlpha,
            };
            
            Ok(RawPixels::new(data, width, height, 2, color_type))
        }
        ColorType::Rgb8 | ColorType::Rgb16 | ColorType::Rgb32F => {
            let rgb_img = img.to_rgb8();
            let width = rgb_img.width();
            let height = rgb_img.height();
            let data = rgb_img.into_raw();
            
            let color_type = match img.color() {
                ColorType::Rgb8 => ImageColorType::Rgb,
                ColorType::Rgb16 => ImageColorType::Rgb16,
                ColorType::Rgb32F => ImageColorType::Rgb32F,
                _ => ImageColorType::Rgb,
            };
            
            Ok(RawPixels::new(data, width, height, 3, color_type))
        }
        ColorType::Rgba8 | ColorType::Rgba16 | ColorType::Rgba32F => {
            let rgba_img = img.to_rgba8();
            let width = rgba_img.width();
            let height = rgba_img.height();
            let data = rgba_img.into_raw();
            
            let color_type = match img.color() {
                ColorType::Rgba8 => ImageColorType::Rgba,
                ColorType::Rgba16 => ImageColorType::Rgba16,
                ColorType::Rgba32F => ImageColorType::Rgba32F,
                _ => ImageColorType::Rgba,
            };
            
            Ok(RawPixels::new(data, width, height, 4, color_type))
        }
        _ => {
            let rgba_img = img.to_rgba8();
            let width = rgba_img.width();
            let height = rgba_img.height();
            let data = rgba_img.into_raw();
            
            Ok(RawPixels::new(data, width, height, 4, ImageColorType::Rgba))
        }
    }
}

pub mod ffi {
    use super::*;
    use std::slice;
    use std::ptr;
    
    #[repr(C)]
    pub struct ImageData {
        data: *mut u8,
        len: usize,
        capacity: usize,
    }
    
    #[repr(C)]
    pub enum ResultCode {
        Success = 0,
        ErrorInvalidFormat = 1,
        ErrorImageProcessing = 2,
        ErrorEncoding = 3,
        ErrorInvalidParameter = 4,
        ErrorNullPointer = 5,
        ErrorUnsupportedOperation = 6,
        ErrorDds = 7,
        ErrorFarbfeld = 8,
    }
    
    #[repr(C)]
    pub struct CImageInfo {
        pub width: u32,
        pub height: u32,
        pub color_type: ImageColorType,
    }
    
    #[no_mangle]
    pub extern "C" fn image_data_free(img: *mut ImageData) {
        if img.is_null() {
            return;
        }
        
        unsafe {
            let img = Box::from_raw(img);
            if !img.data.is_null() {
                Vec::from_raw_parts(img.data, img.len, img.capacity);
            }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn image_convert_advanced(
        input_data: *const u8,
        input_len: usize,
        output_format: Format,
        jpeg_quality: u8,
        png_compression: u8,
        png_filter: PngFilterType,
        ico_max_dimension: u32,
        ico_generate_multiple: bool,
        dds_mipmaps: bool,
        output: *mut *mut ImageData,
    ) -> ResultCode {
        if input_data.is_null() || output.is_null() {
            return ResultCode::ErrorNullPointer;
        }
        
        let input_slice = slice::from_raw_parts(input_data, input_len);
        
        let compression = CompressionSettings {
            jpeg_quality: if jpeg_quality == 0 { 90 } else { jpeg_quality },
            png_compression: if png_compression > 9 { 6 } else { png_compression },
            png_filter,
            ico_settings: IcoSettings {
                max_dimension: if ico_max_dimension == 0 { 256 } else { ico_max_dimension },
                generate_multiple_sizes: ico_generate_multiple,
            },
            dds_mipmaps,
        };
        
        let converter = ImageConverter::new(output_format)
            .with_compression(compression);
        
        match converter.convert(input_slice) {
            Ok(mut result) => {
                let len = result.len();
                let capacity = result.capacity();
                let data = result.as_mut_ptr();
                std::mem::forget(result);
                
                let image_data = Box::new(ImageData {
                    data,
                    len,
                    capacity,
                });
                
                *output = Box::into_raw(image_data);
                ResultCode::Success
            }
            Err(e) => match e {
                ConversionError::InvalidFormat => ResultCode::ErrorInvalidFormat,
                ConversionError::InvalidParameter(_) => ResultCode::ErrorInvalidParameter,
                ConversionError::EncodingError(_) => ResultCode::ErrorEncoding,
                ConversionError::ImageError(_) => ResultCode::ErrorImageProcessing,
                ConversionError::UnsupportedOperation(_) => ResultCode::ErrorUnsupportedOperation,
                ConversionError::DdsError(_) => ResultCode::ErrorDds,
                ConversionError::FarbfeldError(_) => ResultCode::ErrorFarbfeld,
                ConversionError::IoError(_) => ResultCode::ErrorEncoding,
            }
        }
    }
    
    #[no_mangle]
    pub unsafe extern "C" fn image_convert(
        input_data: *const u8,
        input_len: usize,
        output_format: Format,
        jpeg_quality: u8,
        png_compression: u8,
        png_filter: PngFilterType,
        output: *mut *mut ImageData,
    ) -> ResultCode {
        image_convert_advanced(
            input_data,
            input_len,
            output_format,
            jpeg_quality,
            png_compression,
            png_filter,
            256,
            false,
            true,
            output,
        )
    }
    
    #[no_mangle]
    pub unsafe extern "C" fn image_get_info(
        input_data: *const u8,
        input_len: usize,
        info: *mut CImageInfo,
    ) -> ResultCode {
        if input_data.is_null() || info.is_null() {
            return ResultCode::ErrorNullPointer;
        }
        
        let input_slice = slice::from_raw_parts(input_data, input_len);
        
        match ImageConverter::get_image_info(input_slice) {
            Ok(image_info) => {
                *info = CImageInfo {
                    width: image_info.width,
                    height: image_info.height,
                    color_type: image_info.color_type,
                };
                ResultCode::Success
            }
            Err(_) => ResultCode::ErrorImageProcessing,
        }
    }
    
    #[no_mangle]
    pub unsafe extern "C" fn image_data_get_ptr(img: *const ImageData) -> *const u8 {
        if img.is_null() {
            return ptr::null();
        }
        (*img).data
    }
    
    #[no_mangle]
    pub unsafe extern "C" fn image_data_get_len(img: *const ImageData) -> usize {
        if img.is_null() {
            return 0;
        }
        (*img).len
    }
    
    #[no_mangle]
    pub unsafe extern "C" fn image_get_raw_pixels(
        input_data: *const u8,
        input_len: usize,
        raw_pixels: *mut RawPixels,
    ) -> ResultCode {
        if input_data.is_null() || raw_pixels.is_null() {
            return ResultCode::ErrorNullPointer;
        }
        
        let input_slice = slice::from_raw_parts(input_data, input_len);
        
        match get_raw_pixels(input_slice) {
            Ok(pixels) => {
                *raw_pixels = pixels;
                ResultCode::Success
            }
            Err(e) => match e {
                ConversionError::InvalidFormat => ResultCode::ErrorInvalidFormat,
                ConversionError::InvalidParameter(_) => ResultCode::ErrorInvalidParameter,
                ConversionError::EncodingError(_) => ResultCode::ErrorEncoding,
                ConversionError::ImageError(_) => ResultCode::ErrorImageProcessing,
                ConversionError::UnsupportedOperation(_) => ResultCode::ErrorUnsupportedOperation,
                ConversionError::DdsError(_) => ResultCode::ErrorDds,
                ConversionError::FarbfeldError(_) => ResultCode::ErrorFarbfeld,
                ConversionError::IoError(_) => ResultCode::ErrorEncoding,
            }
        }
    }
    
    #[no_mangle]
    pub extern "C" fn raw_pixels_free(pixels: RawPixels) {
        free_raw_pixels(pixels);
    }
    
    #[no_mangle]
    pub unsafe extern "C" fn raw_pixels_get_data(pixels: *const RawPixels) -> *const u8 {
        if pixels.is_null() {
            return ptr::null();
        }
        (*pixels).data
    }
    
    #[no_mangle]
    pub unsafe extern "C" fn raw_pixels_get_width(pixels: *const RawPixels) -> u32 {
        if pixels.is_null() {
            return 0;
        }
        (*pixels).width
    }
    
    #[no_mangle]
    pub unsafe extern "C" fn raw_pixels_get_height(pixels: *const RawPixels) -> u32 {
        if pixels.is_null() {
            return 0;
        }
        (*pixels).height
    }
    
    #[no_mangle]
    pub unsafe extern "C" fn raw_pixels_get_channels(pixels: *const RawPixels) -> u32 {
        if pixels.is_null() {
            return 0;
        }
        (*pixels).channels
    }
    
    #[no_mangle]
    pub unsafe extern "C" fn raw_pixels_get_len(pixels: *const RawPixels) -> usize {
        if pixels.is_null() {
            return 0;
        }
        (*pixels).len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_conversion() {
        let png_data = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        
        let converter = ImageConverter::new(Format::Jpeg);
        let result = converter.convert(&png_data);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_to_bitmap() {
        let png_data = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        
        let result = to_bitmap(&png_data);
        assert!(result.is_ok());
        
        let bitmap_data = result.unwrap();

        assert!(bitmap_data.len() > 2);
        assert_eq!(bitmap_data[0], b'B');
        assert_eq!(bitmap_data[1], b'M');
    }
    
    #[test]
    fn test_get_raw_pixels() {
        let png_data = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        
        let result = get_raw_pixels(&png_data);
        assert!(result.is_ok());
        
        let pixels = result.unwrap();

        assert!(pixels.len > 0);
        assert_eq!(pixels.width, 1);
        assert_eq!(pixels.height, 1);
        assert!(pixels.channels == 3 || pixels.channels == 4);

        free_raw_pixels(pixels);
    }
    
    #[test]
    fn test_dds_conversion() {
        let test_data = vec![0; 100];
        let converter = ImageConverter::new(Format::Dds);
        let result = converter.convert(&test_data);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_farbfeld_conversion() {
        let test_data = vec![0; 100];
        let converter = ImageConverter::new(Format::Farbfeld);
        let result = converter.convert(&test_data);
        assert!(result.is_err());
    }
}