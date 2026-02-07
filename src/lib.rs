use image::{
    DynamicImage, ImageError, ImageFormat,
    codecs::jpeg::JpegEncoder,
    codecs::png::{PngEncoder, CompressionType, FilterType},
    codecs::ico::IcoEncoder,
    codecs::bmp::BmpEncoder,
    codecs::tiff::TiffEncoder,
    ColorType, Rgb, ImageEncoder
};
use std::io::Cursor;
use thiserror::Error;

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
        img.write_to(output, ImageFormat::Dds)?;
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
        img.write_to(output, ImageFormat::Farbfeld)?;
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
}