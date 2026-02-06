use image::{
    DynamicImage, ImageBuffer, ImageError, ImageFormat, ImageOutputFormat,
    codecs::jpeg::JpegEncoder,
    codecs::png::{PngEncoder, CompressionType, FilterType},
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
}

#[derive(Debug, Clone)]
pub struct CompressionSettings {
    pub jpeg_quality: u8,
    pub png_compression: u8,
    pub png_filter: PngFilterType,
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

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            jpeg_quality: 90,
            png_compression: 6,
            png_filter: PngFilterType::Sub,
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
    
    pub fn convert(&self, input_data: &[u8]) -> Result<Vec<u8>> {
        let img = image::load_from_memory(input_data)?;
        self.convert_image(&img)
    }
    
    pub fn convert_image(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        let mut output = Cursor::new(Vec::new());
        
        match self.output_format {
            Format::Png => {
                let encoder = PngEncoder::new_with_quality(
                    &mut output,
                    match self.compression.png_compression {
                        0..=1 => CompressionType::Fast,
                        2..=5 => CompressionType::Default,
                        _ => CompressionType::Best,
                    },
                    self.compression.png_filter.to_filter_type(),
                );
                
                img.write_with_encoder(encoder)?;
            }
            Format::Jpeg => {
                let mut encoder = JpegEncoder::new_with_quality(
                    &mut output,
                    self.compression.jpeg_quality,
                );
                
                let rgb_img = img.to_rgb8();
                encoder.encode(
                    rgb_img.as_raw(),
                    rgb_img.width(),
                    rgb_img.height(),
                    image::ColorType::Rgb8,
                )?;
            }
            _ => {
                img.write_to(&mut output, self.output_format.to_image_format())?;
            }
        }
        
        Ok(output.into_inner())
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
    pub unsafe extern "C" fn image_convert(
        input_data: *const u8,
        input_len: usize,
        output_format: Format,
        jpeg_quality: u8,
        png_compression: u8,
        png_filter: PngFilterType,
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
            }
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
    fn test_png_to_jpeg() {
        let img = DynamicImage::ImageRgb8(
            ImageBuffer::from_fn(100, 100, |x, y| {
                image::Rgb([
                    (x % 256) as u8,
                    (y % 256) as u8,
                    ((x + y) % 256) as u8,
                ])
            })
        );
        
        let mut png_data = Cursor::new(Vec::new());
        img.write_to(&mut png_data, ImageFormat::Png).unwrap();
        
        let converter = ImageConverter::new(Format::Jpeg)
            .with_jpeg_quality(85).unwrap();
        
        let jpeg_data = converter.convert(png_data.get_ref()).unwrap();
        assert!(!jpeg_data.is_empty());
        
        let result_img = image::load_from_memory(&jpeg_data).unwrap();
        assert_eq!(result_img.width(), 100);
        assert_eq!(result_img.height(), 100);
    }
}
