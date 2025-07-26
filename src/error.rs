//! Error types for the simple-knn-wgpu library.
//! 
//! This module provides comprehensive error handling for all operations
//! in the library, ensuring robust error reporting and recovery.

use thiserror::Error;

/// Main error type for the simple-knn-wgpu library.
#[derive(Error, Debug)]
pub enum KnnError {
    /// Error occurred during GPU device initialization or adapter selection.
    #[error("GPU initialization failed: {0}")]
    GpuInitError(String),
    
    /// Error occurred during shader compilation or pipeline creation.
    #[error("Shader compilation failed: {0}")]
    ShaderError(String),
    
    /// Error occurred during buffer creation or memory allocation.
    #[error("Buffer allocation failed: {0}")]
    BufferError(String),
    
    /// Error occurred during compute pass execution.
    #[error("Compute execution failed: {0}")]
    ComputeError(String),
    
    /// Invalid input data provided to the algorithm.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Error occurred during data transfer between CPU and GPU.
    #[error("Data transfer failed: {0}")]
    TransferError(String),
    
    /// The requested operation is not supported on the current hardware.
    #[error("Operation not supported: {0}")]
    NotSupported(String),
    
    /// Generic wgpu error wrapper.
    #[error("wgpu error: {0}")]
    WgpuError(#[from] wgpu::Error),
    
    /// Error occurred during buffer mapping operations.
    #[error("Buffer mapping failed: {0}")]
    BufferMapError(#[from] wgpu::BufferAsyncError),
}

/// Result type alias for operations that may fail with a KnnError.
pub type Result<T> = std::result::Result<T, KnnError>;

/// Validation error types for input data.
#[derive(Error, Debug)]
pub enum ValidationError {
    /// Points array has invalid dimensions.
    #[error("Invalid points array: expected shape [N, 3], got shape [{0}, {1}]")]
    InvalidShape(usize, usize),
    
    /// Points array is empty.
    #[error("Points array is empty")]
    EmptyArray,
    
    /// Points array is too large for GPU processing.
    #[error("Points array too large: {0} points exceeds maximum of {1}")]
    TooLarge(usize, usize),
    
    /// Points contain invalid values (NaN or infinity).
    #[error("Points contain invalid values at index {0}")]
    InvalidValues(usize),
}

impl From<ValidationError> for KnnError {
    fn from(err: ValidationError) -> Self {
        KnnError::InvalidInput(err.to_string())
    }
} 