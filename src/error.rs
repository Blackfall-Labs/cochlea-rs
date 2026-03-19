//! Error types for cochlea-rs.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CochleaError {
    #[error("Audio error: {0}")]
    Audio(String),

    #[error("MFCC extraction produced no frames")]
    EmptyAudio,

    #[error("Formant estimation failed: {0}")]
    FormantEstimation(String),

    #[error("Phoneme bank error: {0}")]
    PhonemeBank(String),

    #[error("Lexicon error: {0}")]
    Lexicon(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CochleaError>;
