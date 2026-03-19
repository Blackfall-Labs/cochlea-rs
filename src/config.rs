//! Configuration for the cochlea transcription pipeline.

use crate::mfcc::MfccConfig;
use std::path::PathBuf;

/// Top-level transcription configuration.
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    /// MFCC extraction configuration.
    pub mfcc: MfccConfig,
    /// Whether to apply cepstral mean normalization.
    pub cmn_enabled: bool,
    /// Pre-emphasis filter coefficient (0.0 to disable).
    pub pre_emphasis_coeff: f32,
    /// Number of leading frames to use for noise estimation (0 to disable).
    pub noise_estimation_frames: usize,
    /// Energy threshold for voice activity detection (0.0 to disable).
    pub vad_threshold: f32,
    /// Number of top phoneme candidates per frame.
    pub phoneme_top_k: usize,
    /// Maximum edit distance for fuzzy pronunciation trie lookup.
    pub max_edit_distance: usize,
    /// Minimum word confidence to include in transcript.
    pub min_word_confidence: f32,
    /// Paths to lexicon files.
    pub lexicon_engram: Option<PathBuf>,
    /// Path to lexicon BytePunch dictionary.
    pub lexicon_dict: Option<PathBuf>,
    /// Path to CML core dictionary.
    pub lexicon_core_dict: Option<PathBuf>,
}

impl TranscriptionConfig {
    /// Configuration preset for wideband audio (16kHz, clean or mild noise).
    pub fn wideband() -> Self {
        Self {
            mfcc: MfccConfig::default(), // 16kHz, 26 coeff, 40 mels
            cmn_enabled: true,
            pre_emphasis_coeff: 0.97,
            noise_estimation_frames: 10,
            vad_threshold: -30.0, // log energy threshold
            phoneme_top_k: 3,
            max_edit_distance: 2,
            min_word_confidence: 0.3,
            lexicon_engram: None,
            lexicon_dict: None,
            lexicon_core_dict: None,
        }
    }

    /// Configuration preset for phone-quality audio (8kHz, compressed, noisy).
    pub fn phone_quality() -> Self {
        Self {
            mfcc: MfccConfig::phone_quality(),
            cmn_enabled: true,
            pre_emphasis_coeff: 0.97,
            noise_estimation_frames: 15, // more frames for noisy estimation
            vad_threshold: -25.0,        // more aggressive VAD for phone
            phoneme_top_k: 5,            // more candidates for noisy conditions
            max_edit_distance: 3,        // more tolerance for errors
            min_word_confidence: 0.2,    // lower bar for noisy audio
            lexicon_engram: None,
            lexicon_dict: None,
            lexicon_core_dict: None,
        }
    }
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self::wideband()
    }
}
