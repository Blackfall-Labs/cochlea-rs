//! MFCC (Mel-Frequency Cepstral Coefficients) extraction.
//!
//! Self-contained MFCC extractor for cochlea-rs. Avoids depending on hush
//! (which has pre-existing compilation issues from ternary-signal API changes).
//!
//! Converts raw audio waveform to frequency-domain features suitable for
//! formant estimation and phoneme recognition.

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// Configuration for MFCC extraction.
#[derive(Debug, Clone)]
pub struct MfccConfig {
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Number of MFCC coefficients to extract.
    pub n_mfcc: usize,
    /// Number of mel filterbank channels.
    pub n_mels: usize,
    /// FFT window size in samples.
    pub n_fft: usize,
    /// Hop length between frames in samples.
    pub hop_length: usize,
    /// Window length in samples.
    pub win_length: usize,
    /// Minimum frequency for mel filterbank.
    pub f_min: f32,
    /// Maximum frequency for mel filterbank.
    pub f_max: f32,
}

impl MfccConfig {
    /// Standard wideband configuration (16kHz).
    pub fn wideband() -> Self {
        Self::default()
    }

    /// Phone-quality configuration (8kHz).
    pub fn phone_quality() -> Self {
        Self {
            sample_rate: 8000,
            n_mfcc: 26,
            n_mels: 26,
            n_fft: 256,
            hop_length: 80,   // 10ms at 8kHz
            win_length: 200,  // 25ms at 8kHz
            f_min: 0.0,
            f_max: 4000.0,
        }
    }

    /// Frame duration in milliseconds.
    pub fn frame_ms(&self) -> f32 {
        self.hop_length as f32 / self.sample_rate as f32 * 1000.0
    }
}

impl Default for MfccConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mfcc: 26,
            n_mels: 40,
            n_fft: 512,
            hop_length: 160,  // 10ms at 16kHz
            win_length: 400,  // 25ms at 16kHz
            f_min: 0.0,
            f_max: 8000.0,
        }
    }
}

/// MFCC feature extractor.
pub struct MfccExtractor {
    config: MfccConfig,
    mel_filterbank: Vec<Vec<f32>>,
    dct_matrix: Vec<Vec<f32>>,
    window: Vec<f32>,
}

impl MfccExtractor {
    /// Create a new MFCC extractor with the given configuration.
    pub fn new(config: MfccConfig) -> Self {
        let mel_filterbank = create_mel_filterbank(&config);
        let dct_matrix = create_dct_matrix(config.n_mfcc, config.n_mels);
        let window = create_hann_window(config.win_length);

        Self {
            config,
            mel_filterbank,
            dct_matrix,
            window,
        }
    }

    /// Extract MFCC features from audio samples.
    ///
    /// Returns a vector of frames, each containing `n_mfcc` coefficients.
    pub fn extract(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        let mut frames = Vec::new();

        // Pad audio.
        let padded_len = audio.len() + self.config.n_fft;
        let mut padded = vec![0.0f32; padded_len];
        padded[..audio.len()].copy_from_slice(audio);

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.config.n_fft);

        let mut pos = 0;
        while pos + self.config.win_length <= padded.len() {
            // Window the frame.
            let mut frame: Vec<Complex<f32>> = (0..self.config.n_fft)
                .map(|i| {
                    if i < self.config.win_length {
                        Complex::new(padded[pos + i] * self.window[i], 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    }
                })
                .collect();

            // FFT.
            fft.process(&mut frame);

            // Power spectrum (positive frequencies only).
            let power_spectrum: Vec<f32> = frame[..self.config.n_fft / 2 + 1]
                .iter()
                .map(|c| c.norm_sqr())
                .collect();

            // Apply mel filterbank.
            let mel_energies: Vec<f32> = self
                .mel_filterbank
                .iter()
                .map(|filter| {
                    let energy: f32 = filter
                        .iter()
                        .zip(power_spectrum.iter())
                        .map(|(f, p)| f * p)
                        .sum();
                    (energy + 1e-10).ln()
                })
                .collect();

            // DCT to get MFCCs.
            let mfcc: Vec<f32> = self
                .dct_matrix
                .iter()
                .map(|row| row.iter().zip(mel_energies.iter()).map(|(d, m)| d * m).sum())
                .collect();

            frames.push(mfcc);
            pos += self.config.hop_length;
        }

        frames
    }

    /// Get the configuration.
    pub fn config(&self) -> &MfccConfig {
        &self.config
    }

    /// Get the DCT matrix (for inverse DCT in formant estimation).
    pub fn dct_matrix(&self) -> &[Vec<f32>] {
        &self.dct_matrix
    }

    /// Get mel filterbank center frequencies in Hz.
    pub fn mel_centers_hz(&self) -> Vec<f32> {
        let mel_min = hz_to_mel(self.config.f_min);
        let mel_max = hz_to_mel(self.config.f_max);
        (0..self.config.n_mels)
            .map(|i| {
                let mel =
                    mel_min + (mel_max - mel_min) * (i as f32 + 0.5) / self.config.n_mels as f32;
                mel_to_hz(mel)
            })
            .collect()
    }
}

fn create_mel_filterbank(config: &MfccConfig) -> Vec<Vec<f32>> {
    let n_fft_bins = config.n_fft / 2 + 1;
    let mel_min = hz_to_mel(config.f_min);
    let mel_max = hz_to_mel(config.f_max);

    let mel_points: Vec<f32> = (0..config.n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (config.n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((config.n_fft as f32 + 1.0) * hz / config.sample_rate as f32).floor() as usize)
        .collect();

    let mut filterbank = Vec::with_capacity(config.n_mels);
    for m in 0..config.n_mels {
        let mut filter = vec![0.0f32; n_fft_bins];
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        if center > left {
            for k in left..center {
                if k < n_fft_bins {
                    filter[k] = (k - left) as f32 / (center - left) as f32;
                }
            }
        }
        if right > center {
            for k in center..right {
                if k < n_fft_bins {
                    filter[k] = (right - k) as f32 / (right - center) as f32;
                }
            }
        }
        filterbank.push(filter);
    }
    filterbank
}

fn create_dct_matrix(n_mfcc: usize, n_mels: usize) -> Vec<Vec<f32>> {
    (0..n_mfcc)
        .map(|k| {
            (0..n_mels)
                .map(|n| {
                    (PI * k as f32 * (2.0 * n as f32 + 1.0) / (2.0 * n_mels as f32)).cos()
                        * (2.0 / n_mels as f32).sqrt()
                })
                .collect()
        })
        .collect()
}

fn create_hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (length - 1) as f32).cos()))
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_sine_wave() {
        let config = MfccConfig::default();
        let extractor = MfccExtractor::new(config);

        // 100ms of 440Hz sine wave at 16kHz.
        let audio: Vec<f32> = (0..1600)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let mfcc = extractor.extract(&audio);
        assert!(!mfcc.is_empty(), "should produce frames");
        assert_eq!(mfcc[0].len(), 26, "should have 26 coefficients");
    }

    #[test]
    fn phone_quality_config() {
        let config = MfccConfig::phone_quality();
        assert_eq!(config.sample_rate, 8000);
        assert_eq!(config.n_fft, 256);
        assert!((config.frame_ms() - 10.0).abs() < 0.1);
    }

    #[test]
    fn wideband_config() {
        let config = MfccConfig::wideband();
        assert_eq!(config.sample_rate, 16000);
        assert!((config.frame_ms() - 10.0).abs() < 0.1);
    }

    #[test]
    fn mel_hz_roundtrip() {
        for hz in [100.0, 500.0, 1000.0, 4000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 0.1);
        }
    }
}
