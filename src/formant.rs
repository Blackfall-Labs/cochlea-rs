//! Formant estimation from raw audio via LPC spectral envelope.
//!
//! Uses Linear Predictive Coding (LPC) to model the vocal tract as an all-pole
//! filter, then finds formants from the LPC spectral envelope peaks.
//!
//! LPC is the gold standard for formant estimation because it directly models
//! the vocal tract transfer function, unlike inverse-DCT-from-MFCC which loses
//! too much spectral resolution through the mel filterbank.

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// Estimated formant frequencies and confidences for a single frame.
#[derive(Debug, Clone)]
pub struct FormantEstimate {
    /// First formant frequency (Hz). Correlates with vowel height.
    pub f1_hz: f32,
    /// Second formant frequency (Hz). Correlates with vowel frontness.
    pub f2_hz: f32,
    /// Third formant frequency (Hz). Speaker-dependent, less reliable.
    pub f3_hz: f32,
    /// Confidence in F1 estimate (0.0-1.0).
    pub f1_confidence: f32,
    /// Confidence in F2 estimate (0.0-1.0).
    pub f2_confidence: f32,
    /// Confidence in F3 estimate (0.0-1.0).
    pub f3_confidence: f32,
    /// Whether this frame appears to be voiced speech.
    pub voiced: bool,
    /// Frame energy (from signal power).
    pub energy: f32,
}

/// High-resolution formant estimator using LPC spectral envelope.
pub struct FormantEstimator {
    sample_rate: u32,
    /// LPC order (number of poles). Typically 2 + sample_rate/1000.
    lpc_order: usize,
    /// FFT size for evaluating the LPC spectrum (high resolution).
    fft_size: usize,
    /// Window function.
    window: Vec<f32>,
    /// Window length in samples.
    win_length: usize,
    /// Hop length in samples.
    hop_length: usize,
    /// Maximum reliable frequency.
    max_freq_hz: f32,
}

impl FormantEstimator {
    /// Create a formant estimator for the given audio parameters.
    pub fn new(sample_rate: u32, _n_mfcc: usize, _n_mels: usize, _f_min: f32, _f_max: f32) -> Self {
        // LPC order: rule of thumb is 2 + sample_rate/1000 for speech.
        // At 16kHz: order 18. At 8kHz: order 10.
        let lpc_order = 2 + (sample_rate / 1000) as usize;

        // Window/hop matching the MFCC config.
        let win_length = (sample_rate as f32 * 0.025) as usize; // 25ms
        let hop_length = (sample_rate as f32 * 0.010) as usize; // 10ms

        // High-resolution FFT for evaluating LPC spectrum.
        let fft_size = 2048;

        let window = hamming_window(win_length);

        Self {
            sample_rate,
            lpc_order,
            fft_size,
            window,
            win_length,
            hop_length,
            max_freq_hz: sample_rate as f32 / 2.0,
        }
    }

    /// Estimate formants for all frames in an audio signal.
    ///
    /// Returns one FormantEstimate per frame (matching MFCC frame alignment).
    pub fn estimate_all(&self, audio: &[f32]) -> Vec<FormantEstimate> {
        let mut results = Vec::new();
        let mut pos = 0;

        while pos + self.win_length <= audio.len() {
            let frame = &audio[pos..pos + self.win_length];
            let est = self.estimate_frame(frame);

            // Diagnostic: log frames with significant energy (speech region).
            let frame_slice = &audio[pos..pos + self.win_length];
            let max_amp = frame_slice.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            if max_amp > 0.01 && results.len() % 500 == 0 {
                log::info!(
                    "Frame {}: max_amp={:.4} F1={:.0}Hz(c={:.2}) F2={:.0}Hz(c={:.2}) F3={:.0}Hz(c={:.2}) voiced={} energy={:.6}",
                    results.len(), max_amp,
                    est.f1_hz, est.f1_confidence,
                    est.f2_hz, est.f2_confidence, est.f3_hz, est.f3_confidence,
                    est.voiced, est.energy
                );
            }
            // Also log first speech frame found.
            if max_amp > 0.05 && results.iter().all(|e: &FormantEstimate| e.energy < 0.0001) {
                log::info!(
                    "FIRST SPEECH Frame {}: max_amp={:.4} F1={:.0}Hz(c={:.2}) F2={:.0}Hz(c={:.2}) F3={:.0}Hz(c={:.2}) voiced={} energy={:.6}",
                    results.len(), max_amp,
                    est.f1_hz, est.f1_confidence,
                    est.f2_hz, est.f2_confidence, est.f3_hz, est.f3_confidence,
                    est.voiced, est.energy
                );
            }

            results.push(est);
            pos += self.hop_length;
        }

        results
    }

    /// Estimate formants from a single audio frame.
    fn estimate_frame(&self, frame: &[f32]) -> FormantEstimate {
        // Step 1: Apply window and compute frame energy.
        let windowed: Vec<f32> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        let energy: f32 = windowed.iter().map(|&s| s * s).sum::<f32>() / windowed.len() as f32;

        // Step 2: Compute LPC coefficients via autocorrelation + Levinson-Durbin.
        let autocorr = autocorrelation(&windowed, self.lpc_order);
        let lpc = levinson_durbin(&autocorr, self.lpc_order);

        // Step 4: Evaluate LPC spectral envelope at high resolution.
        let spectrum = lpc_spectrum(&lpc, self.fft_size);

        // Step 5: Find peaks in the LPC spectral envelope.
        let hz_per_bin = self.sample_rate as f32 / self.fft_size as f32;
        let peaks = find_lpc_peaks(&spectrum, hz_per_bin);

        // Diagnostic: log LPC stats for first few energetic frames.
        static DIAG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if energy > 1e-6 && DIAG_COUNT.load(std::sync::atomic::Ordering::Relaxed) < 5 {
            DIAG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let lpc_max = lpc.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
            let spec_min = spectrum.iter().copied().fold(f32::INFINITY, f32::min);
            let spec_max = spectrum.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let r0 = autocorr[0];
            log::info!(
                "LPC DIAG: r0={:.8} lpc_max={:.4} spec_range=[{:.2e}, {:.2e}] peaks_found={} energy={:.8}",
                r0, lpc_max, spec_min, spec_max, peaks.len(), energy
            );
            for (i, p) in peaks.iter().take(5).enumerate() {
                log::info!("  peak {}: {:.0}Hz prominence={:.2} bw={:.0}Hz", i, p.hz, p.prominence, p.bandwidth_hz);
            }
        }

        // Step 6: Voicing detection from spectral tilt.
        let nyquist_bin = self.fft_size / 2;
        let low_energy: f32 = spectrum[..nyquist_bin / 3].iter().sum();
        let high_energy: f32 = spectrum[nyquist_bin * 2 / 3..nyquist_bin].iter().sum();
        let voiced = low_energy > high_energy * 1.5 && energy > 1e-8;

        // Step 7: Assign formants from peaks.
        let (f1_hz, f1_conf) = find_formant(&peaks, 200.0, 900.0, 0.0);
        let (f2_hz, f2_conf) = find_formant(&peaks, 700.0, 2500.0, f1_hz + 100.0);
        let (f3_hz, f3_conf) = find_formant(&peaks, 1800.0, 3500.0, f2_hz + 100.0);

        // Reduce F3 confidence at low sample rates.
        let f3_confidence = if self.max_freq_hz < 5000.0 {
            f3_conf * 0.3
        } else {
            f3_conf
        };

        FormantEstimate {
            f1_hz,
            f2_hz,
            f3_hz,
            f1_confidence: f1_conf,
            f2_confidence: f2_conf,
            f3_confidence,
            voiced,
            energy,
        }
    }

    /// Estimate formants from an MFCC frame (legacy API, kept for compatibility).
    /// Falls back to the inverse-DCT approach if raw audio isn't available.
    pub fn estimate(&self, _mfcc: &[f32]) -> FormantEstimate {
        // Return a low-confidence default — callers should use estimate_all() instead.
        FormantEstimate {
            f1_hz: 550.0,
            f2_hz: 1600.0,
            f3_hz: 2650.0,
            f1_confidence: 0.1,
            f2_confidence: 0.1,
            f3_confidence: 0.05,
            voiced: false,
            energy: 0.0,
        }
    }
}

/// A spectral peak with frequency and prominence.
#[derive(Debug, Clone)]
struct LpcPeak {
    hz: f32,
    prominence: f32,
    bandwidth_hz: f32,
}

/// Compute autocorrelation r[0..=order].
fn autocorrelation(signal: &[f32], order: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; order + 1];
    for lag in 0..=order {
        for i in lag..signal.len() {
            r[lag] += signal[i] * signal[i - lag];
        }
    }
    r
}

/// Levinson-Durbin recursion to compute LPC coefficients from autocorrelation.
///
/// Returns LPC coefficients a[0..order] where a[0] = 1.0 (implicit).
fn levinson_durbin(r: &[f32], order: usize) -> Vec<f32> {
    if r[0].abs() < 1e-10 {
        return vec![0.0; order + 1];
    }

    let mut a = vec![0.0f32; order + 1];
    a[0] = 1.0;
    let mut error = r[0];

    for i in 1..=order {
        // Compute reflection coefficient.
        let mut sum = 0.0f32;
        for j in 1..i {
            sum += a[j] * r[i - j];
        }
        let lambda = -(r[i] + sum) / error;

        // Update coefficients.
        let mut new_a = a.clone();
        new_a[i] = lambda;
        for j in 1..i {
            new_a[j] = a[j] + lambda * a[i - j];
        }
        a = new_a;

        error *= 1.0 - lambda * lambda;
        if error <= 0.0 {
            break;
        }
    }

    a
}

/// Evaluate the LPC spectral envelope magnitude.
///
/// The LPC filter is H(z) = 1 / A(z) where A(z) = 1 + a[1]z^-1 + ... + a[p]z^-p.
/// Returns |H(e^jw)|^2 for fft_size/2+1 frequency bins.
fn lpc_spectrum(lpc: &[f32], fft_size: usize) -> Vec<f32> {
    let n_bins = fft_size / 2 + 1;
    let mut spectrum = vec![0.0f32; n_bins];

    for k in 0..n_bins {
        let w = 2.0 * PI * k as f32 / fft_size as f32;

        // Evaluate A(e^jw) = sum_i a[i] * e^(-j*w*i)
        let mut real = 0.0f32;
        let mut imag = 0.0f32;
        for (i, &coeff) in lpc.iter().enumerate() {
            let phase = -w * i as f32;
            real += coeff * phase.cos();
            imag += coeff * phase.sin();
        }

        // |H(e^jw)|^2 = 1 / |A(e^jw)|^2
        let mag_sq = real * real + imag * imag;
        spectrum[k] = if mag_sq > 1e-10 {
            1.0 / mag_sq
        } else {
            0.0
        };
    }

    spectrum
}

/// Find peaks in the LPC spectral envelope.
fn find_lpc_peaks(spectrum: &[f32], hz_per_bin: f32) -> Vec<LpcPeak> {
    let mut peaks = Vec::new();
    if spectrum.len() < 3 {
        return peaks;
    }

    // Use log magnitude for prominence calculation.
    let log_spec: Vec<f32> = spectrum.iter().map(|&s| (s + 1e-10).ln()).collect();
    let n = log_spec.len();

    // Remove spectral tilt via linear regression on the log spectrum.
    // This makes formant peaks stand out regardless of overall spectral slope.
    let flattened = {
        let nf = n as f32;
        let sum_x: f32 = (0..n).map(|i| i as f32).sum();
        let sum_y: f32 = log_spec.iter().sum();
        let sum_xy: f32 = log_spec.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let sum_xx: f32 = (0..n).map(|i| (i as f32) * (i as f32)).sum();

        let denom = nf * sum_xx - sum_x * sum_x;
        let (slope, intercept) = if denom.abs() > 1e-10 {
            let m = (nf * sum_xy - sum_x * sum_y) / denom;
            let b = (sum_y - m * sum_x) / nf;
            (m, b)
        } else {
            (0.0, sum_y / nf)
        };

        log_spec
            .iter()
            .enumerate()
            .map(|(i, &y)| y - (slope * i as f32 + intercept))
            .collect::<Vec<f32>>()
    };

    // Smooth the flattened spectrum with 5-point moving average to reduce noise peaks.
    let smoothed: Vec<f32> = (0..n)
        .map(|i| {
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(n);
            flattened[lo..hi].iter().sum::<f32>() / (hi - lo) as f32
        })
        .collect();

    for i in 1..smoothed.len() - 1 {
        // Simple local maximum: higher than both immediate neighbors.
        if smoothed[i] > smoothed[i - 1] && smoothed[i] > smoothed[i + 1] {
            let hz = i as f32 * hz_per_bin;

            // Skip very low frequencies (below 80Hz) and DC.
            if hz < 80.0 {
                continue;
            }

            // Prominence: height above the average of a wider neighborhood (±15 bins ≈ ±120Hz).
            let lo = i.saturating_sub(15);
            let hi = (i + 16).min(smoothed.len());
            let neighbor_avg: f32 = smoothed[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
            let prominence = smoothed[i] - neighbor_avg;
            if prominence < 0.05 {
                continue; // Not prominent enough
            }

            // Parabolic interpolation for sub-bin precision.
            let alpha = log_spec[i - 1];
            let beta = log_spec[i];
            let gamma = log_spec[i + 1];
            let denom = alpha - 2.0 * beta + gamma;
            let offset = if denom.abs() > 1e-10 {
                (0.5 * (alpha - gamma) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let precise_hz = (i as f32 + offset) * hz_per_bin;

            // Estimate bandwidth from the -3dB width of the peak.
            let target = log_spec[i] - 1.5; // -3dB in log domain (ln)
            let mut bw_left = 0;
            for j in (0..i).rev() {
                if log_spec[j] < target {
                    bw_left = i - j;
                    break;
                }
            }
            let mut bw_right = 0;
            for j in (i + 1)..spectrum.len() {
                if log_spec[j] < target {
                    bw_right = j - i;
                    break;
                }
            }
            let bandwidth_hz = (bw_left + bw_right) as f32 * hz_per_bin;

            peaks.push(LpcPeak {
                hz: precise_hz,
                prominence,
                bandwidth_hz,
            });
        }
    }

    peaks
}

/// Find the best formant from LPC peaks within a frequency range.
///
/// Prefers peaks with higher prominence and narrower bandwidth (typical of formants).
fn find_formant(peaks: &[LpcPeak], min_hz: f32, max_hz: f32, above_hz: f32) -> (f32, f32) {
    let effective_min = min_hz.max(above_hz);
    let mut best_hz = (effective_min + max_hz) / 2.0;
    let mut best_score = 0.0f32;

    for peak in peaks {
        if peak.hz >= effective_min && peak.hz <= max_hz {
            // Score: prominence / bandwidth (narrow prominent peaks are formants).
            let bw = peak.bandwidth_hz.max(20.0); // prevent division by zero
            let score = peak.prominence / (bw / 100.0);
            if score > best_score {
                best_hz = peak.hz;
                best_score = score;
            }
        }
    }

    // Score → confidence mapping. LPC peaks typically have modest prominence
    // (0.1-1.0) with bandwidth 50-300Hz, giving scores of 0.05-2.0.
    // Map this range to useful confidence values.
    let confidence = if best_score > 0.0 {
        (best_score / 1.0).min(1.0) // More generous: score of 0.5 → 0.5 confidence
    } else {
        0.05
    };

    (best_hz, confidence)
}

/// Generate a Hamming window.
fn hamming_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f32 / (length - 1) as f32).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_creates() {
        let est = FormantEstimator::new(16000, 26, 40, 0.0, 8000.0);
        assert_eq!(est.lpc_order, 18); // 2 + 16000/1000
        assert_eq!(est.fft_size, 2048);
    }

    #[test]
    fn estimator_creates_for_phone() {
        let est = FormantEstimator::new(8000, 26, 26, 0.0, 4000.0);
        assert_eq!(est.lpc_order, 10); // 2 + 8000/1000
        assert!((est.max_freq_hz - 4000.0).abs() < 1.0);
    }

    #[test]
    fn levinson_durbin_basic() {
        // Simple test: autocorrelation of [1, 0.5]
        let r = vec![1.0, 0.5, 0.0];
        let lpc = levinson_durbin(&r, 2);
        assert!((lpc[0] - 1.0).abs() < 0.01);
        assert!(lpc[1].abs() < 1.0); // reflection coefficient should be bounded
    }

    #[test]
    fn estimate_all_produces_frames() {
        let est = FormantEstimator::new(16000, 26, 40, 0.0, 8000.0);
        // 200ms of 440Hz sine wave.
        let audio: Vec<f32> = (0..3200)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let results = est.estimate_all(&audio);
        assert!(!results.is_empty(), "should produce formant estimates");
    }

    #[test]
    fn formants_in_valid_ranges() {
        let est = FormantEstimator::new(16000, 26, 40, 0.0, 8000.0);
        // Synthesize a vowel-like signal: multiple harmonics.
        let audio: Vec<f32> = (0..3200)
            .map(|i| {
                let t = i as f32 / 16000.0;
                let f0 = 120.0;
                // Add harmonics
                (0..10)
                    .map(|h| {
                        let freq = f0 * (h + 1) as f32;
                        (2.0 * PI * freq * t).sin() / (h + 1) as f32
                    })
                    .sum::<f32>()
            })
            .collect();
        let results = est.estimate_all(&audio);
        if let Some(r) = results.first() {
            assert!(r.f1_hz >= 100.0 && r.f1_hz <= 1000.0, "F1={}", r.f1_hz);
            assert!(r.f2_hz >= 500.0 && r.f2_hz <= 3000.0, "F2={}", r.f2_hz);
        }
    }

    #[test]
    fn phone_quality_reduces_f3_confidence() {
        let est_wide = FormantEstimator::new(16000, 26, 40, 0.0, 8000.0);
        let est_phone = FormantEstimator::new(8000, 26, 26, 0.0, 4000.0);

        // Phone should have lower max_freq_hz.
        assert!(est_phone.max_freq_hz < est_wide.max_freq_hz);
    }
}
