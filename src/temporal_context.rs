//! Temporal context accumulation for phoneme classification.
//!
//! Bridges MFCC extraction and phoneme classification by accumulating
//! frames in a TemporalField (no-decay ring buffer) and extracting
//! temporal features: current frame, temporal mean, and delta coefficients.
//!
//! The human ear doesn't classify isolated 10ms slices — it accumulates
//! acoustic evidence over time. This module provides that temporal context.

use temporal_field::{FieldConfig, TemporalField, Signal};

/// Number of MFCC dimensions per frame.
const MFCC_DIMS: usize = 26;

/// Temporal window size in frames (70ms at 10ms/frame).
///
/// Phoneme transitions span 30-80ms. 70ms captures the current phoneme
/// plus its leading/trailing transition context without blurring across
/// phoneme boundaries (typical phoneme: 60-220ms).
const TEMPORAL_WINDOW: usize = 7;

/// Ring buffer capacity in frames (200ms at 10ms/frame).
const RING_BUFFER_FRAMES: usize = 20;

/// Total feature dimensions: current (26) + mean (26) + delta (26).
pub const TEMPORAL_FEATURE_DIM: usize = MFCC_DIMS * 3; // 78

/// Accumulates MFCC frames in a TemporalField and extracts temporal features.
///
/// Each MFCC frame is encoded as ternary Signals and written into a no-decay
/// ring buffer. Feature extraction reads a temporal window and computes:
/// - Current frame (instantaneous spectrum)
/// - Temporal mean (smoothed spectral context)
/// - Delta coefficients (rate of spectral change)
pub struct TemporalAccumulator {
    field: TemporalField,
    window_size: usize,
    frames_written: usize,
}

impl TemporalAccumulator {
    /// Create a new temporal accumulator.
    ///
    /// Uses a no-decay ring buffer: 26 dimensions, 20 frames (200ms).
    /// Signals persist without attenuation — pure temporal data tracking.
    pub fn new() -> Self {
        let config = FieldConfig::no_decay(MFCC_DIMS, RING_BUFFER_FRAMES);
        Self {
            field: TemporalField::new(config),
            window_size: TEMPORAL_WINDOW,
            frames_written: 0,
        }
    }

    /// Write an MFCC frame (f32 coefficients) into the temporal field.
    ///
    /// Encodes each f32 coefficient as a ternary Signal (polarity from sign,
    /// magnitude scaled by 12.75, multiplier 1), then writes to the field
    /// and advances the write head.
    pub fn write_frame(&mut self, mfcc: &[f32]) {
        let signals: Vec<Signal> = (0..MFCC_DIMS)
            .map(|i| {
                let val = mfcc.get(i).copied().unwrap_or(0.0);
                mfcc_to_signal(val)
            })
            .collect();

        self.field.clear_current();
        self.field.set_region(&signals, 0..MFCC_DIMS);
        self.field.advance_write_head();
        self.frames_written += 1;
    }

    /// Extract temporal features from the current window.
    ///
    /// Returns a 78-dim f32 vector: `[current(26), mean(26), delta(26)]`.
    ///
    /// - **Current**: the most recent frame's spectrum
    /// - **Mean**: averaged spectrum across the window (from `region_mean`)
    /// - **Delta**: `(newest - oldest) / window_size` — rate of spectral change
    ///
    /// If fewer than `window_size` frames have been written, uses whatever
    /// is available (graceful degradation for the first few frames).
    pub fn extract_features(&self) -> Vec<f32> {
        let effective_window = self.frames_written.min(self.window_size).max(1);
        let window = self.field.read_window(effective_window);

        let mut features = Vec::with_capacity(TEMPORAL_FEATURE_DIM);

        // Part 1: Current frame (last frame in window).
        if let Some(frame) = window.last() {
            for i in 0..MFCC_DIMS {
                features.push(signal_to_f32(frame.get(i)));
            }
        } else {
            features.extend(std::iter::repeat(0.0f32).take(MFCC_DIMS));
        }

        // Part 2: Temporal mean across the window.
        let mean_signals = self.field.region_mean(0..MFCC_DIMS, effective_window);
        for s in &mean_signals {
            features.push(signal_to_f32(*s));
        }

        // Part 3: Delta coefficients (spectral change across window).
        // Raw difference (not divided by window size) to keep on same scale as MFCC.
        if window.len() >= 2 {
            let newest = window.last().unwrap();
            let oldest = window.first().unwrap();
            for i in 0..MFCC_DIMS {
                let diff = signal_to_f32(newest.get(i)) - signal_to_f32(oldest.get(i));
                features.push(diff);
            }
        } else {
            features.extend(std::iter::repeat(0.0f32).take(MFCC_DIMS));
        }

        features
    }

    /// Reset the accumulator (clear all frames).
    pub fn reset(&mut self) {
        self.field.clear();
        self.frames_written = 0;
    }

    /// Number of frames written so far.
    pub fn frames_written(&self) -> usize {
        self.frames_written
    }
}

/// Convert an f32 MFCC coefficient to a ternary Signal.
///
/// Uses the same scaling as `phoneme_classifier::encode_mfcc`:
/// polarity from sign (dead zone ±0.01), magnitude = `|val| * 12.75`,
/// multiplier = 1.
fn mfcc_to_signal(val: f32) -> Signal {
    if val.abs() < 0.01 {
        Signal::ZERO
    } else if val > 0.0 {
        Signal::positive((val.abs() * 12.75).min(255.0) as u8)
    } else {
        Signal::negative((val.abs() * 12.75).min(255.0) as u8)
    }
}

/// Convert a Signal back to approximate f32 (inverse of mfcc_to_signal).
fn signal_to_f32(s: Signal) -> f32 {
    s.current() as f32 / 12.75
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_dimensions() {
        let mut acc = TemporalAccumulator::new();
        let frame = vec![1.0; MFCC_DIMS];

        // Even before writing, extract_features should return 78 dims.
        let features = acc.extract_features();
        assert_eq!(features.len(), TEMPORAL_FEATURE_DIM);

        // After writing one frame.
        acc.write_frame(&frame);
        let features = acc.extract_features();
        assert_eq!(features.len(), TEMPORAL_FEATURE_DIM);

        // After writing many frames.
        for _ in 0..20 {
            acc.write_frame(&frame);
        }
        let features = acc.extract_features();
        assert_eq!(features.len(), TEMPORAL_FEATURE_DIM);
    }

    #[test]
    fn delta_zero_for_constant_input() {
        let mut acc = TemporalAccumulator::new();
        let frame = vec![5.0; MFCC_DIMS];

        for _ in 0..10 {
            acc.write_frame(&frame);
        }

        let features = acc.extract_features();
        // Delta coefficients (last 26 dims) should be near zero for constant input.
        let delta = &features[2 * MFCC_DIMS..];
        for &d in delta {
            assert!(d.abs() < 1.0, "delta should be ~0 for constant input, got {}", d);
        }
    }

    #[test]
    fn delta_nonzero_for_changing_input() {
        let mut acc = TemporalAccumulator::new();

        // Write frames with increasing energy.
        for i in 0..10 {
            let frame: Vec<f32> = (0..MFCC_DIMS)
                .map(|_| i as f32 * 2.0)
                .collect();
            acc.write_frame(&frame);
        }

        let features = acc.extract_features();
        let delta = &features[2 * MFCC_DIMS..];
        // At least some delta coefficients should be nonzero.
        let nonzero = delta.iter().filter(|&&d| d.abs() > 0.01).count();
        assert!(nonzero > 0, "delta should be nonzero for changing input");
    }

    #[test]
    fn signal_roundtrip() {
        let values = [0.0, 5.0, -3.0, 0.005, 10.0, -0.5];
        for &val in &values {
            let signal = mfcc_to_signal(val);
            let recovered = signal_to_f32(signal);
            if val.abs() < 0.01 {
                assert_eq!(recovered, 0.0, "dead zone should produce zero");
            } else {
                // Sign should be preserved.
                assert_eq!(val.signum(), recovered.signum(),
                    "sign mismatch for {}: got {}", val, recovered);
            }
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut acc = TemporalAccumulator::new();
        for _ in 0..5 {
            acc.write_frame(&vec![10.0; MFCC_DIMS]);
        }
        assert_eq!(acc.frames_written(), 5);

        acc.reset();
        assert_eq!(acc.frames_written(), 0);

        let features = acc.extract_features();
        // After reset, all features should be zero.
        for &f in &features {
            assert_eq!(f, 0.0, "features should be zero after reset");
        }
    }
}
