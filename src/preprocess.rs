//! Audio preprocessing for noise-robust transcription.
//!
//! Applies conditioning steps that make noisy/phone-quality audio survivable
//! before MFCC extraction and formant estimation.

/// Apply pre-emphasis filter to boost high frequencies.
///
/// Phone codecs and microphones attenuate high-frequency energy.
/// Pre-emphasis compensates: y[n] = x[n] - coeff * x[n-1].
/// Typical coefficient: 0.97.
pub fn pre_emphasis(samples: &mut [f32], coeff: f32) {
    if samples.len() < 2 || coeff == 0.0 {
        return;
    }
    // Process in reverse to avoid overwriting needed values.
    for i in (1..samples.len()).rev() {
        samples[i] -= coeff * samples[i - 1];
    }
    samples[0] *= 1.0 - coeff;
}

/// Cepstral Mean Normalization — subtract per-coefficient mean across all frames.
///
/// Removes channel effects (phone microphone coloring, codec EQ).
/// Critical for phone-quality audio where the recording chain adds a
/// constant spectral bias to every frame.
pub fn cepstral_mean_normalization(frames: &mut [Vec<f32>]) {
    if frames.is_empty() {
        return;
    }
    let n_coeffs = frames[0].len();
    let n_frames = frames.len() as f32;

    // Compute per-coefficient mean.
    let mut means = vec![0.0f32; n_coeffs];
    for frame in frames.iter() {
        for (i, &val) in frame.iter().enumerate() {
            if i < n_coeffs {
                means[i] += val;
            }
        }
    }
    for mean in &mut means {
        *mean /= n_frames;
    }

    // Subtract mean from each frame.
    for frame in frames.iter_mut() {
        for (i, val) in frame.iter_mut().enumerate() {
            if i < n_coeffs {
                *val -= means[i];
            }
        }
    }
}

/// Voice Activity Detection using the c0 (energy) coefficient.
///
/// Returns a boolean per frame: `true` = speech, `false` = silence/noise.
/// The threshold is in log-energy space (c0 is approximately log total energy).
pub fn energy_vad(frames: &[Vec<f32>], threshold: f32) -> Vec<bool> {
    frames
        .iter()
        .map(|frame| {
            // c0 is the zeroth MFCC coefficient (log energy).
            let energy = frame.first().copied().unwrap_or(f32::NEG_INFINITY);
            energy > threshold
        })
        .collect()
}

/// Estimate noise spectrum from leading silence frames.
///
/// Returns the mean MFCC vector of the first `n_frames` frames,
/// which can be subtracted from all frames for simple spectral subtraction.
pub fn estimate_noise_profile(frames: &[Vec<f32>], n_frames: usize) -> Vec<f32> {
    if frames.is_empty() {
        return Vec::new();
    }
    let n = n_frames.min(frames.len());
    let n_coeffs = frames[0].len();
    let mut profile = vec![0.0f32; n_coeffs];

    for frame in &frames[..n] {
        for (i, &val) in frame.iter().enumerate() {
            if i < n_coeffs {
                profile[i] += val;
            }
        }
    }

    let n_f32 = n as f32;
    for val in &mut profile {
        *val /= n_f32;
    }
    profile
}

/// Subtract noise profile from MFCC frames (simple spectral subtraction in cepstral domain).
///
/// Subtracts the estimated noise spectrum from each frame. Values are floored
/// at a minimum to prevent negative energy artifacts.
pub fn spectral_subtraction(frames: &mut [Vec<f32>], noise_profile: &[f32]) {
    if noise_profile.is_empty() {
        return;
    }
    for frame in frames.iter_mut() {
        for (i, val) in frame.iter_mut().enumerate() {
            if let Some(&noise) = noise_profile.get(i) {
                *val -= noise;
                // Floor at a small negative value to prevent artifacts.
                if *val < -50.0 {
                    *val = -50.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_emphasis_boosts_high_freq() {
        // A constant signal should become near-zero after pre-emphasis
        // (constant = zero frequency = removed).
        let mut samples = vec![1.0f32; 100];
        pre_emphasis(&mut samples, 0.97);

        // First sample is special.
        assert!((samples[0] - 0.03).abs() < 0.01);
        // Subsequent samples should be near zero (1.0 - 0.97*1.0 = 0.03).
        for &s in &samples[1..] {
            assert!(s.abs() < 0.04, "constant signal should be suppressed: {}", s);
        }
    }

    #[test]
    fn pre_emphasis_passes_impulse() {
        // An impulse at t=0 should mostly survive.
        let mut samples = vec![0.0f32; 100];
        samples[0] = 1.0;
        pre_emphasis(&mut samples, 0.97);

        assert!((samples[0] - 0.03).abs() < 0.01);
        assert!((samples[1] - (-0.97)).abs() < 0.01);
    }

    #[test]
    fn cmn_zeros_mean() {
        let mut frames = vec![
            vec![10.0, 20.0, 30.0],
            vec![12.0, 22.0, 32.0],
            vec![8.0, 18.0, 28.0],
        ];
        cepstral_mean_normalization(&mut frames);

        // Mean of each coefficient should now be ~0.
        let n = frames.len() as f32;
        for coeff in 0..3 {
            let mean: f32 = frames.iter().map(|f| f[coeff]).sum::<f32>() / n;
            assert!(
                mean.abs() < 1e-5,
                "coefficient {} mean should be zero, got {}",
                coeff,
                mean
            );
        }
    }

    #[test]
    fn cmn_empty_is_noop() {
        let mut frames: Vec<Vec<f32>> = Vec::new();
        cepstral_mean_normalization(&mut frames);
        assert!(frames.is_empty());
    }

    #[test]
    fn vad_detects_speech() {
        let frames = vec![
            vec![-40.0, 0.0], // silence
            vec![-35.0, 0.0], // silence
            vec![-10.0, 0.0], // speech
            vec![-5.0, 0.0],  // speech
            vec![-38.0, 0.0], // silence
        ];
        let vad = energy_vad(&frames, -30.0);
        assert_eq!(vad, vec![false, false, true, true, false]);
    }

    #[test]
    fn noise_profile_averages_leading_frames() {
        let frames = vec![
            vec![2.0, 4.0],
            vec![4.0, 6.0],
            vec![100.0, 200.0], // speech — should not be included
        ];
        let profile = estimate_noise_profile(&frames, 2);
        assert!((profile[0] - 3.0).abs() < 1e-5);
        assert!((profile[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn spectral_subtraction_removes_noise() {
        let mut frames = vec![
            vec![10.0, 20.0],
            vec![12.0, 22.0],
        ];
        let noise = vec![3.0, 5.0];
        spectral_subtraction(&mut frames, &noise);
        assert!((frames[0][0] - 7.0).abs() < 1e-5);
        assert!((frames[0][1] - 15.0).abs() < 1e-5);
        assert!((frames[1][0] - 9.0).abs() < 1e-5);
    }
}
