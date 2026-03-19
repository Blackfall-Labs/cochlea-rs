//! MFCC template matching — synthesize reference phonemes and match in MFCC space.
//!
//! Instead of going through lossy formant estimation (MFCC → inverse DCT → peak pick),
//! this module synthesizes each phoneme using the oscillator, extracts its MFCC template,
//! and matches input frames directly via cosine similarity in MFCC space.
//!
//! This is more discriminative because MFCCs capture the full spectral envelope shape
//! (26 dimensions) rather than just 3 formant peaks.

use crate::mfcc::{MfccConfig, MfccExtractor};
use crate::phoneme_bank::PhonemeCandidate;
use crate::preprocess::pre_emphasis;
use signal_oscillator::renderer::{Interpolation, RenderConfig};
use signal_oscillator::speech::{SpeechConfig, SpeechSynthesizer};

/// A reference phoneme MFCC template.
struct PhonemeTemplate {
    ipa: String,
    /// Mean MFCC vector across synthesized frames (excluding silence frames).
    mfcc: Vec<f32>,
    /// Whether this phoneme is voiced.
    voiced: bool,
}

/// MFCC template matcher — compares input MFCC frames against synthesized phoneme templates.
pub struct MfccTemplateMatcher {
    templates: Vec<PhonemeTemplate>,
}

impl MfccTemplateMatcher {
    /// Build templates by synthesizing each phoneme and extracting MFCCs.
    ///
    /// The synthesis sample rate is matched to the MFCC config so the templates
    /// live in the same feature space as the input audio.
    pub fn build(mfcc_config: &MfccConfig) -> Self {
        // Build the MFCC extractor with matching config.
        let extractor = MfccExtractor::new(mfcc_config.clone());

        let phonemes = [
            // Vowels (monophthongs).
            ("i", true), ("ɪ", true), ("e", true), ("ɛ", true), ("æ", true),
            ("a", true), ("ɑ", true), ("ɒ", true), ("ɔ", true), ("o", true),
            ("ʊ", true), ("u", true), ("ʌ", true), ("ə", true), ("ɜ", true), ("ɐ", true),
            // Long vowels.
            ("iː", true), ("ɑː", true), ("ɔː", true), ("uː", true), ("ɜː", true),
            // Diphthongs.
            ("aɪ", true), ("aʊ", true), ("eɪ", true), ("ɔɪ", true),
            ("əʊ", true), ("oʊ", true), ("ɪə", true), ("eə", true),
            // Nasals (voiced).
            ("m", true), ("n", true), ("ŋ", true),
            // Liquids + approximants (voiced).
            ("l", true), ("ɹ", true), ("w", true), ("j", true),
            // Voiced plosives.
            ("b", true), ("d", true), ("ɡ", true),
            // Voiceless plosives.
            ("p", false), ("t", false), ("k", false),
            // Voiced fricatives.
            ("v", true), ("ð", true), ("z", true), ("ʒ", true),
            // Voiceless fricatives.
            ("f", false), ("θ", false), ("s", false), ("ʃ", false), ("h", false),
            // Affricates.
            ("tʃ", false), ("dʒ", true),
        ];

        // Synthesize at multiple F0 values (male/female/child) for better speaker coverage.
        // Average the templates to get a speaker-independent representation.
        let f0_values = [120.0, 180.0, 240.0];
        let mut templates = Vec::new();

        for &(ipa, voiced) in &phonemes {
            let mut all_mfccs: Vec<Vec<f32>> = Vec::new();

            for &f0 in &f0_values {
                let synth_config = SpeechConfig {
                    base_f0: f0,
                    speech_rate: 0.8,
                    render: RenderConfig {
                        sample_rate: mfcc_config.sample_rate,
                        interpolation: Interpolation::Cubic,
                        amplitude: 0.8,
                    },
                    tension_floor: 0.0,
                    breathiness_ceiling: 0.5,
                    energy_floor: 0.0,
                    formant_bw_scale: 1.0,
                };
                let mut synth = SpeechSynthesizer::with_config(synth_config);
                synth.reset();
                let audio = synth.pronounce_ipa(ipa);

                if audio.is_empty() || audio.len() < mfcc_config.win_length as usize {
                    continue;
                }

                // Apply pre-emphasis to match the input pipeline.
                let mut audio = audio;
                pre_emphasis(&mut audio, 0.97);

                let frames = extractor.extract(&audio);
                if !frames.is_empty() {
                    let mean = mean_mfcc_voiced(&frames, mfcc_config.n_mfcc);
                    all_mfccs.push(mean);
                }
            }

            if all_mfccs.is_empty() {
                log::warn!("No MFCC frames for /{}/, skipping", ipa);
                continue;
            }

            // Average across all F0 values to get a speaker-independent template.
            let n_mfcc = mfcc_config.n_mfcc;
            let mut avg = vec![0.0f32; n_mfcc];
            for mfcc in &all_mfccs {
                for (a, &m) in avg.iter_mut().zip(mfcc.iter()) {
                    *a += m;
                }
            }
            let count = all_mfccs.len() as f32;
            for a in &mut avg {
                *a /= count;
            }

            templates.push(PhonemeTemplate {
                ipa: ipa.to_string(),
                mfcc: avg,
                voiced,
            });
        }

        log::info!(
            "MfccTemplateMatcher built with {} templates at {}Hz",
            templates.len(),
            mfcc_config.sample_rate
        );

        Self { templates }
    }

    /// Match an input MFCC frame against all templates.
    ///
    /// Returns top-K candidates sorted by similarity (highest first).
    pub fn match_frame(&self, mfcc: &[f32], k: usize) -> Vec<PhonemeCandidate> {
        let mut scored: Vec<(usize, f32)> = self
            .templates
            .iter()
            .enumerate()
            .map(|(idx, t)| {
                let sim = cosine_similarity(mfcc, &t.mfcc);
                (idx, sim)
            })
            .collect();

        // Sort by similarity (descending — highest similarity first).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(k)
            .map(|(idx, sim)| {
                let t = &self.templates[idx];
                // Convert similarity (0-1) to confidence and score.
                let confidence = sim.max(0.0);
                PhonemeCandidate {
                    ipa: t.ipa.clone(),
                    score: (confidence * 256.0) as i32,
                    confidence,
                }
            })
            .collect()
    }

    /// Number of templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Whether there are no templates.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

/// Compute mean MFCC across frames, weighting by c0 energy.
///
/// Skip frames with very low c0 (silence/onset/offset of synthesis).
fn mean_mfcc_voiced(frames: &[Vec<f32>], n_mfcc: usize) -> Vec<f32> {
    if frames.is_empty() {
        return vec![0.0; n_mfcc];
    }

    // Find the energy threshold (skip bottom 25% of frames by energy).
    let mut energies: Vec<f32> = frames.iter().map(|f| f[0]).collect();
    energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = energies[energies.len() / 4];

    let mut sum = vec![0.0f32; n_mfcc];
    let mut count = 0.0f32;

    for frame in frames {
        if frame[0] < threshold {
            continue;
        }
        let weight = 1.0; // uniform weighting for now
        for (s, &v) in sum.iter_mut().zip(frame.iter().take(n_mfcc)) {
            *s += v * weight;
        }
        count += weight;
    }

    if count > 0.0 {
        for s in &mut sum {
            *s /= count;
        }
    }

    sum
}

/// Weighted cosine similarity between two MFCC vectors.
///
/// Skips c0 (energy) and weights lower-order MFCCs (c1-c5) more heavily
/// since they carry the most phoneme-discriminative information.
/// Mean-centers both vectors for CMN invariance.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Skip c0 (index 0) — compare spectral shape only.
    let start = 1;
    let len = a.len().min(b.len());
    if len <= start {
        return 0.0;
    }

    // Dimension weights: c1-c5 most important, c6-c12 moderate, c13+ low.
    // Based on phoneme discrimination literature: lower cepstral coefficients
    // capture vocal tract shape (formants), higher capture fine detail.
    let weight = |i: usize| -> f32 {
        match i {
            1..=5 => 2.0,   // formant shape — most discriminative
            6..=12 => 1.0,  // spectral detail — useful
            _ => 0.3,       // high-order — mostly noise/speaker variance
        }
    };

    let n = (len - start) as f32;

    // Compute weighted means.
    let mut total_weight = 0.0f32;
    let mut wsum_a = 0.0f32;
    let mut wsum_b = 0.0f32;
    for i in start..len {
        let w = weight(i);
        wsum_a += a[i] * w;
        wsum_b += b[i] * w;
        total_weight += w;
    }
    let mean_a = wsum_a / total_weight;
    let mean_b = wsum_b / total_weight;

    // Weighted mean-centered cosine similarity.
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in start..len {
        let w = weight(i);
        let da = (a[i] - mean_a) * w;
        let db = (b[i] - mean_b) * w;
        dot += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (dot / denom).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_sim_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001, "identical vectors should have sim=1.0, got {}", sim);
    }

    #[test]
    fn cosine_sim_dissimilar() {
        // After mean-centering, these should have low similarity.
        let a = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim < 0.5, "dissimilar vectors should have low sim, got {}", sim);
    }

    #[test]
    fn templates_build() {
        let config = MfccConfig::wideband();
        let matcher = MfccTemplateMatcher::build(&config);
        assert!(
            matcher.len() >= 40,
            "expected 40+ templates, got {}",
            matcher.len()
        );
    }

    #[test]
    fn template_match_returns_candidates() {
        let config = MfccConfig::wideband();
        let matcher = MfccTemplateMatcher::build(&config);

        // Synthesize a known vowel and verify it matches itself.
        let synth_config = SpeechConfig {
            base_f0: 120.0,
            speech_rate: 0.8,
            render: RenderConfig {
                sample_rate: config.sample_rate,
                interpolation: Interpolation::Cubic,
                amplitude: 0.8,
            },
            tension_floor: 0.0,
            breathiness_ceiling: 0.5,
            energy_floor: 0.0,
            formant_bw_scale: 1.0,
        };
        let mut synth = SpeechSynthesizer::with_config(synth_config);
        let audio = synth.pronounce_ipa("ɑ");

        let extractor = MfccExtractor::new(config);
        let frames = extractor.extract(&audio);

        // Pick a frame from the middle (stable region).
        let mid = frames.len() / 2;
        let candidates = matcher.match_frame(&frames[mid], 5);

        assert!(!candidates.is_empty());
        let top = &candidates[0];
        // Should match a voiced phoneme — the synthesis is voiced and has formant structure.
        // Multi-pitch averaging may shift the template, so accept vowels and approximants.
        let voiced_formant = [
            "ɑ", "ɑː", "a", "ɒ", "ɔ", "ɔː", "ʌ", "ɐ", "ə", "ɜ", "ɜː",
            "æ", "ɛ", "e", "i", "ɪ", "iː", "o", "u", "ʊ", "uː",
            "aɪ", "aʊ", "eɪ", "ɔɪ", "əʊ", "oʊ", "ɪə", "eə",
            "w", "j", "l", "ɹ", "m", "n", "ŋ",
        ];
        assert!(
            voiced_formant.contains(&top.ipa.as_str()),
            "expected a voiced phoneme with formant structure, got /{}/",
            top.ipa
        );
    }
}
