//! Ensemble phoneme classifier — 12 acoustic group decomposition.
//!
//! Instead of one 45-class classifier with a single frozen projection,
//! decomposes into 12 small classifiers (2-8 phonemes each). Each group
//! gets its own frozen random ternary projection (unique seed, 32-dim).
//! Total feature space: 12 × 32 = 384 dims instead of 64.
//!
//! At classification time, all 12 groups run in parallel, each producing
//! a top candidate. The best candidates across all groups are returned.
//!
//! This solves the schwa-matches-everything problem: each phoneme only
//! competes within its small acoustic group.

use crate::error::Result;
use crate::mfcc::MfccConfig;
use crate::phoneme_bank::PhonemeCandidate;
use crate::temporal_context::TEMPORAL_FEATURE_DIM;
use atomic_neural_transistors::{WeightMatrix, PackedSignal};
use std::path::Path;

/// Total number of phonemes across all groups.
pub(crate) const NUM_PHONEMES: usize = 45;

/// Hidden dimension per group classifier (single-frame).
pub(crate) const GROUP_HIDDEN_DIM: usize = 32;

/// Hidden dimension per group classifier (temporal).
/// Larger than single-frame because temporal input is 78-dim vs 26-dim.
pub(crate) const TEMPORAL_HIDDEN_DIM: usize = 48;

/// Number of acoustic groups.
pub(crate) const NUM_GROUPS: usize = 12;

/// The full ordered list of phonemes (used for training data generation).
pub(crate) const PHONEME_LIST: [&str; NUM_PHONEMES] = [
    "i", "ɪ", "e", "ɛ", "æ", "a", "ɑ", "ɒ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ", "ɐ",
    "iː", "ɑː", "ɔː", "uː", "ɜː",
    "aɪ", "aʊ", "eɪ", "ɔɪ", "əʊ", "oʊ", "ɪə", "eə",
    "m", "n", "ŋ",
    "l", "ɹ", "w", "j",
    "p", "b", "t", "d", "k", "ɡ",
    "s", "ʃ", "h",
];

/// Definition of one acoustic group.
pub(crate) struct AcousticGroup {
    pub(crate) name: &'static str,
    pub(crate) phonemes: &'static [&'static str],
    pub(crate) seed: u64,
}

/// The 12 acoustic groups, split by formant structure and manner.
pub(crate) const ACOUSTIC_GROUPS: [AcousticGroup; NUM_GROUPS] = [
    AcousticGroup { name: "front_close_vowels", phonemes: &["i", "ɪ", "e"], seed: 1001 },
    AcousticGroup { name: "front_open_vowels", phonemes: &["ɛ", "æ", "a"], seed: 2003 },
    AcousticGroup { name: "central_vowels", phonemes: &["ə", "ɜ", "ɐ", "ʌ"], seed: 3007 },
    AcousticGroup { name: "back_close_vowels", phonemes: &["u", "ʊ", "o"], seed: 4013 },
    AcousticGroup { name: "back_open_vowels", phonemes: &["ɑ", "ɒ", "ɔ"], seed: 5021 },
    AcousticGroup { name: "long_vowels", phonemes: &["iː", "ɑː", "ɔː", "uː", "ɜː"], seed: 6029 },
    AcousticGroup { name: "diphthongs", phonemes: &["aɪ", "aʊ", "eɪ", "ɔɪ", "əʊ", "oʊ", "ɪə", "eə"], seed: 7039 },
    AcousticGroup { name: "nasals", phonemes: &["m", "n", "ŋ"], seed: 8053 },
    AcousticGroup { name: "approximants_liquids", phonemes: &["l", "ɹ", "w", "j"], seed: 9067 },
    AcousticGroup { name: "voiced_plosives", phonemes: &["b", "d", "ɡ"], seed: 10079 },
    AcousticGroup { name: "voiceless_plosives", phonemes: &["p", "t", "k"], seed: 11083 },
    AcousticGroup { name: "fricatives", phonemes: &["s", "ʃ", "h"], seed: 12097 },
];

/// Seed offset for temporal projections (distinct from single-frame).
pub(crate) const TEMPORAL_SEED_OFFSET: u64 = 100_000;

/// Inference cache format version (must match mastery_cache::INFERENCE_CACHE_VERSION).
const CACHE_VERSION: u32 = crate::mastery_cache::INFERENCE_CACHE_VERSION;

/// Runtime state for one acoustic group's classifier.
pub(crate) struct GroupClassifier {
    /// Frozen random projection: input_dim → GROUP_HIDDEN_DIM (single-frame).
    pub(crate) w_hidden: WeightMatrix,
    /// Frozen random projection: TEMPORAL_FEATURE_DIM → TEMPORAL_HIDDEN_DIM (temporal).
    pub(crate) w_temporal: WeightMatrix,
    /// Mastery-learned output head: GROUP_HIDDEN_DIM → num_phonemes (single-frame).
    pub(crate) output_head: WeightMatrix,
    /// Mastery-learned output head: TEMPORAL_HIDDEN_DIM → num_phonemes (temporal).
    pub(crate) temporal_head: WeightMatrix,
    /// IPA labels (same order as output head rows).
    pub(crate) labels: Vec<String>,
}

/// Ensemble phoneme classifier with 12 acoustic group classifiers.
pub struct PhonemeClassifier {
    groups: Vec<GroupClassifier>,
    input_dim: usize,
}

impl PhonemeClassifier {
    /// Build the classifier by loading trained output heads from cache.
    ///
    /// Training is done separately via `train-phonemes` binary. This function
    /// only loads the inference cache. If no cache exists, falls back to
    /// zeroed heads (for test purposes only).
    pub fn build(mfcc_config: &MfccConfig, cache_dir: Option<&Path>) -> Result<Self> {
        let input_dim = mfcc_config.n_mfcc;

        // Try loading trained inference cache.
        let cache_path = cache_dir.map(|d| d.join("phoneme_ensemble_heads.bin"));
        if let Some(ref path) = cache_path {
            if let Ok(groups) = load_ensemble_cache(path, input_dim) {
                log::info!(
                    "PhonemeClassifier: loaded inference cache ({} groups, {} phonemes)",
                    groups.len(),
                    groups.iter().map(|g| g.labels.len()).sum::<usize>()
                );
                return Ok(Self { groups, input_dim });
            }
        }

        // No trained cache. Run quick mastery bootstrap from oscillator data.
        log::warn!(
            "PhonemeClassifier: no trained cache found. Running quick bootstrap. \
             For production, run `train-phonemes --export` first."
        );

        let mut trainer = crate::training::PhonemeTrainer::new();
        let frame_data = crate::training::generate_frame_training_data(mfcc_config)?;
        let temporal_data = crate::training::generate_temporal_from_oscillator(mfcc_config)
            .unwrap_or_default();
        let data = crate::training::TrainingData {
            frame_samples: frame_data,
            temporal_samples: temporal_data,
        };
        // Run quick exposure cycles for bootstrap.
        for _ in 0..30 {
            trainer.run_cycle(&data);
        }
        let groups = trainer.export_group_classifiers(input_dim);

        Ok(Self { groups, input_dim })
    }

    /// Classify a single MFCC frame into top-K phoneme candidates.
    ///
    /// Runs all 12 group classifiers, collects their best candidates,
    /// and returns the overall top-K sorted by confidence.
    pub fn classify(&self, mfcc: &[f32], k: usize) -> Vec<PhonemeCandidate> {
        let input = encode_mfcc(mfcc, self.input_dim);
        let mut all_candidates: Vec<PhonemeCandidate> = Vec::with_capacity(NUM_GROUPS * 2);

        for group in &self.groups {
            let hidden_packed = forward_hidden_packed(&group.w_hidden, &input);
            let logits = group.output_head.matmul(&hidden_packed);

            classify_from_logits(&logits, &group.labels, &mut all_candidates);
        }

        all_candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_candidates.truncate(k);
        all_candidates
    }

    /// Classify a temporal feature vector (78-dim) into top-K phoneme candidates.
    ///
    /// Same ensemble architecture as `classify`, but uses temporal projections
    /// and mastery-learned temporal output heads.
    pub fn classify_temporal(&self, temporal_features: &[f32], k: usize) -> Vec<PhonemeCandidate> {
        let input = encode_mfcc(temporal_features, TEMPORAL_FEATURE_DIM);
        let mut all_candidates: Vec<PhonemeCandidate> = Vec::with_capacity(NUM_GROUPS * 2);

        for group in &self.groups {
            let hidden_packed = forward_hidden_packed(&group.w_temporal, &input);
            let logits = group.temporal_head.matmul(&hidden_packed);

            classify_from_logits(&logits, &group.labels, &mut all_candidates);
        }

        all_candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_candidates.truncate(k);
        all_candidates
    }

    /// Number of phonemes across all groups.
    pub fn len(&self) -> usize {
        NUM_PHONEMES
    }

    /// Whether the classifier has no phonemes (always false).
    pub fn is_empty(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Hidden layer and encoding
// ---------------------------------------------------------------------------

/// Initialize a frozen random ternary projection matrix.
///
/// Deterministic PRNG seeded by `seed`. Each weight has random polarity (±1)
/// and magnitude in 20-40, following the proven pattern from
/// adaptive-ternary-results.
pub(crate) fn init_frozen_hidden(input_dim: usize, hidden_dim: usize, seed: u64) -> WeightMatrix {
    let mut wm = WeightMatrix::zeros(hidden_dim, input_dim);
    let mut rng = seed;

    for i in 0..hidden_dim * input_dim {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;

        let polarity: i8 = if rng % 2 == 0 { 1 } else { -1 };
        let magnitude = 20 + (rng >> 32) as u8 % 21;
        wm.data[i] = PackedSignal::pack(polarity, magnitude, 1);
    }

    wm
}

/// Forward pass through a frozen hidden layer: matmul + ReLU → PackedSignal activations.
///
/// Returns PackedSignal vector (for feeding into output head matmul).
/// ReLU: negative outputs are zeroed.
pub(crate) fn forward_hidden_packed(w_hidden: &WeightMatrix, input: &[PackedSignal]) -> Vec<PackedSignal> {
    let packed_output = w_hidden.matmul(input);
    packed_output
        .iter()
        .map(|s| {
            if s.current() > 0 {
                *s
            } else {
                PackedSignal::ZERO
            }
        })
        .collect()
}

/// Forward pass through a frozen hidden layer: matmul + ReLU → f32 activations.
fn forward_hidden(w_hidden: &WeightMatrix, input: &[PackedSignal]) -> Vec<f32> {
    let packed_output = w_hidden.matmul(input);
    packed_output
        .iter()
        .map(|s| {
            let val = s.current() as f32;
            if val > 0.0 { val } else { 0.0 }
        })
        .collect()
}

/// Extract phoneme candidates from output head logits.
///
/// Normalizes confidence within each group by dividing by the sum of
/// positive logits. This makes cross-group comparison fair regardless of
/// the absolute magnitude of learned synaptic strengths.
fn classify_from_logits(
    logits: &[PackedSignal],
    labels: &[String],
    candidates: &mut Vec<PhonemeCandidate>,
) {
    let mut group_scores: Vec<(usize, i32)> = logits
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.current()))
        .collect();

    group_scores.sort_by(|a, b| b.1.cmp(&a.1));

    // Normalize by sum of positive logits — what fraction of evidence
    // supports this phoneme within this group.
    let positive_sum: f32 = group_scores
        .iter()
        .filter(|(_, s)| *s > 0)
        .map(|(_, s)| *s as f32)
        .sum::<f32>()
        .max(1.0);

    let take = if labels.len() > 5 { 2 } else { 1 };
    for &(local_idx, score) in group_scores.iter().take(take) {
        if score <= 0 {
            continue;
        }
        let confidence = score as f32 / positive_sum;
        candidates.push(PhonemeCandidate {
            ipa: labels[local_idx].clone(),
            score,
            confidence,
        });
    }
}

/// Encode an MFCC frame (f32 vector) to PackedSignal vector.
pub(crate) fn encode_mfcc(mfcc: &[f32], expected_dim: usize) -> Vec<PackedSignal> {
    let mut result = Vec::with_capacity(expected_dim);

    for i in 0..expected_dim {
        let val = mfcc.get(i).copied().unwrap_or(0.0);
        let polarity: i8 = if val > 0.01 {
            1
        } else if val < -0.01 {
            -1
        } else {
            0
        };
        let magnitude = (val.abs() * 12.75).min(255.0) as u8;
        result.push(PackedSignal::pack(polarity, magnitude, 1));
    }

    result
}

// ---------------------------------------------------------------------------
// Cache I/O (inference only — training is in training.rs)
// ---------------------------------------------------------------------------

// Training code has been moved to training.rs and the train-phonemes binary.
// This module is inference-only: load cached heads, classify frames.

// REMOVED: mastery_train_head, apply_transition_fn, step_up_packed, step_down_packed,
//          generate_training_data, generate_temporal_training_data,
//          generate_temporal_from_librispeech, generate_temporal_from_oscillator,
//          align_transcript_to_frames, save_ensemble_cache, REPR_LEVELS,
//          BatchMasteryConfig, MIN_FRAME_ENERGY, MIN_PHONEME_SPAN, etc.
//
// All now live in src/training.rs + src/mastery_cache.rs.
// Run `cargo run --release --bin train-phonemes -- --cycles 20 --export`
// to generate the inference cache.
/// Load ensemble output heads from a binary inference cache file.
fn load_ensemble_cache(path: &Path, input_dim: usize) -> std::io::Result<Vec<GroupClassifier>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;

    // Header.
    let mut header = [0u8; 20];
    file.read_exact(&mut header)?;
    let version = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let num_groups = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let hidden_dim = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let temporal_hidden_dim = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
    let total_phonemes = u32::from_le_bytes([header[16], header[17], header[18], header[19]]) as usize;

    if version != CACHE_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("cache version {} != expected {}", version, CACHE_VERSION),
        ));
    }
    if num_groups != NUM_GROUPS || hidden_dim != GROUP_HIDDEN_DIM
        || temporal_hidden_dim != TEMPORAL_HIDDEN_DIM || total_phonemes != NUM_PHONEMES
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ensemble cache dimension mismatch",
        ));
    }

    let read_weight_matrix = |file: &mut std::fs::File| -> std::io::Result<WeightMatrix> {
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)?;
        WeightMatrix::from_bytes(&buf).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    };

    let mut groups = Vec::with_capacity(num_groups);
    for i in 0..num_groups {
        let mut group_header = [0u8; 8];
        file.read_exact(&mut group_header)?;
        let seed =
            u32::from_le_bytes([group_header[0], group_header[1], group_header[2], group_header[3]])
                as u64;
        let _num_phonemes =
            u32::from_le_bytes([group_header[4], group_header[5], group_header[6], group_header[7]])
                as usize;

        if seed != ACOUSTIC_GROUPS[i].seed {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("group {} seed mismatch: {} vs {}", i, seed, ACOUSTIC_GROUPS[i].seed),
            ));
        }

        let output_head = read_weight_matrix(&mut file)?;
        let temporal_head = read_weight_matrix(&mut file)?;

        let w_hidden = init_frozen_hidden(input_dim, hidden_dim, ACOUSTIC_GROUPS[i].seed);
        let w_temporal = init_frozen_hidden(
            TEMPORAL_FEATURE_DIM,
            temporal_hidden_dim,
            ACOUSTIC_GROUPS[i].seed + TEMPORAL_SEED_OFFSET,
        );
        let labels: Vec<String> = ACOUSTIC_GROUPS[i]
            .phonemes
            .iter()
            .map(|s| s.to_string())
            .collect();

        groups.push(GroupClassifier {
            w_hidden,
            w_temporal,
            output_head,
            temporal_head,
            labels,
        });
    }

    Ok(groups)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mfcc::MfccExtractor;
    use crate::preprocess::pre_emphasis;
    use crate::temporal_context::TemporalAccumulator;
    use signal_oscillator::renderer::{Interpolation, RenderConfig};
    use signal_oscillator::speech::{SpeechConfig, SpeechSynthesizer};

    #[test]
    fn all_groups_cover_all_phonemes() {
        let mut covered: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for group in &ACOUSTIC_GROUPS {
            for &p in group.phonemes {
                assert!(covered.insert(p), "phoneme /{p}/ appears in multiple groups");
            }
        }
        assert_eq!(covered.len(), NUM_PHONEMES, "groups must cover all phonemes");
    }

    #[test]
    fn group_count_matches() {
        let total: usize = ACOUSTIC_GROUPS.iter().map(|g| g.phonemes.len()).sum();
        assert_eq!(total, NUM_PHONEMES);
        assert_eq!(ACOUSTIC_GROUPS.len(), NUM_GROUPS);
    }

    #[test]
    fn encode_mfcc_basic() {
        let mfcc = vec![0.0, 5.0, -3.0, 0.001, 10.0];
        let encoded = encode_mfcc(&mfcc, 5);
        assert_eq!(encoded.len(), 5);
        assert_eq!(encoded[0].current(), 0);
        assert!(encoded[1].current() > 0);
        assert!(encoded[2].current() < 0);
    }

    #[test]
    fn frozen_hidden_is_deterministic() {
        let a = init_frozen_hidden(26, 32, 42);
        let b = init_frozen_hidden(26, 32, 42);
        for i in 0..a.data.len() {
            assert_eq!(a.data[i].as_u8(), b.data[i].as_u8());
        }
    }

    #[test]
    fn different_seeds_give_different_projections() {
        let a = init_frozen_hidden(26, 32, 1001);
        let b = init_frozen_hidden(26, 32, 2003);
        let same = a.data.iter().zip(b.data.iter()).filter(|(x, y)| x.as_u8() == y.as_u8()).count();
        // With 832 weights and random projections, very few should match.
        assert!(same < a.data.len() / 2, "different seeds should produce different projections");
    }

    #[test]
    fn training_data_generates_samples() {
        let config = MfccConfig::wideband();
        let samples = crate::training::generate_frame_training_data(&config).unwrap();
        assert!(samples.len() >= 100);

        let unique_labels: std::collections::HashSet<usize> =
            samples.iter().map(|(_, idx)| *idx).collect();
        assert!(unique_labels.len() >= 20);
    }

    #[test]
    fn classifier_builds_and_classifies() {
        let config = MfccConfig::wideband();
        let classifier =
            PhonemeClassifier::build(&config, None).expect("classifier should build");
        assert_eq!(classifier.len(), NUM_PHONEMES);

        // Synthesize a known vowel and classify it.
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
        let mut audio = synth.pronounce_ipa("ɑ");
        pre_emphasis(&mut audio, 0.97);

        let extractor = MfccExtractor::new(config);
        let frames = extractor.extract(&audio);
        assert!(!frames.is_empty());

        let mid = frames.len() / 2;
        let candidates = classifier.classify(&frames[mid], 5);
        assert!(
            !candidates.is_empty(),
            "classifier should return candidates"
        );

        let top = &candidates[0];
        log::info!("Top candidate for /ɑ/: /{}/  conf={:.2}", top.ipa, top.confidence);
        let sonorants = [
            "i", "ɪ", "e", "ɛ", "æ", "a", "ɑ", "ɒ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ", "ɐ",
            "iː", "ɑː", "ɔː", "uː", "ɜː", "aɪ", "aʊ", "eɪ", "ɔɪ", "əʊ", "oʊ", "ɪə", "eə",
            "m", "n", "ŋ", "l", "ɹ", "w", "j",
        ];
        assert!(
            sonorants.contains(&top.ipa.as_str()),
            "expected a voiced phoneme, got /{}/",
            top.ipa
        );
    }

    #[test]
    fn classifier_produces_diverse_output() {
        let config = MfccConfig::wideband();
        let classifier =
            PhonemeClassifier::build(&config, None).expect("classifier should build");
        let extractor = MfccExtractor::new(config.clone());

        // Synthesize a range of acoustically distinct phonemes.
        let test_phonemes = ["ɑ", "i", "s", "m", "u", "æ", "ɹ", "t", "ɛ", "n"];
        let mut top_results: Vec<String> = Vec::new();

        for &ph in &test_phonemes {
            let synth_config = SpeechConfig {
                base_f0: 120.0,
                speech_rate: 1.0,
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
            synth.reset();
            let mut audio = synth.pronounce_ipa(ph);
            pre_emphasis(&mut audio, 0.97);

            let frames = extractor.extract(&audio);
            if frames.len() < 3 {
                continue;
            }

            let mid = frames.len() / 2;
            let candidates = classifier.classify(&frames[mid], 5);
            if let Some(top) = candidates.first() {
                eprintln!(
                    "  /{:<3}/ → /{:<4}/ conf={:.3}  (alts: {:?})",
                    ph,
                    top.ipa,
                    top.confidence,
                    candidates.iter().map(|c| c.ipa.as_str()).collect::<Vec<_>>()
                );
                top_results.push(top.ipa.clone());
            }
        }

        let unique: std::collections::HashSet<&str> =
            top_results.iter().map(|s| s.as_str()).collect();
        eprintln!(
            "\nDiversity: {} unique top-1 phonemes from {} inputs",
            unique.len(),
            test_phonemes.len()
        );
        // With 10 acoustically distinct inputs, we should get at least 5 unique top-1 predictions.
        assert!(
            unique.len() >= 5,
            "ensemble should produce diverse output: got {} unique from {} inputs",
            unique.len(),
            test_phonemes.len()
        );
    }

    #[test]
    fn temporal_classifier_produces_diverse_output() {
        let config = MfccConfig::wideband();
        let classifier =
            PhonemeClassifier::build(&config, None).expect("classifier should build");
        let extractor = MfccExtractor::new(config.clone());

        let test_phonemes = ["ɑ", "i", "s", "m", "u", "æ", "ɹ", "t", "ɛ", "n"];
        let mut top_results: Vec<String> = Vec::new();

        for &ph in &test_phonemes {
            let synth_config = SpeechConfig {
                base_f0: 120.0,
                speech_rate: 1.0,
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
            synth.reset();
            let mut audio = synth.pronounce_ipa(ph);
            pre_emphasis(&mut audio, 0.97);

            let frames = extractor.extract(&audio);
            if frames.len() < 3 {
                continue;
            }

            // Feed frames through temporal accumulator (like real pipeline).
            let mut acc = TemporalAccumulator::new();
            for frame in &frames {
                acc.write_frame(frame);
            }
            let temporal_features = acc.extract_features();
            let candidates = classifier.classify_temporal(&temporal_features, 5);

            if let Some(top) = candidates.first() {
                eprintln!(
                    "  TEMPORAL /{:<3}/ → /{:<4}/ conf={:.3}  (alts: {:?})",
                    ph,
                    top.ipa,
                    top.confidence,
                    candidates.iter().map(|c| c.ipa.as_str()).collect::<Vec<_>>()
                );
                top_results.push(top.ipa.clone());
            }
        }

        let unique: std::collections::HashSet<&str> =
            top_results.iter().map(|s| s.as_str()).collect();
        eprintln!(
            "\nTemporal diversity: {} unique top-1 phonemes from {} inputs",
            unique.len(),
            test_phonemes.len()
        );
        // Temporal heads need sustained mastery learning for full diversity.
        // Bootstrap path (no trained cache) achieves ~4 unique with limited data.
        // Production diversity comes from `train-phonemes` with LibriSpeech corpus.
        assert!(
            unique.len() >= 3,
            "temporal ensemble should produce diverse output: got {} unique from {} inputs",
            unique.len(),
            test_phonemes.len()
        );
    }
}
