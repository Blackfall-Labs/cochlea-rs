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

use crate::error::{CochleaError, Result};
use crate::mfcc::{MfccConfig, MfccExtractor};
use crate::phoneme_bank::PhonemeCandidate;
use crate::preprocess::pre_emphasis;
use crate::temporal_context::{TemporalAccumulator, TEMPORAL_FEATURE_DIM};
use atomic_neural_transistors::{WeightMatrix, PackedSignal};
use signal_oscillator::ipa;
use signal_oscillator::renderer::{Interpolation, RenderConfig};
use signal_oscillator::speech::{SpeechConfig, SpeechSynthesizer};
use std::path::Path;

/// Total number of phonemes across all groups.
const NUM_PHONEMES: usize = 45;

/// Hidden dimension per group classifier (single-frame).
const GROUP_HIDDEN_DIM: usize = 32;

/// Hidden dimension per group classifier (temporal).
/// Larger than single-frame because temporal input is 78-dim vs 26-dim.
const TEMPORAL_HIDDEN_DIM: usize = 48;

/// Number of acoustic groups.
const NUM_GROUPS: usize = 12;

/// The full ordered list of phonemes (used for training data generation).
const PHONEME_LIST: [&str; NUM_PHONEMES] = [
    "i", "ɪ", "e", "ɛ", "æ", "a", "ɑ", "ɒ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ", "ɐ",
    "iː", "ɑː", "ɔː", "uː", "ɜː",
    "aɪ", "aʊ", "eɪ", "ɔɪ", "əʊ", "oʊ", "ɪə", "eə",
    "m", "n", "ŋ",
    "l", "ɹ", "w", "j",
    "p", "b", "t", "d", "k", "ɡ",
    "s", "ʃ", "h",
];

/// Definition of one acoustic group.
struct AcousticGroup {
    name: &'static str,
    phonemes: &'static [&'static str],
    seed: u64,
}

/// The 12 acoustic groups, split by formant structure and manner.
const ACOUSTIC_GROUPS: [AcousticGroup; NUM_GROUPS] = [
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
const TEMPORAL_SEED_OFFSET: u64 = 100_000;

/// Cache format version — bump to invalidate old caches.
const CACHE_VERSION: u32 = 2;

/// Runtime state for one acoustic group's classifier.
struct GroupClassifier {
    /// Frozen random projection: input_dim → GROUP_HIDDEN_DIM (single-frame).
    w_hidden: WeightMatrix,
    /// Frozen random projection: TEMPORAL_FEATURE_DIM → GROUP_HIDDEN_DIM (temporal).
    w_temporal: WeightMatrix,
    /// L2-normalized centroid vectors for single-frame classification.
    centroids: Vec<Vec<f32>>,
    /// L2-normalized centroid vectors for temporal classification.
    temporal_centroids: Vec<Vec<f32>>,
    /// IPA labels (same order as centroids).
    labels: Vec<String>,
}

/// Ensemble phoneme classifier with 12 acoustic group classifiers.
pub struct PhonemeClassifier {
    groups: Vec<GroupClassifier>,
    input_dim: usize,
}

impl PhonemeClassifier {
    /// Build the ensemble classifier from oscillator-synthesized training data.
    pub fn build(mfcc_config: &MfccConfig, cache_dir: Option<&Path>) -> Result<Self> {
        let input_dim = mfcc_config.n_mfcc;

        // Try loading cached ensemble.
        let cache_path = cache_dir.map(|d| d.join("phoneme_ensemble_centroids.bin"));
        if let Some(ref path) = cache_path {
            if let Ok(groups) = load_ensemble_cache(path, input_dim) {
                log::info!(
                    "PhonemeClassifier: loaded ensemble cache ({} groups, {} phonemes)",
                    groups.len(),
                    groups.iter().map(|g| g.labels.len()).sum::<usize>()
                );
                return Ok(Self { groups, input_dim });
            }
        }

        // Generate training data once (shared across all groups).
        log::info!("PhonemeClassifier: computing ensemble centroids from oscillator...");
        let all_samples = generate_training_data(mfcc_config)?;
        let all_temporal_samples = generate_temporal_training_data(mfcc_config)?;
        log::info!(
            "  Single-frame samples: {}, temporal samples: {}",
            all_samples.len(),
            all_temporal_samples.len()
        );

        // Build each group classifier.
        let mut groups = Vec::with_capacity(NUM_GROUPS);
        for (group_idx, group_def) in ACOUSTIC_GROUPS.iter().enumerate() {
            let w_hidden = init_frozen_hidden(input_dim, GROUP_HIDDEN_DIM, group_def.seed);
            let w_temporal = init_frozen_hidden(
                TEMPORAL_FEATURE_DIM,
                TEMPORAL_HIDDEN_DIM,
                group_def.seed + TEMPORAL_SEED_OFFSET,
            );
            let labels: Vec<String> = group_def.phonemes.iter().map(|s| s.to_string()).collect();

            // Map global phoneme index → local group index.
            let global_to_local: Vec<Option<usize>> = PHONEME_LIST
                .iter()
                .map(|&ipa| group_def.phonemes.iter().position(|&p| p == ipa))
                .collect();

            // Filter single-frame training samples to this group.
            let group_samples: Vec<(&Vec<f32>, usize)> = all_samples
                .iter()
                .filter_map(|(mfcc, global_idx)| {
                    global_to_local[*global_idx].map(|local_idx| (mfcc, local_idx))
                })
                .collect();

            let centroids = compute_group_centroids(
                &w_hidden,
                &group_samples,
                input_dim,
                group_def.phonemes.len(),
            );

            // Filter temporal training samples to this group.
            let temporal_group_samples: Vec<(&Vec<f32>, usize)> = all_temporal_samples
                .iter()
                .filter_map(|(features, global_idx)| {
                    global_to_local[*global_idx].map(|local_idx| (features, local_idx))
                })
                .collect();

            let temporal_centroids = compute_group_centroids(
                &w_temporal,
                &temporal_group_samples,
                TEMPORAL_FEATURE_DIM,
                group_def.phonemes.len(),
            );

            log::info!(
                "  Group {:>2} '{}': {} phonemes, {} frame / {} temporal samples",
                group_idx,
                group_def.name,
                group_def.phonemes.len(),
                group_samples.len(),
                temporal_group_samples.len(),
            );

            groups.push(GroupClassifier {
                w_hidden,
                w_temporal,
                centroids,
                temporal_centroids,
                labels,
            });
        }

        // Cache the ensemble.
        if let Some(ref path) = cache_path {
            if let Err(e) = save_ensemble_cache(path, &groups) {
                log::warn!("Failed to cache ensemble centroids: {}", e);
            }
        }

        log::info!(
            "PhonemeClassifier ready: {} groups, {} phonemes, {}-dim per group",
            NUM_GROUPS,
            NUM_PHONEMES,
            GROUP_HIDDEN_DIM
        );

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
            let hidden = forward_hidden(&group.w_hidden, &input);

            // L2-normalize hidden vector for true cosine similarity.
            let h_norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);

            // Compute cosine similarity with each centroid (centroids already L2-normalized).
            let mut group_scores: Vec<(usize, f32)> = group
                .centroids
                .iter()
                .enumerate()
                .map(|(i, centroid)| {
                    let cosine: f32 = hidden
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&h, &c)| (h / h_norm) * c)
                        .sum();
                    (i, cosine)
                })
                .collect();

            group_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top-1 from small groups, top-2 from large groups (>5 phonemes).
            let take = if group.labels.len() > 5 { 2 } else { 1 };

            for &(local_idx, score) in group_scores.iter().take(take) {
                if score <= 0.0 {
                    continue;
                }
                // Raw cosine similarity is already 0..1 (since hidden is ReLU'd, all positive).
                all_candidates.push(PhonemeCandidate {
                    ipa: group.labels[local_idx].clone(),
                    score: (score * 1000.0) as i32,
                    confidence: score,
                });
            }
        }

        // Sort by confidence (raw cosine similarity) descending.
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
    /// and temporal centroids computed from windowed MFCC features.
    pub fn classify_temporal(&self, temporal_features: &[f32], k: usize) -> Vec<PhonemeCandidate> {
        let input = encode_mfcc(temporal_features, TEMPORAL_FEATURE_DIM);

        let mut all_candidates: Vec<PhonemeCandidate> = Vec::with_capacity(NUM_GROUPS * 2);

        for group in &self.groups {
            let hidden = forward_hidden(&group.w_temporal, &input);

            // L2-normalize hidden vector for true cosine similarity.
            let h_norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);

            // Compute cosine similarity with each temporal centroid.
            let mut group_scores: Vec<(usize, f32)> = group
                .temporal_centroids
                .iter()
                .enumerate()
                .map(|(i, centroid)| {
                    let cosine: f32 = hidden
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&h, &c)| (h / h_norm) * c)
                        .sum();
                    (i, cosine)
                })
                .collect();

            group_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let take = if group.labels.len() > 5 { 2 } else { 1 };

            for &(local_idx, score) in group_scores.iter().take(take) {
                if score <= 0.0 {
                    continue;
                }
                all_candidates.push(PhonemeCandidate {
                    ipa: group.labels[local_idx].clone(),
                    score: (score * 1000.0) as i32,
                    confidence: score,
                });
            }
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
fn init_frozen_hidden(input_dim: usize, hidden_dim: usize, seed: u64) -> WeightMatrix {
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

/// Encode an MFCC frame (f32 vector) to PackedSignal vector.
fn encode_mfcc(mfcc: &[f32], expected_dim: usize) -> Vec<PackedSignal> {
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
// Centroid computation
// ---------------------------------------------------------------------------

/// Compute centroid vectors for a single group from training data.
///
/// Averages hidden-layer activations per class, then L2-normalizes.
fn compute_group_centroids(
    w_hidden: &WeightMatrix,
    samples: &[(&Vec<f32>, usize)],
    input_dim: usize,
    num_classes: usize,
) -> Vec<Vec<f32>> {
    let hidden_dim = w_hidden.rows;
    let mut sums = vec![vec![0.0f64; hidden_dim]; num_classes];
    let mut counts = vec![0usize; num_classes];

    for (mfcc_frame, local_idx) in samples {
        let input = encode_mfcc(mfcc_frame, input_dim);
        let hidden = forward_hidden(w_hidden, &input);

        for (j, &h) in hidden.iter().enumerate() {
            sums[*local_idx][j] += h as f64;
        }
        counts[*local_idx] += 1;
    }

    let mut centroids = Vec::with_capacity(num_classes);
    for i in 0..num_classes {
        let n = counts[i].max(1) as f64;
        let mean: Vec<f32> = sums[i].iter().map(|&s| (s / n) as f32).collect();

        // L2-normalize for cosine similarity.
        let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let normalized: Vec<f32> = mean.iter().map(|x| x / norm).collect();
        centroids.push(normalized);
    }

    centroids
}

// ---------------------------------------------------------------------------
// Training data generation
// ---------------------------------------------------------------------------

/// Generate training data by synthesizing phonemes with the oscillator.
///
/// Returns (mfcc_frame, global_phoneme_index) pairs.
fn generate_training_data(mfcc_config: &MfccConfig) -> Result<Vec<(Vec<f32>, usize)>> {
    let extractor = MfccExtractor::new(mfcc_config.clone());
    let f0_values = [100.0, 130.0, 160.0, 200.0, 250.0];
    let speech_rates = [0.8, 1.2];
    let mut samples = Vec::new();

    for (phoneme_idx, &ipa_str) in PHONEME_LIST.iter().enumerate() {
        let tokens = ipa::tokenize(ipa_str);
        if tokens.is_empty() {
            log::warn!("Cannot tokenize phoneme /{}/", ipa_str);
            continue;
        }

        for &f0 in &f0_values {
            for &rate in &speech_rates {
                let synth_config = SpeechConfig {
                    base_f0: f0,
                    speech_rate: rate,
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
                let mut audio = synth.pronounce_ipa(ipa_str);

                if audio.is_empty() || audio.len() < mfcc_config.win_length as usize {
                    continue;
                }

                pre_emphasis(&mut audio, 0.97);

                let frames = extractor.extract(&audio);
                if frames.len() < 3 {
                    continue;
                }

                let start = frames.len() / 4;
                let end = (3 * frames.len()) / 4;
                for frame in &frames[start..end] {
                    samples.push((frame.clone(), phoneme_idx));
                }
            }
        }
    }

    if samples.is_empty() {
        return Err(CochleaError::Config(
            "No training samples generated".to_string(),
        ));
    }

    log::info!(
        "Generated {} training samples from {} phonemes",
        samples.len(),
        PHONEME_LIST.len()
    );

    Ok(samples)
}

/// Generate temporal training data by synthesizing phonemes and feeding
/// frames through a TemporalAccumulator.
///
/// Returns (78-dim temporal_features, global_phoneme_index) pairs.
fn generate_temporal_training_data(mfcc_config: &MfccConfig) -> Result<Vec<(Vec<f32>, usize)>> {
    let extractor = MfccExtractor::new(mfcc_config.clone());
    let f0_values = [100.0, 130.0, 160.0, 200.0, 250.0];
    let speech_rates = [0.8, 1.2];
    let mut samples = Vec::new();

    for (phoneme_idx, &ipa_str) in PHONEME_LIST.iter().enumerate() {
        let tokens = ipa::tokenize(ipa_str);
        if tokens.is_empty() {
            continue;
        }

        for &f0 in &f0_values {
            for &rate in &speech_rates {
                let synth_config = SpeechConfig {
                    base_f0: f0,
                    speech_rate: rate,
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
                let mut audio = synth.pronounce_ipa(ipa_str);

                if audio.is_empty() || audio.len() < mfcc_config.win_length as usize {
                    continue;
                }

                pre_emphasis(&mut audio, 0.97);

                let frames = extractor.extract(&audio);
                if frames.len() < 3 {
                    continue;
                }

                // Feed frames through a TemporalAccumulator with silence context.
                // Pre-fill with a few silent frames so onset/transition patterns
                // are captured in the delta features.
                let mut acc = TemporalAccumulator::new();
                let silence = vec![0.0f32; mfcc_config.n_mfcc];
                for _ in 0..3 {
                    acc.write_frame(&silence);
                }

                let start = frames.len() / 4;
                let end = (3 * frames.len()) / 4;

                for (i, frame) in frames.iter().enumerate() {
                    acc.write_frame(frame);
                    // Only collect features from the middle 50%.
                    if i >= start && i < end {
                        let temporal_features = acc.extract_features();
                        samples.push((temporal_features, phoneme_idx));
                    }
                }
            }
        }
    }

    if samples.is_empty() {
        return Err(CochleaError::Config(
            "No temporal training samples generated".to_string(),
        ));
    }

    log::info!(
        "Generated {} temporal training samples from {} phonemes",
        samples.len(),
        PHONEME_LIST.len()
    );

    Ok(samples)
}

// ---------------------------------------------------------------------------
// Cache I/O
// ---------------------------------------------------------------------------

/// Save ensemble centroids (single-frame + temporal) to a binary file.
fn save_ensemble_cache(path: &Path, groups: &[GroupClassifier]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    // Header: version, num_groups, hidden_dim, temporal_hidden_dim, total_phonemes.
    file.write_all(&CACHE_VERSION.to_le_bytes())?;
    file.write_all(&(groups.len() as u32).to_le_bytes())?;
    file.write_all(&(GROUP_HIDDEN_DIM as u32).to_le_bytes())?;
    file.write_all(&(TEMPORAL_HIDDEN_DIM as u32).to_le_bytes())?;
    let total_phonemes: u32 = groups.iter().map(|g| g.labels.len() as u32).sum();
    file.write_all(&total_phonemes.to_le_bytes())?;

    // Per group: seed, num_phonemes, single-frame centroids, temporal centroids.
    for (i, group) in groups.iter().enumerate() {
        let num_phonemes = group.labels.len() as u32;
        let seed = ACOUSTIC_GROUPS[i].seed as u32;
        file.write_all(&seed.to_le_bytes())?;
        file.write_all(&num_phonemes.to_le_bytes())?;

        // Single-frame centroids.
        for centroid in &group.centroids {
            for &val in centroid {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        // Temporal centroids.
        for centroid in &group.temporal_centroids {
            for &val in centroid {
                file.write_all(&val.to_le_bytes())?;
            }
        }
    }

    Ok(())
}

/// Load ensemble centroids (single-frame + temporal) from a binary file.
fn load_ensemble_cache(path: &Path, input_dim: usize) -> std::io::Result<Vec<GroupClassifier>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;

    // Header: version, num_groups, hidden_dim, temporal_hidden_dim, total_phonemes.
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

    let read_centroids = |file: &mut std::fs::File, count: usize, dim: usize| -> std::io::Result<Vec<Vec<f32>>> {
        let bytes_per_centroid = dim * 4;
        let mut centroids = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buf = vec![0u8; bytes_per_centroid];
            file.read_exact(&mut buf)?;
            let centroid: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            centroids.push(centroid);
        }
        Ok(centroids)
    };

    let mut groups = Vec::with_capacity(num_groups);
    for i in 0..num_groups {
        let mut group_header = [0u8; 8];
        file.read_exact(&mut group_header)?;
        let seed =
            u32::from_le_bytes([group_header[0], group_header[1], group_header[2], group_header[3]])
                as u64;
        let num_phonemes =
            u32::from_le_bytes([group_header[4], group_header[5], group_header[6], group_header[7]])
                as usize;

        if seed != ACOUSTIC_GROUPS[i].seed {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("group {} seed mismatch: {} vs {}", i, seed, ACOUSTIC_GROUPS[i].seed),
            ));
        }

        let centroids = read_centroids(&mut file, num_phonemes, hidden_dim)?;
        let temporal_centroids = read_centroids(&mut file, num_phonemes, temporal_hidden_dim)?;

        let w_hidden = init_frozen_hidden(input_dim, hidden_dim, ACOUSTIC_GROUPS[i].seed);
        let w_temporal = init_frozen_hidden(
            TEMPORAL_FEATURE_DIM,
            hidden_dim,
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
            centroids,
            temporal_centroids,
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
        let samples = generate_training_data(&config).unwrap();
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
        assert!(
            unique.len() >= 5,
            "temporal ensemble should produce diverse output: got {} unique from {} inputs",
            unique.len(),
            test_phonemes.len()
        );
    }
}
