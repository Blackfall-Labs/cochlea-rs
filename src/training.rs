//! Phased mastery learning engine for phoneme classifier output heads.
//!
//! Implements sustained mastery learning following the thinkr pattern:
//! - Persistent MasteryState across cycles (pressure accumulates over time)
//! - Full LibriSpeech corpus exposure each cycle (not subsampled)
//! - Phase progression: Exposure → Refinement → Stabilization
//! - Checkpoint after every cycle for resume capability
//!
//! Training is a separate binary (`train-phonemes`), not part of inference.

use crate::cmudict;
use crate::error::{CochleaError, Result};
use crate::librispeech;
use crate::mastery_cache::{
    self, TrainableGroupState, TrainingPhase, TrainingState,
};
use crate::mfcc::{MfccConfig, MfccExtractor};
use crate::phoneme_classifier::{
    encode_mfcc, forward_hidden_packed, init_frozen_hidden,
    ACOUSTIC_GROUPS, GROUP_HIDDEN_DIM, NUM_GROUPS, NUM_PHONEMES,
    PHONEME_LIST, TEMPORAL_HIDDEN_DIM, TEMPORAL_SEED_OFFSET,
};
use crate::preprocess::pre_emphasis;
use crate::temporal_context::{TemporalAccumulator, TEMPORAL_FEATURE_DIM};
use atomic_neural_transistors::core::weight_matrix::packed_from_current;
use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use atomic_neural_transistors::WeightMatrix;
use signal_oscillator::ipa;
use signal_oscillator::renderer::{Interpolation, RenderConfig};
use signal_oscillator::speech::{SpeechConfig, SpeechSynthesizer};
use ternary_signal::PackedSignal;
use std::path::Path;

/// Minimum MFCC frame energy (coeff 0) to consider a frame speech-active.
const MIN_FRAME_ENERGY: f32 = 2.0;

/// Minimum phoneme span width (in frames) for alignment labeling.
const MIN_PHONEME_SPAN: usize = 3;

/// Pre-computed training data for one mastery cycle.
pub struct TrainingData {
    /// Single-frame samples: (MFCC coeffs, global phoneme index).
    pub frame_samples: Vec<(Vec<f32>, usize)>,
    /// Temporal samples: (78-dim features, global phoneme index).
    pub temporal_samples: Vec<(Vec<f32>, usize)>,
}

/// Metrics from one mastery cycle.
pub struct CycleMetrics {
    pub cycle: u64,
    pub phase: TrainingPhase,
    pub total_frame_transitions: u64,
    pub total_temporal_transitions: u64,
    pub per_group_frame_transitions: Vec<(String, u64)>,
    pub per_group_temporal_transitions: Vec<(String, u64)>,
    pub frame_samples_processed: usize,
    pub temporal_samples_processed: usize,
}

/// Phased mastery learning engine.
pub struct PhonemeTrainer {
    groups: Vec<TrainableGroup>,
    total_cycles: u64,
    total_samples: u64,
}

struct TrainableGroup {
    w_hidden: WeightMatrix,
    w_temporal: WeightMatrix,
    output_head: WeightMatrix,
    temporal_head: WeightMatrix,
    output_mastery: MasteryState,
    temporal_mastery: MasteryState,
    labels: Vec<String>,
    seed: u64,
}

impl PhonemeTrainer {
    /// Create a fresh trainer with zeroed output heads.
    pub fn new() -> Self {
        let phase = TrainingPhase::Exposure;
        let config = phase.mastery_config();
        let mut groups = Vec::with_capacity(NUM_GROUPS);

        for group_def in &ACOUSTIC_GROUPS {
            let w_hidden = init_frozen_hidden(26, GROUP_HIDDEN_DIM, group_def.seed);
            let w_temporal = init_frozen_hidden(
                TEMPORAL_FEATURE_DIM,
                TEMPORAL_HIDDEN_DIM,
                group_def.seed + TEMPORAL_SEED_OFFSET,
            );
            let num_classes = group_def.phonemes.len();
            let labels: Vec<String> = group_def.phonemes.iter().map(|s| s.to_string()).collect();

            let output_head = WeightMatrix::zeros(num_classes, GROUP_HIDDEN_DIM);
            let temporal_head = WeightMatrix::zeros(num_classes, TEMPORAL_HIDDEN_DIM);
            let output_mastery = MasteryState::new(
                num_classes * GROUP_HIDDEN_DIM,
                config.clone(),
            );
            let temporal_mastery = MasteryState::new(
                num_classes * TEMPORAL_HIDDEN_DIM,
                config.clone(),
            );

            groups.push(TrainableGroup {
                w_hidden,
                w_temporal,
                output_head,
                temporal_head,
                output_mastery,
                temporal_mastery,
                labels,
                seed: group_def.seed,
            });
        }

        PhonemeTrainer {
            groups,
            total_cycles: 0,
            total_samples: 0,
        }
    }

    /// Resume training from a saved state file.
    pub fn resume(path: &Path) -> Result<Self> {
        let state = mastery_cache::load_training_state(path)
            .map_err(|e| CochleaError::Config(format!("Failed to load training state: {}", e)))?;

        let mut groups = Vec::with_capacity(state.num_groups);
        for (i, gs) in state.groups.into_iter().enumerate() {
            let w_hidden = init_frozen_hidden(26, GROUP_HIDDEN_DIM, ACOUSTIC_GROUPS[i].seed);
            let w_temporal = init_frozen_hidden(
                TEMPORAL_FEATURE_DIM,
                TEMPORAL_HIDDEN_DIM,
                ACOUSTIC_GROUPS[i].seed + TEMPORAL_SEED_OFFSET,
            );
            let labels: Vec<String> = ACOUSTIC_GROUPS[i]
                .phonemes
                .iter()
                .map(|s| s.to_string())
                .collect();

            groups.push(TrainableGroup {
                w_hidden,
                w_temporal,
                output_head: gs.output_head,
                temporal_head: gs.temporal_head,
                output_mastery: gs.output_mastery,
                temporal_mastery: gs.temporal_mastery,
                labels,
                seed: ACOUSTIC_GROUPS[i].seed,
            });
        }

        Ok(PhonemeTrainer {
            groups,
            total_cycles: state.total_cycles,
            total_samples: state.total_samples,
        })
    }

    /// Current training phase (determined by total cycle count).
    pub fn current_phase(&self) -> TrainingPhase {
        TrainingPhase::from_cycle(self.total_cycles)
    }

    /// Total cycles completed so far.
    pub fn total_cycles(&self) -> u64 {
        self.total_cycles
    }

    /// Run one mastery cycle over all training data.
    ///
    /// Cycle-level mastery learning:
    /// 1. Accumulate pressure signals across ALL samples in the cycle
    /// 2. After all samples, threshold-check each weight once
    /// 3. Apply at most 1 transition per weight per cycle (weaken-before-flip)
    /// 4. Decay residual pressure for cross-cycle persistence
    ///
    /// This prevents the oscillation that occurs when per-sample updates
    /// trigger hundreds of transitions per weight per cycle. The net pressure
    /// across the full corpus determines whether a weight transitions —
    /// contradictory signals cancel, consistent signals accumulate.
    pub fn run_cycle(&mut self, data: &TrainingData) -> CycleMetrics {
        let phase = self.current_phase();
        let config = phase.mastery_config();

        // Build global→local index maps for each group.
        let global_to_local: Vec<Vec<Option<usize>>> = ACOUSTIC_GROUPS
            .iter()
            .map(|group_def| {
                PHONEME_LIST
                    .iter()
                    .map(|&ipa| group_def.phonemes.iter().position(|&p| p == ipa))
                    .collect()
            })
            .collect();

        let mut per_group_frame_transitions = Vec::with_capacity(NUM_GROUPS);
        let mut per_group_temporal_transitions = Vec::with_capacity(NUM_GROUPS);
        let mut total_frame_trans = 0u64;
        let mut total_temporal_trans = 0u64;

        for (group_idx, group) in self.groups.iter_mut().enumerate() {
            let num_classes = group.labels.len();
            let g2l = &global_to_local[group_idx];

            // Update mastery config for current phase.
            group.output_mastery.config = config.clone();
            group.temporal_mastery.config = config.clone();

            let frame_weight_count = num_classes * GROUP_HIDDEN_DIM;
            let temporal_weight_count = num_classes * TEMPORAL_HIDDEN_DIM;

            // Cycle-level pressure accumulators (start from persistent pressure).
            let mut frame_pressure = group.output_mastery.pressure.clone();
            let mut temporal_pressure = group.temporal_mastery.pressure.clone();

            // Participation counters for this cycle.
            let mut frame_participation = vec![0u32; frame_weight_count];
            let mut temporal_participation = vec![0u32; temporal_weight_count];

            // --- Phase 1: Accumulate pressure across ALL frame samples ---
            for (mfcc, global_idx) in &data.frame_samples {
                if let Some(local_idx) = g2l[*global_idx] {
                    let input = encode_mfcc(mfcc, 26);
                    let hidden = forward_hidden_packed(&group.w_hidden, &input);
                    let logits = group.output_head.matmul(&hidden);

                    accumulate_pressure(
                        &hidden,
                        &logits,
                        local_idx,
                        num_classes,
                        &mut frame_pressure,
                        &mut frame_participation,
                    );
                }
            }

            // --- Phase 1: Accumulate pressure across ALL temporal samples ---
            for (features, global_idx) in &data.temporal_samples {
                if let Some(local_idx) = g2l[*global_idx] {
                    let input = encode_mfcc(features, TEMPORAL_FEATURE_DIM);
                    let hidden = forward_hidden_packed(&group.w_temporal, &input);
                    let logits = group.temporal_head.matmul(&hidden);

                    accumulate_pressure(
                        &hidden,
                        &logits,
                        local_idx,
                        num_classes,
                        &mut temporal_pressure,
                        &mut temporal_participation,
                    );
                }
            }

            // --- Phase 2: Threshold check + transitions (at most 1 per weight) ---
            let frame_trans = apply_cycle_transitions(
                &mut group.output_head,
                &mut frame_pressure,
                &frame_participation,
                &config,
            );

            let temporal_trans = apply_cycle_transitions(
                &mut group.temporal_head,
                &mut temporal_pressure,
                &temporal_participation,
                &config,
            );

            // --- Phase 3: Store residual pressure + decay ---
            // Pressure that didn't reach threshold persists to next cycle.
            for (i, p) in frame_pressure.iter().enumerate() {
                if frame_participation[i] >= config.participation_gate {
                    group.output_mastery.pressure[i] = *p;
                }
                // Below participation gate: pressure unchanged from prior cycle.
            }
            for (i, p) in temporal_pressure.iter().enumerate() {
                if temporal_participation[i] >= config.participation_gate {
                    group.temporal_mastery.pressure[i] = *p;
                }
            }

            group.output_mastery.decay();
            group.temporal_mastery.decay();

            group.output_mastery.transitions += frame_trans;
            group.temporal_mastery.transitions += temporal_trans;

            total_frame_trans += frame_trans;
            total_temporal_trans += temporal_trans;

            per_group_frame_transitions.push((
                ACOUSTIC_GROUPS[group_idx].name.to_string(),
                frame_trans,
            ));
            per_group_temporal_transitions.push((
                ACOUSTIC_GROUPS[group_idx].name.to_string(),
                temporal_trans,
            ));
        }

        self.total_cycles += 1;
        self.total_samples += (data.frame_samples.len() + data.temporal_samples.len()) as u64;

        CycleMetrics {
            cycle: self.total_cycles,
            phase,
            total_frame_transitions: total_frame_trans,
            total_temporal_transitions: total_temporal_trans,
            per_group_frame_transitions,
            per_group_temporal_transitions,
            frame_samples_processed: data.frame_samples.len(),
            temporal_samples_processed: data.temporal_samples.len(),
        }
    }

    /// Save full training state (MasteryState + WeightMatrix per group).
    pub fn save(&self, path: &Path) -> Result<()> {
        let state = self.to_training_state();
        mastery_cache::save_training_state(path, &state)
            .map_err(|e| CochleaError::Config(format!("Failed to save training state: {}", e)))
    }

    /// Export inference-only cache (WeightMatrix bytes, no mastery state).
    pub fn export_inference(&self, path: &Path) -> Result<()> {
        let state = self.to_training_state();
        let seeds: Vec<u64> = ACOUSTIC_GROUPS.iter().map(|g| g.seed).collect();
        mastery_cache::export_inference_cache(path, &state, &seeds)
            .map_err(|e| CochleaError::Config(format!("Failed to export inference cache: {}", e)))
    }

    /// Export trained heads as GroupClassifier instances for inference.
    pub fn export_group_classifiers(&self, input_dim: usize) -> Vec<crate::phoneme_classifier::GroupClassifier> {
        self.groups
            .iter()
            .map(|g| crate::phoneme_classifier::GroupClassifier {
                w_hidden: init_frozen_hidden(input_dim, GROUP_HIDDEN_DIM, g.seed),
                w_temporal: init_frozen_hidden(
                    TEMPORAL_FEATURE_DIM,
                    TEMPORAL_HIDDEN_DIM,
                    g.seed + TEMPORAL_SEED_OFFSET,
                ),
                output_head: g.output_head.clone(),
                temporal_head: g.temporal_head.clone(),
                labels: g.labels.clone(),
            })
            .collect()
    }

    fn to_training_state(&self) -> TrainingState {
        let groups: Vec<TrainableGroupState> = self
            .groups
            .iter()
            .map(|g| TrainableGroupState {
                seed: g.seed,
                num_phonemes: g.labels.len(),
                output_head: g.output_head.clone(),
                temporal_head: g.temporal_head.clone(),
                output_mastery: g.output_mastery.clone(),
                temporal_mastery: g.temporal_mastery.clone(),
            })
            .collect();

        TrainingState {
            groups,
            phase: self.current_phase(),
            total_cycles: self.total_cycles,
            total_samples: self.total_samples,
            num_groups: NUM_GROUPS,
            group_hidden_dim: GROUP_HIDDEN_DIM,
            temporal_hidden_dim: TEMPORAL_HIDDEN_DIM,
            total_phonemes: NUM_PHONEMES,
        }
    }
}

// ---------------------------------------------------------------------------
// Cycle-level mastery helpers
// ---------------------------------------------------------------------------

/// Accumulate pressure signals from one sample into cycle-level accumulators.
///
/// Same pressure formula as MasteryState::update() (direction × input_sign ×
/// activity_strength × error_mag), but accumulates into external vectors
/// instead of triggering immediate transitions.
fn accumulate_pressure(
    hidden: &[PackedSignal],
    logits: &[PackedSignal],
    correct_class: usize,
    num_classes: usize,
    pressure: &mut [i32],
    participation: &mut [u32],
) {
    let hidden_dim = hidden.len();

    // Activity threshold: top 25% of hidden activations.
    let max_input = hidden
        .iter()
        .map(|s| s.current().unsigned_abs())
        .max()
        .unwrap_or(1)
        .max(1);
    let activity_threshold = max_input / 4;

    for i in 0..num_classes {
        let target_val: i64 = if i == correct_class { 128 } else { -32 };
        let error = target_val - logits[i].current() as i64;
        if error == 0 {
            continue;
        }
        let direction = error.signum() as i32;

        for j in 0..hidden_dim {
            let w_idx = i * hidden_dim + j;
            let input_abs = hidden[j].current().unsigned_abs();
            let input_sign = (hidden[j].current() as i64).signum() as i32;

            if input_abs > 0 {
                participation[w_idx] = participation[w_idx].saturating_add(1);
            }

            if input_abs <= activity_threshold {
                continue;
            }

            let activity_strength = ((input_abs - activity_threshold) as i64 * 15
                / max_input as i64)
                .max(1) as i32;
            let error_mag = ((error.abs().min(127) as i32) + 31) / 32;

            pressure[w_idx] += direction * input_sign * activity_strength * error_mag;
        }
    }
}

/// Representable positive magnitudes in PackedSignal.
const REPR_LEVELS: &[i32] = &[
    0, 1, 4, 16, 32, 64, 128, 255, 256, 512, 1020, 1024, 2048, 4080, 4096,
    8160, 8192, 16320, 16384, 32640, 32768, 65025,
];

/// Step current value up to next representable magnitude.
fn step_up(current: i32) -> i32 {
    let abs = current.unsigned_abs() as i32;
    for &level in REPR_LEVELS {
        if level > abs {
            return if current >= 0 { level } else { -level };
        }
    }
    current
}

/// Step current value down to previous representable magnitude.
fn step_down(current: i32) -> i32 {
    let abs = current.unsigned_abs() as i32;
    let mut prev = 0i32;
    for &level in REPR_LEVELS {
        if level >= abs {
            break;
        }
        prev = level;
    }
    if current > 0 { prev } else if current < 0 { -prev } else { 0 }
}

/// Apply cycle-level transitions: check accumulated pressure against threshold,
/// apply at most 1 transition per weight using weaken-before-flip.
///
/// After a transition, the threshold is subtracted from the pressure (draining
/// the "spent" portion). Residual pressure carries to the next cycle — a strong
/// consensus signal resolves over multiple cycles rather than being wasted.
///
/// Returns the number of transitions applied.
fn apply_cycle_transitions(
    weights: &mut WeightMatrix,
    pressure: &mut [i32],
    participation: &[u32],
    config: &MasteryConfig,
) -> u64 {
    let mut transitions = 0u64;

    for w_idx in 0..pressure.len() {
        // Participation gate: weight must have been active in enough samples.
        if participation[w_idx] < config.participation_gate {
            continue;
        }

        let p = pressure[w_idx];

        // Threshold gate: net pressure must exceed threshold.
        if p.abs() < config.pressure_threshold {
            continue;
        }

        let needed_direction = p.signum();
        let current = weights.data[w_idx].current();
        let current_sign = if current > 0 { 1 } else if current < 0 { -1 } else { 0 };

        if current_sign == needed_direction {
            // Polarity matches: strengthen.
            let stepped = step_up(current);
            if stepped != current {
                weights.data[w_idx] = packed_from_current(stepped);
                transitions += 1;
            }
        } else if current_sign == -needed_direction {
            // Polarity opposes: weaken first.
            let stepped = step_down(current);
            weights.data[w_idx] = packed_from_current(stepped);
            transitions += 1;
        } else {
            // Zero: set initial polarity.
            let initial = if needed_direction > 0 { 1 } else { -1 };
            weights.data[w_idx] = packed_from_current(initial);
            transitions += 1;
        }

        // Drain spent pressure: subtract threshold in the direction of transition.
        pressure[w_idx] -= needed_direction * config.pressure_threshold;
    }

    transitions
}

// ---------------------------------------------------------------------------
// Training data generation
// ---------------------------------------------------------------------------

/// Load all training data from LibriSpeech spool + oscillator synthesis.
pub fn load_training_data(
    spool_path: &Path,
    dict_path: &Path,
    mfcc_config: &MfccConfig,
) -> Result<TrainingData> {
    let frame_samples = generate_frame_training_data(mfcc_config)?;

    let temporal_samples = generate_temporal_from_librispeech(spool_path, dict_path)?;

    log::info!(
        "Training data loaded: {} frame samples, {} temporal samples",
        frame_samples.len(),
        temporal_samples.len()
    );

    Ok(TrainingData {
        frame_samples,
        temporal_samples,
    })
}

/// Generate single-frame training data from oscillator synthesis (clean labels).
pub(crate) fn generate_frame_training_data(mfcc_config: &MfccConfig) -> Result<Vec<(Vec<f32>, usize)>> {
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
            "No frame training samples generated".to_string(),
        ));
    }

    log::info!(
        "Generated {} frame training samples from {} phonemes",
        samples.len(),
        PHONEME_LIST.len()
    );

    Ok(samples)
}

/// Generate temporal training data from oscillator synthesis (for bootstrap/tests).
pub(crate) fn generate_temporal_from_oscillator(mfcc_config: &MfccConfig) -> Result<Vec<(Vec<f32>, usize)>> {
    let extractor = MfccExtractor::new(mfcc_config.clone());
    let f0_values = [100.0, 130.0, 160.0, 200.0, 250.0];
    let speech_rates = [0.8, 1.0, 1.2];
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

            let mut acc = TemporalAccumulator::new();
            for frame in &frames {
                acc.write_frame(frame);
            }
            let features = acc.extract_features();
            samples.push((features, phoneme_idx));
            }
        }
    }

    if samples.is_empty() {
        return Err(CochleaError::Config(
            "No temporal training samples generated from oscillator".to_string(),
        ));
    }

    Ok(samples)
}

/// Generate temporal training data from LibriSpeech spool + CMUdict alignment.
///
/// Uses the FULL corpus — all 2,703 utterances, all aligned frames.
/// No subsampling. The mastery learning handles noise through sustained
/// pressure accumulation across cycles.
fn generate_temporal_from_librispeech(
    spool_path: &Path,
    dict_path: &Path,
) -> Result<Vec<(Vec<f32>, usize)>> {
    // Load word→IPA map.
    let word_map = cmudict::load_word_to_ipa_map(dict_path)
        .map_err(|e| CochleaError::Config(format!("CMUdict: {}", e)))?;

    // Build phoneme→global_index map.
    let phoneme_to_idx: std::collections::HashMap<&str, usize> = PHONEME_LIST
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();

    // Read full spool.
    let speech_samples = librispeech::read_spool(spool_path)
        .map_err(|e| CochleaError::Config(format!("Spool: {}", e)))?;

    log::info!(
        "Processing {} LibriSpeech utterances for temporal training (full corpus)",
        speech_samples.len()
    );

    let mut samples = Vec::new();
    let mut aligned_count = 0u32;

    for speech in &speech_samples {
        if speech.frames.is_empty() || speech.transcript.is_empty() {
            continue;
        }

        let aligned = align_transcript_to_frames(
            &speech.transcript,
            &speech.frames,
            &word_map,
            &phoneme_to_idx,
        );

        if aligned.is_empty() {
            continue;
        }
        aligned_count += 1;

        // Feed ALL frames through TemporalAccumulator, collect labeled ones.
        let mut acc = TemporalAccumulator::new();
        for (frame_idx, frame) in speech.frames.iter().enumerate() {
            acc.write_frame(frame);

            if let Some(&phoneme_idx) = aligned.get(&frame_idx) {
                let temporal_features = acc.extract_features();
                samples.push((temporal_features, phoneme_idx));
            }
        }
    }

    log::info!(
        "LibriSpeech temporal: {} samples from {} aligned utterances ({} total)",
        samples.len(),
        aligned_count,
        speech_samples.len()
    );

    if samples.is_empty() {
        return Err(CochleaError::Config(
            "No temporal samples from LibriSpeech".to_string(),
        ));
    }

    Ok(samples)
}

/// Align a transcript's phonemes to MFCC frame indices.
///
/// Proportional forced alignment: splits transcript into words, looks up
/// each word's IPA pronunciation in CMUdict, then distributes phonemes
/// evenly across the available frames.
///
/// Returns a map of frame_index → global_phoneme_index for the center
/// frame of each phoneme's span.
fn align_transcript_to_frames(
    transcript: &str,
    frames: &[Vec<f32>],
    word_map: &std::collections::HashMap<String, Vec<&'static str>>,
    phoneme_to_idx: &std::collections::HashMap<&str, usize>,
) -> std::collections::HashMap<usize, usize> {
    let mut result = std::collections::HashMap::new();

    let words: Vec<&str> = transcript.split_whitespace().collect();
    let mut phoneme_sequence: Vec<&str> = Vec::new();

    for word in &words {
        let lower = word.to_lowercase();
        let clean: String = lower.chars().filter(|c| c.is_alphabetic()).collect();
        if clean.is_empty() {
            continue;
        }
        if let Some(ipa_phones) = word_map.get(&clean) {
            phoneme_sequence.extend(ipa_phones.iter());
        }
    }

    if phoneme_sequence.is_empty() || frames.is_empty() {
        return result;
    }

    let n_phonemes = phoneme_sequence.len();
    let n_frames = frames.len();

    // Skip first and last 5% (likely silence).
    let margin = n_frames / 20;
    let active_start = margin;
    let active_end = n_frames.saturating_sub(margin).max(active_start + 1);
    let active_frames = active_end - active_start;

    if active_frames < n_phonemes {
        return result;
    }

    let frames_per_phoneme = active_frames as f32 / n_phonemes as f32;

    for (ph_idx, &phoneme) in phoneme_sequence.iter().enumerate() {
        if let Some(&global_idx) = phoneme_to_idx.get(phoneme) {
            let span_start = active_start + (ph_idx as f32 * frames_per_phoneme) as usize;
            let span_end = active_start + ((ph_idx + 1) as f32 * frames_per_phoneme) as usize;
            let span_width = span_end.saturating_sub(span_start);

            if span_width < MIN_PHONEME_SPAN {
                continue;
            }

            // Only label the center frame.
            let mid = (span_start + span_end) / 2;
            if mid < n_frames {
                let energy = frames[mid][0].abs();
                if energy >= MIN_FRAME_ENERGY {
                    result.insert(mid, global_idx);
                }
            }
        }
    }

    result
}
