//! Persistence for phoneme classifier training state.
//!
//! Two file types:
//! - **Training state** (`phoneme_training_state.bin`): full MasteryState + WeightMatrix
//!   per group, with phase metadata. Allows resuming training across sessions.
//! - **Inference cache** (`phoneme_ensemble_heads.bin`): WeightMatrix bytes only.
//!   Loaded by PhonemeClassifier for inference without any training overhead.

use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use atomic_neural_transistors::WeightMatrix;
use std::io::{self, Read, Write};
use std::path::Path;

/// Training state file version.
const TRAINING_STATE_VERSION: u32 = 1;

/// Inference cache file version.
pub(crate) const INFERENCE_CACHE_VERSION: u32 = 9;

/// Training phase progression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingPhase {
    /// Cycles 1-20: low thresholds, establish initial polarities.
    Exposure = 0,
    /// Cycles 21-50: moderate thresholds, contradictory pressure cancels.
    Refinement = 1,
    /// Cycles 51+: high thresholds, lock in final synaptic strengths.
    Stabilization = 2,
}

impl TrainingPhase {
    pub fn from_cycle(cycle: u64) -> Self {
        if cycle < 20 {
            TrainingPhase::Exposure
        } else if cycle < 50 {
            TrainingPhase::Refinement
        } else {
            TrainingPhase::Stabilization
        }
    }

    fn from_u32(v: u32) -> Self {
        match v {
            0 => TrainingPhase::Exposure,
            1 => TrainingPhase::Refinement,
            _ => TrainingPhase::Stabilization,
        }
    }

    /// MasteryConfig for this phase.
    ///
    /// Thresholds are scaled for corpus-level learning (tens of thousands of
    /// samples per cycle). MasteryState::update() accumulates pressure scaled
    /// by activity_strength (up to 15) × error_mag (up to 4), so a single
    /// sample can contribute up to ±60 pressure. Thresholds must be high
    /// enough that transitions require consistent net signal across many
    /// samples, not just a few strong ones.
    pub fn mastery_config(&self) -> MasteryConfig {
        match self {
            TrainingPhase::Exposure => MasteryConfig {
                pressure_threshold: 200,
                decay_rate: 20,
                participation_gate: 50,
            },
            TrainingPhase::Refinement => MasteryConfig {
                pressure_threshold: 500,
                decay_rate: 50,
                participation_gate: 100,
            },
            TrainingPhase::Stabilization => MasteryConfig {
                pressure_threshold: 1000,
                decay_rate: 100,
                participation_gate: 200,
            },
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TrainingPhase::Exposure => "Exposure",
            TrainingPhase::Refinement => "Refinement",
            TrainingPhase::Stabilization => "Stabilization",
        }
    }
}

/// Training state header.
struct TrainingHeader {
    version: u32,
    num_groups: u32,
    group_hidden_dim: u32,
    temporal_hidden_dim: u32,
    total_phonemes: u32,
    phase: u32,
    total_cycles: u64,
    total_samples: u64,
}

impl TrainingHeader {
    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&self.num_groups.to_le_bytes())?;
        w.write_all(&self.group_hidden_dim.to_le_bytes())?;
        w.write_all(&self.temporal_hidden_dim.to_le_bytes())?;
        w.write_all(&self.total_phonemes.to_le_bytes())?;
        w.write_all(&self.phase.to_le_bytes())?;
        w.write_all(&self.total_cycles.to_le_bytes())?;
        w.write_all(&self.total_samples.to_le_bytes())?;
        Ok(())
    }

    fn read_from(r: &mut impl Read) -> io::Result<Self> {
        let mut buf = [0u8; 40]; // 5×u32 + 1×u32 + 2×u64 = 40 bytes
        r.read_exact(&mut buf)?;
        Ok(TrainingHeader {
            version: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            num_groups: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            group_hidden_dim: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            temporal_hidden_dim: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            total_phonemes: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
            phase: u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]),
            total_cycles: u64::from_le_bytes([
                buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
            ]),
            total_samples: u64::from_le_bytes([
                buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
            ]),
        })
    }
}

/// Serialize a MasteryState to bytes.
fn serialize_mastery_state(state: &MasteryState, w: &mut impl Write) -> io::Result<()> {
    let weight_count = state.pressure.len() as u32;
    w.write_all(&weight_count.to_le_bytes())?;

    // Pressure (i32 per weight).
    for &p in &state.pressure {
        w.write_all(&p.to_le_bytes())?;
    }

    // Participation (u32 per weight).
    for &p in &state.participation {
        w.write_all(&p.to_le_bytes())?;
    }

    // Config.
    w.write_all(&state.config.pressure_threshold.to_le_bytes())?;
    w.write_all(&state.config.decay_rate.to_le_bytes())?;
    w.write_all(&state.config.participation_gate.to_le_bytes())?;

    // Counters.
    w.write_all(&state.steps.to_le_bytes())?;
    w.write_all(&state.transitions.to_le_bytes())?;

    Ok(())
}

/// Deserialize a MasteryState from bytes.
fn deserialize_mastery_state(r: &mut impl Read) -> io::Result<MasteryState> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let weight_count = u32::from_le_bytes(buf4) as usize;

    // Pressure.
    let mut pressure = Vec::with_capacity(weight_count);
    for _ in 0..weight_count {
        r.read_exact(&mut buf4)?;
        pressure.push(i32::from_le_bytes(buf4));
    }

    // Participation.
    let mut participation = Vec::with_capacity(weight_count);
    for _ in 0..weight_count {
        r.read_exact(&mut buf4)?;
        participation.push(u32::from_le_bytes(buf4));
    }

    // Config.
    r.read_exact(&mut buf4)?;
    let pressure_threshold = i32::from_le_bytes(buf4);
    r.read_exact(&mut buf4)?;
    let decay_rate = i32::from_le_bytes(buf4);
    r.read_exact(&mut buf4)?;
    let participation_gate = u32::from_le_bytes(buf4);

    // Counters.
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8)?;
    let steps = u64::from_le_bytes(buf8);
    r.read_exact(&mut buf8)?;
    let transitions = u64::from_le_bytes(buf8);

    Ok(MasteryState {
        pressure,
        participation,
        config: MasteryConfig {
            pressure_threshold,
            decay_rate,
            participation_gate,
        },
        steps,
        transitions,
    })
}

/// Write a len-prefixed WeightMatrix.
fn write_weight_matrix(wm: &WeightMatrix, w: &mut impl Write) -> io::Result<()> {
    let bytes = wm.to_bytes();
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(&bytes)?;
    Ok(())
}

/// Read a len-prefixed WeightMatrix.
fn read_weight_matrix(r: &mut impl Read) -> io::Result<WeightMatrix> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let len = u32::from_le_bytes(buf4) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    WeightMatrix::from_bytes(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Data for one trainable group (everything that needs to persist).
pub struct TrainableGroupState {
    pub seed: u64,
    pub num_phonemes: usize,
    pub output_head: WeightMatrix,
    pub temporal_head: WeightMatrix,
    pub output_mastery: MasteryState,
    pub temporal_mastery: MasteryState,
}

/// Full training state.
pub struct TrainingState {
    pub groups: Vec<TrainableGroupState>,
    pub phase: TrainingPhase,
    pub total_cycles: u64,
    pub total_samples: u64,
    pub num_groups: usize,
    pub group_hidden_dim: usize,
    pub temporal_hidden_dim: usize,
    pub total_phonemes: usize,
}

/// Save full training state (MasteryState + WeightMatrix per group).
pub fn save_training_state(path: &Path, state: &TrainingState) -> io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    let header = TrainingHeader {
        version: TRAINING_STATE_VERSION,
        num_groups: state.num_groups as u32,
        group_hidden_dim: state.group_hidden_dim as u32,
        temporal_hidden_dim: state.temporal_hidden_dim as u32,
        total_phonemes: state.total_phonemes as u32,
        phase: state.phase as u32,
        total_cycles: state.total_cycles,
        total_samples: state.total_samples,
    };
    header.write_to(&mut file)?;

    for group in &state.groups {
        file.write_all(&(group.seed as u32).to_le_bytes())?;
        file.write_all(&(group.num_phonemes as u32).to_le_bytes())?;

        write_weight_matrix(&group.output_head, &mut file)?;
        write_weight_matrix(&group.temporal_head, &mut file)?;
        serialize_mastery_state(&group.output_mastery, &mut file)?;
        serialize_mastery_state(&group.temporal_mastery, &mut file)?;
    }

    Ok(())
}

/// Load full training state.
pub fn load_training_state(path: &Path) -> io::Result<TrainingState> {
    let mut file = std::fs::File::open(path)?;
    let header = TrainingHeader::read_from(&mut file)?;

    if header.version != TRAINING_STATE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "training state version {} != expected {}",
                header.version, TRAINING_STATE_VERSION
            ),
        ));
    }

    let mut groups = Vec::with_capacity(header.num_groups as usize);
    for _ in 0..header.num_groups {
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let seed = u32::from_le_bytes([buf8[0], buf8[1], buf8[2], buf8[3]]) as u64;
        let num_phonemes = u32::from_le_bytes([buf8[4], buf8[5], buf8[6], buf8[7]]) as usize;

        let output_head = read_weight_matrix(&mut file)?;
        let temporal_head = read_weight_matrix(&mut file)?;
        let output_mastery = deserialize_mastery_state(&mut file)?;
        let temporal_mastery = deserialize_mastery_state(&mut file)?;

        groups.push(TrainableGroupState {
            seed,
            num_phonemes,
            output_head,
            temporal_head,
            output_mastery,
            temporal_mastery,
        });
    }

    Ok(TrainingState {
        groups,
        phase: TrainingPhase::from_u32(header.phase),
        total_cycles: header.total_cycles,
        total_samples: header.total_samples,
        num_groups: header.num_groups as usize,
        group_hidden_dim: header.group_hidden_dim as usize,
        temporal_hidden_dim: header.temporal_hidden_dim as usize,
        total_phonemes: header.total_phonemes as usize,
    })
}

/// Export inference-only cache (WeightMatrix bytes, no mastery state).
/// This is what PhonemeClassifier::build() loads for inference.
pub fn export_inference_cache(
    path: &Path,
    state: &TrainingState,
    acoustic_group_seeds: &[u64],
) -> io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    // Header: version, num_groups, hidden_dim, temporal_hidden_dim, total_phonemes.
    file.write_all(&INFERENCE_CACHE_VERSION.to_le_bytes())?;
    file.write_all(&(state.num_groups as u32).to_le_bytes())?;
    file.write_all(&(state.group_hidden_dim as u32).to_le_bytes())?;
    file.write_all(&(state.temporal_hidden_dim as u32).to_le_bytes())?;
    file.write_all(&(state.total_phonemes as u32).to_le_bytes())?;

    // Per group: seed, num_phonemes, output_head bytes, temporal_head bytes.
    for (i, group) in state.groups.iter().enumerate() {
        let seed = acoustic_group_seeds[i] as u32;
        file.write_all(&seed.to_le_bytes())?;
        file.write_all(&(group.num_phonemes as u32).to_le_bytes())?;

        write_weight_matrix(&group.output_head, &mut file)?;
        write_weight_matrix(&group.temporal_head, &mut file)?;
    }

    Ok(())
}
