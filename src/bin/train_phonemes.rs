//! Dedicated training binary for phoneme classifier output heads.
//!
//! Runs sustained mastery learning over LibriSpeech data, checkpointing
//! after every cycle. Exports inference-only cache when done.
//!
//! Usage:
//!   cargo run --release --bin train-phonemes -- --cycles 20 --export
//!   cargo run --release --bin train-phonemes -- --cycles 30  # resumes

use clap::Parser;
use cochlea_rs::config::TranscriptionConfig;
use cochlea_rs::training::{self, PhonemeTrainer};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "train-phonemes")]
#[command(about = "Sustained mastery learning for phoneme classifier output heads")]
struct Cli {
    /// Path to dev-clean.spool.
    #[arg(long, default_value = "../hush/data/dev-clean.spool")]
    spool: PathBuf,

    /// Path to CMUdict pronunciation dictionary.
    #[arg(long, default_value = "data/cmudict.dict")]
    dict: PathBuf,

    /// Directory for training state and inference cache files.
    #[arg(long, default_value = "data")]
    cache_dir: PathBuf,

    /// Number of mastery cycles to run this session.
    #[arg(long, default_value = "10")]
    cycles: u64,

    /// Export inference-only cache after training.
    #[arg(long)]
    export: bool,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let state_path = cli.cache_dir.join("phoneme_training_state.bin");
    let inference_path = cli.cache_dir.join("phoneme_ensemble_heads.bin");

    // Resume or create trainer.
    let mut trainer = if state_path.exists() {
        eprintln!("Resuming from saved training state: {}", state_path.display());
        match PhonemeTrainer::resume(&state_path) {
            Ok(t) => {
                eprintln!(
                    "  {} cycles completed, phase: {}",
                    t.total_cycles(),
                    t.current_phase().name()
                );
                t
            }
            Err(e) => {
                eprintln!("Error loading training state: {}", e);
                eprintln!("Starting fresh.");
                PhonemeTrainer::new()
            }
        }
    } else {
        eprintln!("No saved state found. Starting fresh training.");
        PhonemeTrainer::new()
    };

    // Load training data.
    let config = TranscriptionConfig::wideband();
    eprintln!("Loading training data...");
    eprintln!("  Spool: {}", cli.spool.display());
    eprintln!("  Dict:  {}", cli.dict.display());

    let data = match training::load_training_data(&cli.spool, &cli.dict, &config.mfcc) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading training data: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!(
        "  {} frame samples, {} temporal samples",
        data.frame_samples.len(),
        data.temporal_samples.len()
    );

    // Run mastery cycles.
    let start_cycle = trainer.total_cycles();
    let target_cycle = start_cycle + cli.cycles;

    eprintln!(
        "\nRunning cycles {}-{} (phase: {})",
        start_cycle + 1,
        target_cycle,
        trainer.current_phase().name()
    );
    eprintln!("{}", "=".repeat(72));

    for _ in 0..cli.cycles {
        let metrics = trainer.run_cycle(&data);

        // Print cycle report.
        eprintln!(
            "\nCycle {:>3} | Phase: {:>13} | Frame trans: {:>8} | Temporal trans: {:>8}",
            metrics.cycle,
            metrics.phase.name(),
            metrics.total_frame_transitions,
            metrics.total_temporal_transitions,
        );

        // Per-group breakdown.
        for ((fname, ft), (_, tt)) in metrics
            .per_group_frame_transitions
            .iter()
            .zip(metrics.per_group_temporal_transitions.iter())
        {
            eprintln!("  {:>24}: frame {:>6} | temporal {:>6}", fname, ft, tt);
        }

        // Checkpoint after every cycle.
        if let Err(e) = trainer.save(&state_path) {
            eprintln!("  WARNING: Failed to save checkpoint: {}", e);
        }
    }

    eprintln!("\n{}", "=".repeat(72));
    eprintln!(
        "Training complete: {} total cycles, phase: {}",
        trainer.total_cycles(),
        trainer.current_phase().name()
    );

    // Export inference cache.
    if cli.export {
        eprintln!("Exporting inference cache: {}", inference_path.display());
        if let Err(e) = trainer.export_inference(&inference_path) {
            eprintln!("Error exporting inference cache: {}", e);
            std::process::exit(1);
        }
        eprintln!("Inference cache exported.");
    } else {
        eprintln!(
            "Run with --export to generate inference cache at {}",
            inference_path.display()
        );
    }
}
