//! Evaluate cochlea's temporal classifier on LibriSpeech spool data.
//!
//! Reads pre-extracted MFCC frames directly from hush's dev-clean.spool,
//! feeds them through the full pipeline (CMN → temporal classification →
//! sequencing → word matching), and compares output against the stored
//! transcripts.

use clap::Parser;
use cochlea_rs::cmudict;
use cochlea_rs::config::TranscriptionConfig;
use cochlea_rs::librispeech;
use cochlea_rs::Transcriber;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "eval-librispeech")]
#[command(about = "Evaluate temporal classifier on LibriSpeech spool samples")]
struct Cli {
    /// Path to dev-clean.spool.
    #[arg(long, default_value = "../hush/data/dev-clean.spool")]
    spool: PathBuf,

    /// Path to CMUdict pronunciation dictionary.
    #[arg(long, default_value = "data/cmudict.dict")]
    dict: PathBuf,

    /// Number of samples to evaluate (0 = all).
    #[arg(long, default_value = "10")]
    count: usize,

    /// Skip first N samples.
    #[arg(long, default_value = "0")]
    skip: usize,

    /// Show phoneme events for each sample.
    #[arg(long)]
    phonemes: bool,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    // Read spool.
    eprintln!("Reading spool: {}", cli.spool.display());
    let samples = librispeech::read_spool(&cli.spool).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    eprintln!("Loaded {} samples from spool", samples.len());

    // Build transcriber with pronunciation trie.
    let config = TranscriptionConfig::wideband();
    let cache_dir = cli.dict.parent().map(|p| p.to_path_buf());

    let mut transcriber = Transcriber::new(config, cache_dir.as_deref())
        .expect("transcriber should build");

    if let Ok(trie) = cmudict::load_cmudict(&cli.dict) {
        eprintln!("Loaded {} pronunciations", trie.len());
        transcriber = transcriber.with_pronunciation_trie(trie);
    }

    // Evaluate samples.
    let end = if cli.count == 0 {
        samples.len()
    } else {
        (cli.skip + cli.count).min(samples.len())
    };

    let mut total_unique_phonemes = 0usize;
    let mut total_words_output = 0usize;
    let mut n_evaluated = 0usize;

    for idx in cli.skip..end {
        let sample = &samples[idx];
        if sample.frames.is_empty() || sample.transcript.is_empty() {
            continue;
        }

        // Feed spool MFCC frames directly through the pipeline.
        let result = match transcriber.transcribe_mfcc_frames(&sample.frames) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[{:>4}] Error: {}", idx, e);
                continue;
            }
        };

        // Count unique phonemes.
        let unique: std::collections::HashSet<&str> =
            result.phoneme_events.iter().map(|e| e.ipa.as_str()).collect();
        total_unique_phonemes += unique.len();
        total_words_output += result.transcript.words.len();
        n_evaluated += 1;

        eprintln!(
            "\n[{:>4}] {} frames, {} speech, {} phoneme events, {} unique phonemes",
            idx,
            result.total_frames,
            result.speech_frames,
            result.phoneme_events.len(),
            unique.len(),
        );
        eprintln!("  EXPECTED:  {}", sample.transcript);
        eprintln!("  GOT:       {}", result.transcript.text);

        // Show phoneme sequence.
        if cli.phonemes {
            for ev in &result.phoneme_events {
                let alts: Vec<&str> = ev.alternatives.iter().map(|(s, _)| s.as_str()).collect();
                eprintln!(
                    "    [{:>4}-{:>4}] /{:<4}/ conf={:.3} alts={:?}",
                    ev.start_frame, ev.end_frame, ev.ipa, ev.confidence, alts,
                );
            }
        }

        // Quick word overlap check.
        let expected_words: std::collections::HashSet<String> = sample
            .transcript
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let got_words: std::collections::HashSet<String> = result
            .transcript
            .words
            .iter()
            .map(|w| w.word.to_lowercase())
            .collect();
        let overlap: Vec<&String> = expected_words.intersection(&got_words).collect();
        if !overlap.is_empty() {
            eprintln!("  MATCHED:   {:?}", overlap);
        }
    }

    eprintln!("\n=== Summary ===");
    eprintln!("Samples evaluated: {}", n_evaluated);
    eprintln!(
        "Avg unique phonemes/sample: {:.1}",
        total_unique_phonemes as f32 / n_evaluated.max(1) as f32
    );
    eprintln!(
        "Avg words output/sample: {:.1}",
        total_words_output as f32 / n_evaluated.max(1) as f32
    );
}
