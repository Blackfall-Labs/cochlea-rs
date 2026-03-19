use clap::Parser;
use cochlea_rs::cmudict;
use cochlea_rs::config::TranscriptionConfig;
use cochlea_rs::Transcriber;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cochlea-transcribe")]
#[command(about = "Phoneme-level speech transcription via mastery-learned ANT classifier")]
struct Cli {
    /// Path to audio file (WAV, FLAC, or MP3).
    #[arg(long)]
    audio: PathBuf,

    /// Path to CMUdict pronunciation dictionary.
    #[arg(long, default_value = "data/cmudict.dict")]
    dict: PathBuf,

    /// Audio quality preset.
    #[arg(long, default_value = "wideband")]
    quality: String,

    /// VAD threshold (log energy). More negative = more permissive.
    #[arg(long)]
    vad_threshold: Option<f32>,

    /// Number of top phoneme candidates per frame.
    #[arg(long)]
    top_k: Option<usize>,

    /// Maximum edit distance for fuzzy word lookup.
    #[arg(long)]
    max_edits: Option<usize>,

    /// Show per-phoneme events (not just transcript).
    #[arg(long)]
    phonemes: bool,

    /// Show timing information for each word/phoneme.
    #[arg(long)]
    timestamps: bool,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let mut config = match cli.quality.as_str() {
        "phone" | "phone_quality" | "8k" => TranscriptionConfig::phone_quality(),
        _ => TranscriptionConfig::wideband(),
    };

    if let Some(vad) = cli.vad_threshold {
        config.vad_threshold = vad;
    }
    if let Some(k) = cli.top_k {
        config.phoneme_top_k = k;
    }
    if let Some(edits) = cli.max_edits {
        config.max_edit_distance = edits;
    }

    eprintln!(
        "cochlea-transcribe: {} quality, {}Hz",
        cli.quality, config.mfcc.sample_rate
    );

    // Load pronunciation dictionary.
    let trie = match cmudict::load_cmudict(&cli.dict) {
        Ok(t) => {
            eprintln!("Loaded {} pronunciations from {}", t.len(), cli.dict.display());
            Some(t)
        }
        Err(e) => {
            eprintln!("Warning: could not load dictionary: {}", e);
            eprintln!("Falling back to phoneme-only output.");
            None
        }
    };

    // Use the dictionary directory as the classifier cache directory.
    let cache_dir = cli.dict.parent().map(|p| p.to_path_buf());
    let mut transcriber = match Transcriber::new(config, cache_dir.as_deref()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error initializing transcriber: {}", e);
            std::process::exit(1);
        }
    };
    if let Some(trie) = trie {
        transcriber = transcriber.with_pronunciation_trie(trie);
    }

    eprintln!("Loading audio: {}", cli.audio.display());
    let result = match transcriber.transcribe_file(&cli.audio) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!(
        "Processed {} frames ({} speech), {} phoneme events",
        result.total_frames, result.speech_frames, result.phoneme_events.len()
    );

    // Diagnostic: show first 20 phoneme events to check for monotony.
    eprintln!("\n--- First 20 phoneme events ---");
    for (i, event) in result.phoneme_events.iter().take(20).enumerate() {
        let alts: Vec<&str> = event.alternatives.iter().map(|(s, _)| s.as_str()).collect();
        eprintln!(
            "  {:>2}. [{:>4}-{:>4}] /{:<4}/ conf={:.3} alts={:?}",
            i, event.start_frame, event.end_frame,
            event.ipa, event.confidence, alts,
        );
    }
    // Count unique phonemes.
    let unique: std::collections::HashSet<&str> =
        result.phoneme_events.iter().map(|e| e.ipa.as_str()).collect();
    eprintln!("Unique phonemes: {} out of {} events", unique.len(), result.phoneme_events.len());

    // Show phoneme events if requested.
    if cli.phonemes {
        eprintln!("\n--- Phoneme Events ---");
        for event in &result.phoneme_events {
            let alts: Vec<&str> = event.alternatives.iter().map(|(s, _)| s.as_str()).collect();
            if cli.timestamps {
                eprintln!(
                    "  [{:>5}ms - {:>5}ms] /{}/  conf={:.2}  alts={:?}",
                    (event.start_frame as f32 * 10.0) as u32,
                    (event.end_frame as f32 * 10.0) as u32,
                    event.ipa,
                    event.confidence,
                    alts,
                );
            } else {
                eprintln!(
                    "  /{}/  conf={:.2}  alts={:?}",
                    event.ipa, event.confidence, alts,
                );
            }
        }
    }

    // Show transcript with timestamps.
    if cli.timestamps && !result.transcript.words.is_empty() {
        eprintln!("\n--- Transcript ---");
        for word in &result.transcript.words {
            eprintln!(
                "  [{:>5}ms - {:>5}ms] \"{}\"  conf={:.2}",
                word.start_ms, word.end_ms, word.word, word.confidence,
            );
        }
    }

    // Always print the transcript text to stdout.
    println!("{}", result.transcript.text);
    eprintln!(
        "\nOverall confidence: {:.2}",
        result.transcript.confidence
    );
}
