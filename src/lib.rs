//! # Cochlea
//!
//! Phoneme-level speech transcription via mastery-learned ANT classifier and
//! lexicon pronunciation matching.
//!
//! ## Architecture
//!
//! ```text
//! Audio → [preprocess: CMN, pre-emphasis]
//!       → [MFCC extraction]
//!       → [ANT PhonemeClassifier: mastery-learned MFCC → phoneme]
//!       → [sequencer: frame alignment, merge, duration check]
//!       → [PronunciationTrie: phoneme seq → word candidates]
//!       → [disambiguator: frequency rank + edit distance]
//!       → Transcript
//! ```
//!
//! The classifier uses a frozen random projection (MFCC→64-dim) plus a
//! mastery-learned output head (64→45 phonemes). Trained from oscillator-
//! synthesized phonemes at multiple F0 values and speech rates, with weights
//! cached to a thermogram file for instant reload.

pub mod audio;
pub mod cmudict;
pub mod config;
pub mod disambiguator;
pub mod error;
pub mod formant;
pub mod librispeech;
pub mod mastery_cache;
pub mod mfcc;
pub mod mfcc_templates;
pub mod phoneme_bank;
pub mod phoneme_classifier;
pub mod phoneme_match;
pub mod preprocess;
pub mod pronunciation;
pub mod sequencer;
pub mod temporal_context;
pub mod training;
pub mod transcript;

use crate::config::TranscriptionConfig;
use crate::error::{CochleaError, Result};
use crate::mfcc::MfccExtractor;
use crate::phoneme_classifier::PhonemeClassifier;
use crate::preprocess::{
    cepstral_mean_normalization, energy_vad, estimate_noise_profile, pre_emphasis,
    spectral_subtraction,
};
use crate::pronunciation::PronunciationTrie;
use crate::sequencer::{PhonemeEvent, Sequencer};
use crate::temporal_context::TemporalAccumulator;
use crate::transcript::{Transcript, TranscribedWord};
use std::path::Path;

/// The cochlea speech transcriber.
///
/// Orchestrates the full pipeline: audio → MFCC extraction → ANT phoneme
/// classification → sequencing → word matching → transcript.
///
/// Phoneme recognition is driven by a mastery-learned ANT classifier that
/// maps MFCC frames directly to phoneme candidates.
pub struct Transcriber {
    config: TranscriptionConfig,
    mfcc_extractor: MfccExtractor,
    phoneme_classifier: PhonemeClassifier,
    sequencer: Sequencer,
    pronunciation_trie: Option<PronunciationTrie>,
}

impl Transcriber {
    /// Build a transcriber with the given configuration.
    ///
    /// The pronunciation trie is optional — without it, the transcriber
    /// outputs phoneme-level results only (no word matching).
    ///
    /// `cache_dir` is where the phoneme classifier stores its thermogram.
    /// Pass `None` to train from scratch every time (slower but no disk I/O).
    pub fn new(config: TranscriptionConfig, cache_dir: Option<&Path>) -> Result<Self> {
        let mfcc_extractor = MfccExtractor::new(config.mfcc.clone());
        let phoneme_classifier = PhonemeClassifier::build(&config.mfcc, cache_dir)?;
        let sequencer = Sequencer::new(config.mfcc.frame_ms());

        log::info!(
            "Transcriber initialized: {}Hz, {} phonemes (mastery-learned)",
            config.mfcc.sample_rate,
            phoneme_classifier.len(),
        );

        Ok(Self {
            config,
            mfcc_extractor,
            phoneme_classifier,
            sequencer,
            pronunciation_trie: None,
        })
    }

    /// Build a transcriber with a pronunciation trie for word matching.
    pub fn with_pronunciation_trie(mut self, trie: PronunciationTrie) -> Self {
        log::info!("Pronunciation trie loaded: {} words", trie.len());
        self.pronunciation_trie = Some(trie);
        self
    }

    /// Transcribe an audio file.
    pub fn transcribe_file(&self, path: &Path) -> Result<TranscriptionResult> {
        let (samples, file_sample_rate) = audio::load_audio(path)?;

        // Resample if needed.
        let target_rate = self.config.mfcc.sample_rate;
        let samples = if file_sample_rate != target_rate {
            log::info!(
                "Resampling {}Hz → {}Hz ({} samples)",
                file_sample_rate,
                target_rate,
                samples.len()
            );
            audio::resample(&samples, file_sample_rate, target_rate)
        } else {
            samples
        };

        self.transcribe_samples(&samples)
    }

    /// Transcribe pre-extracted MFCC frames directly (e.g., from a spool file).
    ///
    /// Skips audio loading, pre-emphasis, and MFCC extraction — starts from
    /// noise estimation onward.
    pub fn transcribe_mfcc_frames(&self, mfcc_frames: &[Vec<f32>]) -> Result<TranscriptionResult> {
        if mfcc_frames.is_empty() {
            return Err(CochleaError::EmptyAudio);
        }

        let mut frames = mfcc_frames.to_vec();

        // Noise estimation and spectral subtraction.
        if self.config.noise_estimation_frames > 0 {
            let noise = estimate_noise_profile(&frames, self.config.noise_estimation_frames);
            spectral_subtraction(&mut frames, &noise);
        }

        // CMN.
        if self.config.cmn_enabled {
            cepstral_mean_normalization(&mut frames);
        }

        // VAD.
        let vad = adaptive_vad(&frames, self.config.vad_threshold);
        let speech_frames = vad.iter().filter(|&&v| v).count();

        // Temporal classification.
        let mut accumulator = TemporalAccumulator::new();
        let top_k = self.config.phoneme_top_k;
        let frame_candidates: Vec<_> = frames
            .iter()
            .map(|frame| {
                accumulator.write_frame(frame);
                let temporal_features = accumulator.extract_features();
                self.phoneme_classifier.classify_temporal(&temporal_features, top_k)
            })
            .collect();

        let vad = &vad[..frame_candidates.len().min(vad.len())];

        // Sequence phonemes.
        let phoneme_events = self.sequencer.sequence(&frame_candidates, vad);

        // Word matching.
        let transcript = if let Some(ref trie) = self.pronunciation_trie {
            self.match_words(&phoneme_events, trie)
        } else {
            self.phoneme_transcript(&phoneme_events)
        };

        Ok(TranscriptionResult {
            transcript,
            phoneme_events,
            total_frames: frames.len(),
            speech_frames,
        })
    }

    /// Transcribe raw audio samples (must match the configured sample rate).
    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<TranscriptionResult> {
        if samples.is_empty() {
            return Err(CochleaError::EmptyAudio);
        }

        // Step 1: Pre-emphasis.
        let mut audio = samples.to_vec();
        if self.config.pre_emphasis_coeff > 0.0 {
            pre_emphasis(&mut audio, self.config.pre_emphasis_coeff);
        }

        // Step 2: MFCC extraction.
        let mut mfcc_frames = self.mfcc_extractor.extract(&audio);
        if mfcc_frames.is_empty() {
            return Err(CochleaError::EmptyAudio);
        }
        log::info!("Extracted {} MFCC frames", mfcc_frames.len());

        // Step 3: Noise estimation and spectral subtraction.
        if self.config.noise_estimation_frames > 0 {
            let noise_profile =
                estimate_noise_profile(&mfcc_frames, self.config.noise_estimation_frames);
            spectral_subtraction(&mut mfcc_frames, &noise_profile);
        }

        // Step 4: Cepstral mean normalization.
        if self.config.cmn_enabled {
            cepstral_mean_normalization(&mut mfcc_frames);
        }

        // Step 5: VAD — adaptive threshold based on actual energy distribution.
        let vad = adaptive_vad(&mfcc_frames, self.config.vad_threshold);
        let speech_frames = vad.iter().filter(|&&v| v).count();
        log::info!(
            "VAD: {}/{} frames are speech",
            speech_frames,
            mfcc_frames.len()
        );

        // Step 6: Temporal phoneme classification.
        // Accumulate MFCC frames in a TemporalField (no decay) and classify
        // the temporal feature vector (current + mean + delta = 78 dims).
        let mut accumulator = TemporalAccumulator::new();
        let top_k = self.config.phoneme_top_k;
        let frame_candidates: Vec<_> = mfcc_frames
            .iter()
            .map(|frame| {
                accumulator.write_frame(frame);
                let temporal_features = accumulator.extract_features();
                self.phoneme_classifier.classify_temporal(&temporal_features, top_k)
            })
            .collect();
        log::info!(
            "Classified {} MFCC frames into phoneme candidates",
            frame_candidates.len()
        );

        let vad = &vad[..frame_candidates.len().min(vad.len())];

        // Step 7: Sequence phonemes.
        let phoneme_events = self.sequencer.sequence(&frame_candidates, vad);
        log::info!("Sequenced {} phoneme events", phoneme_events.len());

        // Step 8: Word matching (if trie is available).
        let transcript = if let Some(ref trie) = self.pronunciation_trie {
            self.match_words(&phoneme_events, trie)
        } else {
            // No trie — produce phoneme-level transcript.
            self.phoneme_transcript(&phoneme_events)
        };

        Ok(TranscriptionResult {
            transcript,
            phoneme_events,
            total_frames: mfcc_frames.len(),
            speech_frames,
        })
    }

    /// Match phoneme events to words using the pronunciation trie.
    ///
    /// First segments events into word-sized groups using frame gaps (silence),
    /// then matches each segment against the trie.
    fn match_words(
        &self,
        events: &[PhonemeEvent],
        trie: &PronunciationTrie,
    ) -> Transcript {
        if events.is_empty() {
            return Transcript::empty();
        }

        // Segment phoneme events into word groups based on frame gaps.
        let segments = segment_into_words(events);
        log::info!("Segmented {} phoneme events into {} word groups", events.len(), segments.len());

        let frame_ms = self.sequencer.frame_ms();
        let max_edits = self.config.max_edit_distance;
        let mut words = Vec::new();

        for segment_refs in &segments {
            if segment_refs.is_empty() {
                continue;
            }

            // Clone segment into owned events for disambiguator API.
            let segment: Vec<PhonemeEvent> =
                segment_refs.iter().map(|e| (*e).clone()).collect();

            // Greedy left-to-right word matching within this segment.
            let mut pos = 0;
            while pos < segment.len() {
                let remaining = segment.len() - pos;

                // Try windows from longest to shortest (prefer longer words).
                let max_window = remaining.min(10);
                let mut best: Option<(String, f32, usize)> = None; // (word, score, window_size)

                for window in (2..=max_window).rev() {
                    let phoneme_seq: Vec<String> =
                        segment[pos..pos + window].iter().map(|e| e.ipa.clone()).collect();

                    // Only exact matches for long sequences, fuzzy for short.
                    let fuzzy_edits = if window >= 5 { max_edits } else { max_edits.min(1) };

                    let mut candidates = trie.lookup_exact(&phoneme_seq);
                    if candidates.is_empty() && fuzzy_edits > 0 {
                        candidates = trie.lookup_fuzzy(&phoneme_seq, fuzzy_edits);
                        // Filter: require edit_distance/phoneme_count <= 0.5.
                        candidates.retain(|c| {
                            c.edit_distance as f32 / window as f32 <= 0.5
                        });
                    }

                    if let Some(word) = disambiguator::select_best(
                        &candidates,
                        &segment[pos..pos + window],
                    ) {
                        let avg_conf: f32 = segment[pos..pos + window]
                            .iter()
                            .map(|e| e.confidence)
                            .sum::<f32>()
                            / window as f32;

                        // Score: reward longer matches and higher confidence.
                        let score = avg_conf * (window as f32).sqrt();
                        if best.as_ref().map_or(true, |(_, s, _)| score > *s) {
                            best = Some((word, score, window));
                        }
                    }
                }

                if let Some((word, _, consumed)) = best {
                    let start_ms = (segment[pos].start_frame as f32 * frame_ms) as u32;
                    let end_ms =
                        (segment[pos + consumed - 1].end_frame as f32 * frame_ms) as u32;
                    let confidence: f32 = segment[pos..pos + consumed]
                        .iter()
                        .map(|e| e.confidence)
                        .sum::<f32>()
                        / consumed as f32;

                    words.push(TranscribedWord {
                        word,
                        start_ms,
                        end_ms,
                        confidence,
                        alternatives: Vec::new(),
                    });
                    pos += consumed;
                } else {
                    // No match — skip this phoneme.
                    pos += 1;
                }
            }
        }

        Transcript::from_words(words)
    }

    /// Produce a phoneme-level transcript (no word matching).
    fn phoneme_transcript(&self, events: &[PhonemeEvent]) -> Transcript {
        let frame_ms = self.sequencer.frame_ms();
        let words: Vec<TranscribedWord> = events
            .iter()
            .map(|e| {
                let start_ms = (e.start_frame as f32 * frame_ms) as u32;
                let end_ms = (e.end_frame as f32 * frame_ms) as u32;
                TranscribedWord {
                    word: e.ipa.clone(),
                    start_ms,
                    end_ms,
                    confidence: e.confidence,
                    alternatives: e
                        .alternatives
                        .iter()
                        .map(|(ipa, _)| ipa.clone())
                        .collect(),
                }
            })
            .collect();

        Transcript::from_words(words)
    }
}

/// Full transcription result including diagnostics.
#[derive(Debug)]
pub struct TranscriptionResult {
    /// The transcript (words or phonemes).
    pub transcript: Transcript,
    /// The raw phoneme events (always available).
    pub phoneme_events: Vec<PhonemeEvent>,
    /// Total number of MFCC frames processed.
    pub total_frames: usize,
    /// Number of frames classified as speech by VAD.
    pub speech_frames: usize,
}

/// Adaptive VAD that computes threshold from the energy distribution.
///
/// After CMN, c0 values are centered around 0, so a fixed threshold doesn't work.
/// Instead, compute the energy histogram and set the threshold at the valley
/// between the noise floor and speech energy peaks.
fn adaptive_vad(frames: &[Vec<f32>], fallback_threshold: f32) -> Vec<bool> {
    if frames.is_empty() {
        return Vec::new();
    }

    // Collect c0 (energy) values.
    let energies: Vec<f32> = frames
        .iter()
        .map(|f| f.first().copied().unwrap_or(f32::NEG_INFINITY))
        .collect();

    // Sort to find percentiles.
    let mut sorted = energies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n < 10 {
        return energy_vad(frames, fallback_threshold);
    }

    // Threshold: midpoint between 25th and 75th percentile.
    // Frames below this are silence; above are speech.
    let p25 = sorted[n / 4];
    let p75 = sorted[3 * n / 4];
    let range = p75 - p25;

    // If the range is very small, all frames are similar — use fallback.
    if range < 0.5 {
        return energy_vad(frames, fallback_threshold);
    }

    // Set threshold at p25 + 30% of the range (favors detecting speech over silence).
    let threshold = p25 + range * 0.30;
    log::info!(
        "Adaptive VAD: p25={:.1}, p75={:.1}, threshold={:.1}",
        p25, p75, threshold
    );

    energies.iter().map(|&e| e > threshold).collect()
}

/// Segment phoneme events into word-sized groups based on frame gaps.
///
/// A gap of more than `gap_frames` between consecutive events indicates
/// a word boundary. In continuous speech, inter-word pauses are typically
/// 100-200ms; within-word pauses (e.g., plosive releases) are shorter.
fn segment_into_words(events: &[PhonemeEvent]) -> Vec<Vec<&PhonemeEvent>> {
    if events.is_empty() {
        return Vec::new();
    }

    // Gap threshold: 8+ frames (80ms at 10ms/frame) = word boundary.
    // This is conservative — catches clear pauses but keeps most words whole.
    let gap_threshold = 8;

    let mut segments: Vec<Vec<&PhonemeEvent>> = Vec::new();
    let mut current: Vec<&PhonemeEvent> = vec![&events[0]];

    for i in 1..events.len() {
        let gap = events[i].start_frame.saturating_sub(events[i - 1].end_frame);
        if gap >= gap_threshold {
            // Word boundary found.
            segments.push(current);
            current = Vec::new();
        }
        current.push(&events[i]);
    }
    if !current.is_empty() {
        segments.push(current);
    }

    segments
}
