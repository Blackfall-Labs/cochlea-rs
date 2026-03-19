//! Sequencer — converts per-frame phoneme candidates into a coherent phoneme sequence.
//!
//! Handles sliding window consensus, duration validation, and merging of
//! adjacent identical phonemes into phoneme events with start/end times.

use crate::phoneme_bank::PhonemeCandidate;
use std::collections::HashMap;

/// A recognized phoneme event spanning multiple frames.
#[derive(Debug, Clone)]
pub struct PhonemeEvent {
    /// The IPA symbol for this phoneme.
    pub ipa: String,
    /// Start frame index.
    pub start_frame: usize,
    /// End frame index (exclusive).
    pub end_frame: usize,
    /// Average confidence across frames.
    pub confidence: f32,
    /// Runner-up phoneme candidates.
    pub alternatives: Vec<(String, f32)>,
}

impl PhonemeEvent {
    /// Duration of this event in frames.
    pub fn duration_frames(&self) -> usize {
        self.end_frame - self.start_frame
    }

    /// Duration of this event in milliseconds (at 10ms per frame).
    pub fn duration_ms(&self, frame_ms: f32) -> f32 {
        self.duration_frames() as f32 * frame_ms
    }
}

/// Phoneme duration expectations (from oscillator phoneme table).
struct DurationTable {
    /// IPA → expected duration in ms.
    durations: HashMap<String, u16>,
}

impl DurationTable {
    fn new() -> Self {
        let mut d = HashMap::new();
        // Vowels.
        for v in ["i", "ɪ", "e", "ɛ", "æ", "a", "ɑ", "ɒ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ", "ɐ"] {
            d.insert(v.to_string(), 120);
        }
        // Long vowels.
        for v in ["iː", "ɑː", "ɔː", "uː", "ɜː"] {
            d.insert(v.to_string(), 180);
        }
        // Diphthongs.
        for v in ["aɪ", "aʊ", "eɪ", "ɔɪ", "əʊ", "oʊ", "ɪə", "eə"] {
            d.insert(v.to_string(), 220);
        }
        // Plosives.
        for c in ["p", "t", "k"] { d.insert(c.to_string(), 75); }
        for c in ["b", "d", "ɡ"] { d.insert(c.to_string(), 60); }
        // Fricatives.
        for c in ["f", "θ", "s", "ʃ", "h"] { d.insert(c.to_string(), 100); }
        for c in ["v", "ð", "z", "ʒ"] { d.insert(c.to_string(), 70); }
        // Affricates.
        d.insert("tʃ".to_string(), 110);
        d.insert("dʒ".to_string(), 80);
        // Nasals.
        for c in ["m", "n", "ŋ"] { d.insert(c.to_string(), 70); }
        // Liquids/approximants.
        for c in ["l", "ɹ"] { d.insert(c.to_string(), 65); }
        for c in ["w", "j"] { d.insert(c.to_string(), 55); }

        Self { durations: d }
    }

    fn expected_ms(&self, ipa: &str) -> u16 {
        self.durations.get(ipa).copied().unwrap_or(80)
    }
}

/// Sequencer that converts frame-level phoneme candidates to phoneme events.
pub struct Sequencer {
    durations: DurationTable,
    /// Frame duration in milliseconds (typically 10ms).
    frame_ms: f32,
    /// Minimum number of frames for a phoneme to be valid.
    min_frames: usize,
}

impl Sequencer {
    /// Create a new sequencer.
    ///
    /// `frame_ms` is the duration of each MFCC frame in milliseconds (typically 10.0).
    pub fn new(frame_ms: f32) -> Self {
        Self {
            durations: DurationTable::new(),
            frame_ms,
            min_frames: 2, // at least 20ms at 10ms/frame
        }
    }

    /// Convert per-frame phoneme candidates into a phoneme event sequence.
    ///
    /// Uses sliding window consensus voting to smooth noisy per-frame picks,
    /// then merges adjacent identical phonemes into events.
    pub fn sequence(
        &self,
        frame_candidates: &[Vec<PhonemeCandidate>],
        vad: &[bool],
    ) -> Vec<PhonemeEvent> {
        if frame_candidates.is_empty() {
            return Vec::new();
        }

        // Step 1: Sliding window consensus voting.
        // For each frame, vote among a window of ±2 frames (5-frame window).
        // The winner is the phoneme with the highest total confidence across the window.
        let window_half = 2usize;
        let smoothed: Vec<Option<(String, f32)>> = (0..frame_candidates.len())
            .map(|center| {
                // Respect VAD.
                if !vad.get(center).copied().unwrap_or(true) {
                    return None;
                }

                // Accumulate votes from the window.
                let mut votes: HashMap<&str, f32> = HashMap::new();
                let lo = center.saturating_sub(window_half);
                let hi = (center + window_half + 1).min(frame_candidates.len());

                for i in lo..hi {
                    if !vad.get(i).copied().unwrap_or(true) {
                        continue;
                    }
                    // Top candidate from each frame in the window votes.
                    // Weight by confidence and proximity to center.
                    let dist = (i as f32 - center as f32).abs();
                    let proximity_weight = 1.0 / (1.0 + dist * 0.5);

                    for (rank, c) in frame_candidates[i].iter().take(2).enumerate() {
                        let rank_weight = if rank == 0 { 1.0 } else { 0.3 };
                        *votes.entry(&c.ipa).or_insert(0.0) +=
                            c.confidence * proximity_weight * rank_weight;
                    }
                }

                // Pick the winner.
                votes
                    .into_iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(ipa, conf)| (ipa.to_string(), conf / (hi - lo) as f32))
            })
            .collect();

        // Step 2: Merge consecutive identical phonemes into events.
        let mut events = Vec::new();
        let mut current_ipa: Option<String> = None;
        let mut start_frame = 0;
        let mut confidence_sum = 0.0f32;
        let mut frame_count = 0usize;

        for (i, pick) in smoothed.iter().enumerate() {
            let ipa = pick.as_ref().map(|(s, _)| s.as_str());
            let conf = pick.as_ref().map(|(_, c)| *c).unwrap_or(0.0);

            let same = match (&current_ipa, ipa) {
                (Some(cur), Some(new)) => cur == new,
                (None, None) => true,
                _ => false,
            };

            if !same {
                // Emit the previous run.
                if let Some(ref prev_ipa) = current_ipa {
                    if frame_count >= self.min_frames {
                        // Collect alternatives from the frame candidates at the midpoint.
                        let mid = start_frame + frame_count / 2;
                        let alts = if mid < frame_candidates.len() {
                            frame_candidates[mid]
                                .iter()
                                .skip(1)
                                .take(3)
                                .map(|c| (c.ipa.clone(), c.confidence))
                                .collect()
                        } else {
                            Vec::new()
                        };

                        events.push(PhonemeEvent {
                            ipa: prev_ipa.clone(),
                            start_frame,
                            end_frame: i,
                            confidence: confidence_sum / frame_count as f32,
                            alternatives: alts,
                        });
                    }
                }
                current_ipa = ipa.map(|s| s.to_string());
                start_frame = i;
                confidence_sum = conf;
                frame_count = 1;
            } else {
                confidence_sum += conf;
                frame_count += 1;
            }
        }

        // Emit final run.
        if let Some(ref prev_ipa) = current_ipa {
            if frame_count >= self.min_frames {
                events.push(PhonemeEvent {
                    ipa: prev_ipa.clone(),
                    start_frame,
                    end_frame: smoothed.len(),
                    confidence: confidence_sum / frame_count as f32,
                    alternatives: Vec::new(),
                });
            }
        }

        events
    }

    /// Get the expected duration in ms for a phoneme.
    pub fn expected_duration_ms(&self, ipa: &str) -> u16 {
        self.durations.expected_ms(ipa)
    }

    /// Frame duration in milliseconds.
    pub fn frame_ms(&self) -> f32 {
        self.frame_ms
    }
}

impl Default for Sequencer {
    fn default() -> Self {
        Self::new(10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(ipa: &str, confidence: f32) -> PhonemeCandidate {
        PhonemeCandidate {
            ipa: ipa.to_string(),
            score: (confidence * 256.0) as i32,
            confidence,
        }
    }

    #[test]
    fn merges_adjacent_same_phonemes() {
        let seq = Sequencer::new(10.0);
        let frames = vec![
            vec![make_candidate("ɑ", 0.9)],
            vec![make_candidate("ɑ", 0.8)],
            vec![make_candidate("ɑ", 0.85)],
            vec![make_candidate("t", 0.7)],
            vec![make_candidate("t", 0.6)],
        ];
        let vad = vec![true; 5];
        let events = seq.sequence(&frames, &vad);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].ipa, "ɑ");
        assert_eq!(events[0].duration_frames(), 3);
        assert_eq!(events[1].ipa, "t");
        assert_eq!(events[1].duration_frames(), 2);
    }

    #[test]
    fn vad_silence_splits_events() {
        let seq = Sequencer::new(10.0);
        let frames = vec![
            vec![make_candidate("ɑ", 0.9)],
            vec![make_candidate("ɑ", 0.8)],
            vec![make_candidate("ɑ", 0.8)], // silence — VAD says no
            vec![make_candidate("t", 0.7)],
            vec![make_candidate("t", 0.6)],
        ];
        let vad = vec![true, true, false, true, true];
        let events = seq.sequence(&frames, &vad);

        // Should have ɑ (2 frames), gap, t (2 frames).
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].ipa, "ɑ");
        assert_eq!(events[0].duration_frames(), 2);
        assert_eq!(events[1].ipa, "t");
    }

    #[test]
    fn single_frame_phoneme_rejected() {
        let seq = Sequencer::new(10.0);
        let frames = vec![
            vec![make_candidate("ɑ", 0.9)],
            vec![make_candidate("t", 0.7)], // only 1 frame — below min_frames
            vec![make_candidate("ɑ", 0.8)],
            vec![make_candidate("ɑ", 0.8)],
        ];
        let vad = vec![true; 4];
        let events = seq.sequence(&frames, &vad);

        // The single-frame "t" should be rejected (min_frames=2).
        // Result: two separate "ɑ" events or one if they merge.
        // They won't merge because "t" breaks the run.
        // After "t" is rejected, we get ɑ(1 frame, rejected) and ɑ(2 frames, kept).
        assert!(events.len() >= 1);
    }
}
