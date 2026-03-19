//! Disambiguator — selects the best word candidate when multiple match.
//!
//! Scoring combines edit distance, phoneme confidence, and word frequency
//! (approximated by lexicon sense count).

use crate::pronunciation::WordCandidate;
use crate::sequencer::PhonemeEvent;

/// Select the best word from a list of candidates.
///
/// Scoring: lower is better.
/// - Edit distance: each edit costs 150 points (heavy penalty).
/// - Confidence bonus: -50 × average phoneme confidence (rewards strong matches).
/// - Frequency bonus: -log2(sense_count + 1) × 30 (strongly rewards common words).
/// - Length penalty: single-letter words penalized unless very common.
pub fn select_best(
    candidates: &[WordCandidate],
    phoneme_events: &[PhonemeEvent],
) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }

    let avg_confidence: f32 = if phoneme_events.is_empty() {
        0.5
    } else {
        phoneme_events.iter().map(|e| e.confidence).sum::<f32>()
            / phoneme_events.len() as f32
    };

    let mut scored: Vec<(&WordCandidate, f32)> = candidates
        .iter()
        .map(|c| {
            let edit_cost = c.edit_distance as f32 * 200.0;
            let confidence_bonus = avg_confidence * c.phoneme_coverage * 50.0;
            // Log-scale frequency bonus — with the new frequency data (1-500),
            // this gives 0 for freq=1, ~120 for freq=100, ~186 for freq=500.
            let frequency_bonus = ((c.sense_count as f32 + 1.0).log2()) * 20.0;

            // Penalize very short words (1-2 chars) that aren't common function words.
            let length_penalty = if c.word.len() <= 2 && c.sense_count < 50 {
                100.0
            } else {
                0.0
            };

            let score = edit_cost - confidence_bonus - frequency_bonus + length_penalty;
            (c, score)
        })
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    scored.first().map(|(c, _)| c.word.clone())
}

/// Select top-N word candidates with their scores.
pub fn rank_candidates(
    candidates: &[WordCandidate],
    phoneme_events: &[PhonemeEvent],
    n: usize,
) -> Vec<(String, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let avg_confidence: f32 = if phoneme_events.is_empty() {
        0.5
    } else {
        phoneme_events.iter().map(|e| e.confidence).sum::<f32>()
            / phoneme_events.len() as f32
    };

    let mut scored: Vec<(String, f32)> = candidates
        .iter()
        .map(|c| {
            let edit_cost = c.edit_distance as f32 * 100.0;
            let confidence_bonus = avg_confidence * c.phoneme_coverage * 50.0;
            let frequency_bonus = ((c.sense_count as f32 + 1.0).log2()) * 10.0;
            let score = edit_cost - confidence_bonus - frequency_bonus;
            (c.word.clone(), score)
        })
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_exact_match_over_fuzzy() {
        let candidates = vec![
            WordCandidate {
                word: "cat".into(),
                edit_distance: 0,
                phoneme_coverage: 1.0,
                sense_count: 5,
            },
            WordCandidate {
                word: "hat".into(),
                edit_distance: 1,
                phoneme_coverage: 1.0,
                sense_count: 5,
            },
        ];
        let events = vec![PhonemeEvent {
            ipa: "k".into(),
            start_frame: 0,
            end_frame: 3,
            confidence: 0.9,
            alternatives: vec![],
        }];

        let best = select_best(&candidates, &events).unwrap();
        assert_eq!(best, "cat");
    }

    #[test]
    fn prefers_common_words() {
        let candidates = vec![
            WordCandidate {
                word: "rare".into(),
                edit_distance: 0,
                phoneme_coverage: 1.0,
                sense_count: 1,
            },
            WordCandidate {
                word: "common".into(),
                edit_distance: 0,
                phoneme_coverage: 1.0,
                sense_count: 20,
            },
        ];

        let best = select_best(&candidates, &[]).unwrap();
        assert_eq!(best, "common");
    }

    #[test]
    fn empty_candidates_returns_none() {
        assert!(select_best(&[], &[]).is_none());
    }
}
