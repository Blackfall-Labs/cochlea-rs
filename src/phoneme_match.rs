//! Direct phoneme matching via formant distance.
//!
//! Instead of encoding phonemes as PackedSignal vectors and querying a DataBank
//! (which has a zero-dimension false-positive problem), this module does direct
//! formant distance comparison. Each phoneme's expected F1/F2/F3 are compared
//! against the estimated formant values using weighted Euclidean distance.
//!
//! This is simpler and more accurate than the DataBank approach for phoneme
//! recognition, though it doesn't scale to millions of entries. For 45 phonemes,
//! brute-force is fast enough.

use crate::formant::FormantEstimate;
use crate::phoneme_bank::PhonemeCandidate;
use signal_oscillator::ipa;
use signal_oscillator::phoneme_table;

/// Reference phoneme with expected acoustic properties.
struct PhonemeRef {
    ipa: String,
    f1_hz: f32,
    f2_hz: f32,
    f3_hz: f32,
    has_formants: bool,
    voiced: bool,
    voicing_energy: f32,
}

/// Direct phoneme matcher using formant distance.
pub struct PhonemeMatcher {
    references: Vec<PhonemeRef>,
}

impl PhonemeMatcher {
    /// Build a matcher from the oscillator's phoneme inventory.
    pub fn build() -> Self {
        let phonemes = [
            // Vowels (monophthongs)
            "i", "ɪ", "e", "ɛ", "æ", "a", "ɑ", "ɒ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ", "ɐ",
            // Long vowels
            "iː", "ɑː", "ɔː", "uː", "ɜː",
            // Diphthongs
            "aɪ", "aʊ", "eɪ", "ɔɪ", "əʊ", "oʊ", "ɪə", "eə",
            // Plosives
            "p", "b", "t", "d", "k", "ɡ",
            // Fricatives
            "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
            // Affricates
            "tʃ", "dʒ",
            // Nasals
            "m", "n", "ŋ",
            // Liquids + approximants
            "l", "ɹ", "w", "j",
        ];

        let mut references = Vec::new();

        for &ipa_str in &phonemes {
            let tokens = ipa::tokenize(ipa_str);
            if let Some(token) = tokens.into_iter().next() {
                if let Some(params) = phoneme_table::lookup(&token) {
                    let (f1, f2, f3, has_formants) = if let Some(ref f) = params.formants {
                        (f.f1_hz, f.f2_hz, f.f3_hz, true)
                    } else {
                        (0.0, 0.0, 0.0, false)
                    };

                    references.push(PhonemeRef {
                        ipa: ipa_str.to_string(),
                        f1_hz: f1,
                        f2_hz: f2,
                        f3_hz: f3,
                        has_formants,
                        voiced: params.voicing_energy > 0.3,
                        voicing_energy: params.voicing_energy,
                    });
                }
            }
        }

        log::info!(
            "PhonemeMatcher built with {} phonemes ({} with formants)",
            references.len(),
            references.iter().filter(|r| r.has_formants).count()
        );

        Self { references }
    }

    /// Find the top-K phoneme candidates for a formant estimate.
    pub fn match_phoneme(&self, f: &FormantEstimate, k: usize) -> Vec<PhonemeCandidate> {
        let has_strong_formants = f.f1_confidence > 0.05 && f.f2_confidence > 0.05;

        let mut scored: Vec<(usize, f32)> = self
            .references
            .iter()
            .enumerate()
            .filter_map(|(idx, r)| {
                // If we detect strong formants, only consider phonemes with formants.
                if has_strong_formants && !r.has_formants {
                    return None;
                }
                // If no strong formants, only consider consonants without formants.
                if !has_strong_formants && r.has_formants {
                    return None;
                }

                let score = if r.has_formants {
                    // Formant distance (weighted).
                    let f1_dist = ((f.f1_hz - r.f1_hz) / 350.0).powi(2); // F1 range ~700Hz
                    let f2_dist = ((f.f2_hz - r.f2_hz) / 950.0).powi(2); // F2 range ~1900Hz
                    let f3_dist = ((f.f3_hz - r.f3_hz) / 1000.0).powi(2); // F3 range ~2000Hz

                    // Weight by confidence.
                    let dist = f1_dist * f.f1_confidence
                        + f2_dist * f.f2_confidence
                        + f3_dist * f.f3_confidence * 0.3; // F3 less important

                    // Voicing mismatch penalty.
                    let voicing_penalty = if f.voiced != r.voiced { 0.5 } else { 0.0 };

                    dist + voicing_penalty
                } else {
                    // For consonants without formants, match on voicing only.
                    // This is crude — consonant identification from MFCCs alone is hard.
                    let voicing_match = if f.voiced == r.voiced { 0.0 } else { 1.0 };
                    voicing_match + 0.5 // base penalty for consonants
                };

                Some((idx, score))
            })
            .collect();

        // Sort by distance (ascending).
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(k)
            .map(|(idx, dist)| {
                let r = &self.references[idx];
                // Convert distance to confidence (inverse).
                let confidence = (1.0 / (1.0 + dist * 3.0))
                    * f.f1_confidence.max(0.1)
                    * f.f2_confidence.max(0.1);

                PhonemeCandidate {
                    ipa: r.ipa.clone(),
                    score: ((1.0 - dist.min(1.0)) * 256.0) as i32,
                    confidence,
                }
            })
            .collect()
    }

    /// Number of phonemes in the matcher.
    pub fn len(&self) -> usize {
        self.references.len()
    }

    /// Whether the matcher is empty.
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_builds() {
        let matcher = PhonemeMatcher::build();
        assert!(
            matcher.len() >= 40,
            "expected 40+ phonemes, got {}",
            matcher.len()
        );
    }

    #[test]
    fn open_a_vowel_matches() {
        let matcher = PhonemeMatcher::build();

        // /ɑ/ has F1=730, F2=1090, F3=2440.
        let query = FormantEstimate {
            f1_hz: 730.0,
            f2_hz: 1090.0,
            f3_hz: 2440.0,
            f1_confidence: 0.9,
            f2_confidence: 0.9,
            f3_confidence: 0.7,
            voiced: true,
            energy: 0.0,
        };

        let candidates = matcher.match_phoneme(&query, 5);
        assert!(!candidates.is_empty());

        let top = &candidates[0];
        // Should be /ɑ/ or a close back vowel.
        let open_back = ["ɑ", "ɑː", "a", "ɒ", "ɔ"];
        assert!(
            open_back.contains(&top.ipa.as_str()),
            "expected open back vowel, got /{}/",
            top.ipa
        );
    }

    #[test]
    fn front_close_vowel_matches() {
        let matcher = PhonemeMatcher::build();

        // /i/ has F1=270, F2=2290, F3=3010.
        let query = FormantEstimate {
            f1_hz: 270.0,
            f2_hz: 2290.0,
            f3_hz: 3010.0,
            f1_confidence: 0.9,
            f2_confidence: 0.9,
            f3_confidence: 0.7,
            voiced: true,
            energy: 0.0,
        };

        let candidates = matcher.match_phoneme(&query, 5);
        let top = &candidates[0];
        let front_close = ["i", "iː", "ɪ"];
        assert!(
            front_close.contains(&top.ipa.as_str()),
            "expected front close vowel, got /{}/",
            top.ipa
        );
    }
}
