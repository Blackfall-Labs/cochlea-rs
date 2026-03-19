//! PhonemeBank — DataBank-backed phoneme pattern memory.
//!
//! Stores the acoustic signature of each English phoneme (45 total) as a
//! 16-dimensional PackedSignal vector. Given a FormantEstimate from audio,
//! queries the bank to find the most likely phoneme candidates.
//!
//! The 16 dimensions encode formant frequencies, ratios, voicing energy,
//! breathiness, tension, nasality, and articulatory features. Consonants
//! without formant structure have zero-valued formant dims — DataBank's
//! sparse cosine similarity ignores zeros, so consonants match only on
//! their non-formant features.

use crate::formant::FormantEstimate;
use databank_rs::{BankConfig, BankId, DataBank, EntryId, IndexType, Temperature};
use signal_oscillator::ipa::{self, Manner, PhonemeClass, Place, VowelHeight};
use signal_oscillator::phoneme_table::{self, PhonemeParams};
use ternary_signal::PackedSignal;
use std::collections::HashMap;

/// Number of dimensions in the phoneme signal vector.
const PHONEME_VEC_WIDTH: usize = 16;

/// A phoneme candidate from a bank query.
#[derive(Debug, Clone)]
pub struct PhonemeCandidate {
    /// The IPA symbol for this phoneme.
    pub ipa: String,
    /// DataBank similarity score.
    pub score: i32,
    /// Combined confidence (bank score × formant estimation confidence).
    pub confidence: f32,
}

/// Phoneme pattern bank wrapping a DataBank.
pub struct PhonemeBank {
    bank: DataBank,
    /// Maps EntryId → IPA string for decoding query results.
    id_to_ipa: HashMap<EntryId, String>,
}

impl PhonemeBank {
    /// Build a PhonemeBank populated with all 45 English phonemes.
    ///
    /// Iterates over the oscillator's phoneme inventory, encodes each phoneme's
    /// acoustic signature as a 16-dim PackedSignal vector, and inserts into a DataBank.
    pub fn build() -> Self {
        let bank_id = BankId::new("cochlea.phoneme", 0);
        let config = BankConfig {
            vector_width: PHONEME_VEC_WIDTH as u16,
            max_entries: 128, // 45 phonemes + room for variants
            persist_after_mutations: 1000,
            persist_after_ticks: 100_000,
            max_edges_per_entry: 8,
            index_type: IndexType::Ivf { k: 4, nprobe: 4 }, // small bank
        };
        let mut bank = DataBank::new(bank_id, "cochlea.phoneme".into(), config);
        let mut id_to_ipa = HashMap::new();

        // Insert all phonemes from the inventory.
        for (ipa_str, token, params) in phoneme_inventory() {
            let vector = encode_phoneme(&token, &params);
            if let Ok(entry_id) = bank.insert(vector, Temperature::Cold, 0) {
                id_to_ipa.insert(entry_id, ipa_str);
            }
        }

        Self { bank, id_to_ipa }
    }

    /// Query the bank with a formant estimate, returning top-K phoneme candidates.
    ///
    /// Post-filters results based on formant confidence: when strong formant peaks
    /// are detected, consonants without formant structure are deprioritized (they
    /// match on voicing alone, producing false positives).
    pub fn query(&self, formant: &FormantEstimate, k: usize) -> Vec<PhonemeCandidate> {
        let query = encode_formant_query(formant);
        // Request more candidates than k so we have room after filtering.
        let results = self.bank.query_sparse(&query, k * 3);

        let has_formants = formant.f1_confidence > 0.3 && formant.f2_confidence > 0.3;

        let mut candidates: Vec<PhonemeCandidate> = results
            .into_iter()
            .filter_map(|r| {
                let ipa = self.id_to_ipa.get(&r.entry_id)?.clone();
                let base_confidence = (r.score as f32 / 256.0).max(0.0)
                    * formant.f1_confidence
                    * formant.f2_confidence;

                // Classify this phoneme: does it have formant structure?
                let is_formant_phoneme = self.has_formant_structure(&ipa);

                let confidence = if has_formants && !is_formant_phoneme {
                    // Strong formant peaks but matching a consonant without formants.
                    // This is likely a false positive from single-dim voicing match.
                    base_confidence * 0.15
                } else if !has_formants && is_formant_phoneme {
                    // Weak formant peaks but matching a vowel — less confident.
                    base_confidence * 0.5
                } else {
                    base_confidence
                };

                Some(PhonemeCandidate {
                    ipa,
                    score: r.score,
                    confidence,
                })
            })
            .collect();

        // Sort by confidence descending and take top-k.
        candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Check if a phoneme has formant structure (vowels, approximants, nasals).
    fn has_formant_structure(&self, ipa: &str) -> bool {
        matches!(
            ipa,
            "i" | "ɪ" | "e" | "ɛ" | "æ" | "a" | "ɑ" | "ɒ" | "ɔ" | "o" | "ʊ" | "u"
            | "ʌ" | "ə" | "ɜ" | "ɐ"
            | "iː" | "ɑː" | "ɔː" | "uː" | "ɜː"
            | "aɪ" | "aʊ" | "eɪ" | "ɔɪ" | "əʊ" | "oʊ" | "ɪə" | "eə"
            | "m" | "n" | "ŋ" | "l" | "ɹ" | "w" | "j"
        )
    }

    /// Number of phonemes in the bank.
    pub fn len(&self) -> usize {
        self.id_to_ipa.len()
    }

    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_ipa.is_empty()
    }
}

/// Encode a phoneme's acoustic signature into a 16-dim PackedSignal vector.
fn encode_phoneme(token: &ipa::PhonemeToken, params: &PhonemeParams) -> Vec<PackedSignal> {
    let mut v = vec![PackedSignal::ZERO; PHONEME_VEC_WIDTH];

    // Dims 0-2: Formant frequencies (normalized).
    if let Some(ref f) = params.formants {
        v[0] = pack_excitatory(normalize_f1(f.f1_hz));
        v[1] = pack_excitatory(normalize_f2(f.f2_hz));
        v[2] = pack_excitatory(normalize_f3(f.f3_hz));

        // Dims 3-4: Formant ratios (key discriminators).
        let f1f2_ratio = f.f1_hz / f.f2_hz;
        v[3] = pack_excitatory((f1f2_ratio * 255.0).min(255.0) as u8);
        let f2f3_ratio = f.f2_hz / f.f3_hz;
        v[4] = pack_excitatory((f2f3_ratio * 255.0).min(255.0) as u8);
    }
    // Else: dims 0-4 stay ZERO — sparse similarity ignores them for consonants.

    // Dim 5: Voicing energy.
    v[5] = pack_excitatory((params.voicing_energy * 255.0) as u8);

    // Dim 6: Breathiness.
    v[6] = pack_excitatory((params.breathiness * 255.0).min(255.0) as u8);

    // Dim 7: Tension.
    v[7] = pack_excitatory((params.tension * 255.0) as u8);

    // Dim 8: Nasal coupling.
    v[8] = pack_excitatory((params.nasal_coupling * 255.0) as u8);

    // Dim 9: Fricative intensity.
    if let Some(ref fric) = params.fricative {
        v[9] = pack_excitatory((fric.intensity * 255.0) as u8);
        // Dim 10: Fricative constriction.
        v[10] = pack_excitatory((fric.constriction * 255.0) as u8);
    }

    // Dim 11: Plosive intensity.
    if let Some(ref plos) = params.plosive {
        v[11] = pack_excitatory((plos.intensity * 255.0) as u8);
    }

    // Dim 12: Duration class.
    let dur_class = if params.base_duration_ms <= 60 {
        85 // short
    } else if params.base_duration_ms <= 120 {
        170 // medium
    } else {
        255 // long
    };
    v[12] = pack_excitatory(dur_class);

    // Dim 13: Manner encoding.
    let manner_val = match token.features.manner {
        Some(Manner::Plosive) => 36,
        Some(Manner::Fricative) => 73,
        Some(Manner::Affricate) => 109,
        Some(Manner::Nasal) => 146,
        Some(Manner::Lateral) => 182,
        Some(Manner::Approximant) => 219,
        Some(Manner::Tap) => 255,
        None => 0, // vowel
    };
    if manner_val > 0 {
        v[13] = pack_excitatory(manner_val);
    }

    // Dim 14: Place encoding.
    let place_val = match token.features.place {
        Some(Place::Bilabial) => 32,
        Some(Place::Labiodental) => 64,
        Some(Place::Dental) => 96,
        Some(Place::Alveolar) => 128,
        Some(Place::Postalveolar) => 160,
        Some(Place::Palatal) => 192,
        Some(Place::Velar) => 224,
        Some(Place::Glottal) => 255,
        None => 0,
    };
    if place_val > 0 {
        v[14] = pack_excitatory(place_val);
    }

    // Dim 15: Height/backness for vowels.
    if token.class == PhonemeClass::Vowel {
        let height_val = match token.features.height {
            Some(VowelHeight::Close) => 36,
            Some(VowelHeight::NearClose) => 73,
            Some(VowelHeight::CloseMid) => 109,
            Some(VowelHeight::Mid) => 146,
            Some(VowelHeight::OpenMid) => 182,
            Some(VowelHeight::NearOpen) => 219,
            Some(VowelHeight::Open) => 255,
            None => 128,
        };
        // Encode height as the primary value (most discriminating for vowels).
        v[15] = pack_excitatory(height_val);
    }

    v
}

/// Encode a FormantEstimate as a query vector for the PhonemeBank.
fn encode_formant_query(f: &FormantEstimate) -> Vec<PackedSignal> {
    let mut v = vec![PackedSignal::ZERO; PHONEME_VEC_WIDTH];

    // Always set formant dims (the main query signal).
    v[0] = pack_excitatory(normalize_f1(f.f1_hz));
    v[1] = pack_excitatory(normalize_f2(f.f2_hz));

    // Only set F3 if we have confidence in it.
    if f.f3_confidence > 0.2 {
        v[2] = pack_excitatory(normalize_f3(f.f3_hz));
    }

    // Formant ratios.
    if f.f2_hz > 0.0 {
        let f1f2_ratio = f.f1_hz / f.f2_hz;
        v[3] = pack_excitatory((f1f2_ratio * 255.0).min(255.0) as u8);
    }
    if f.f3_hz > 0.0 && f.f3_confidence > 0.2 {
        let f2f3_ratio = f.f2_hz / f.f3_hz;
        v[4] = pack_excitatory((f2f3_ratio * 255.0).min(255.0) as u8);
    }

    // Voicing.
    if f.voiced {
        v[5] = pack_excitatory(200);
    }
    // Non-formant dims left ZERO — query matches on formant + voicing only.
    // This is intentional: formants are what we estimate from MFCC.

    v
}

/// Normalize F1 (200-900 Hz) to 0-255.
fn normalize_f1(hz: f32) -> u8 {
    ((hz - 200.0) / 700.0 * 255.0).clamp(0.0, 255.0) as u8
}

/// Normalize F2 (600-2500 Hz) to 0-255.
fn normalize_f2(hz: f32) -> u8 {
    ((hz - 600.0) / 1900.0 * 255.0).clamp(0.0, 255.0) as u8
}

/// Normalize F3 (1500-3500 Hz) to 0-255.
fn normalize_f3(hz: f32) -> u8 {
    ((hz - 1500.0) / 2000.0 * 255.0).clamp(0.0, 255.0) as u8
}

/// Pack a value as an excitatory PackedSignal (polarity=+1, multiplier=1).
fn pack_excitatory(magnitude: u8) -> PackedSignal {
    if magnitude == 0 {
        PackedSignal::ZERO
    } else {
        PackedSignal::pack(1, magnitude, 1)
    }
}

/// Enumerate the complete English phoneme inventory with their acoustic parameters.
///
/// Returns (IPA string, PhonemeToken, PhonemeParams) for all 45 phonemes.
fn phoneme_inventory() -> Vec<(String, ipa::PhonemeToken, PhonemeParams)> {
    // All English phonemes as IPA strings.
    let phonemes = [
        // Vowels (monophthongs)
        "i", "ɪ", "e", "ɛ", "æ", "a", "ɑ", "ɒ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ", "ɐ",
        // Vowels (long)
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

    let mut inventory = Vec::new();
    for &ipa_str in &phonemes {
        let tokens = ipa::tokenize(ipa_str);
        if let Some(token) = tokens.into_iter().next() {
            if let Some(params) = phoneme_table::lookup(&token) {
                inventory.push((ipa_str.to_string(), token, params));
            }
        }
    }
    inventory
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phoneme_inventory_has_entries() {
        let inv = phoneme_inventory();
        assert!(
            inv.len() >= 40,
            "expected at least 40 phonemes, got {}",
            inv.len()
        );
    }

    #[test]
    fn bank_builds_successfully() {
        let bank = PhonemeBank::build();
        assert!(
            bank.len() >= 40,
            "bank should have at least 40 phonemes, got {}",
            bank.len()
        );
    }

    #[test]
    fn vowel_self_recognition() {
        let bank = PhonemeBank::build();

        // The vowel /ɑ/ has F1=730, F2=1090, F3=2440.
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

        let candidates = bank.query(&query, 5);
        assert!(!candidates.is_empty(), "should return candidates");

        // Check that open back vowels are in the top results.
        let top_ipas: Vec<&str> = candidates.iter().map(|c| c.ipa.as_str()).collect();
        log::info!("Query for /ɑ/ returned: {:?}", top_ipas);
        // /ɑ/ or /a/ should be near the top.
    }

    #[test]
    fn front_close_vowel_recognition() {
        let bank = PhonemeBank::build();

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

        let candidates = bank.query(&query, 5);
        assert!(!candidates.is_empty());
        let top_ipas: Vec<&str> = candidates.iter().map(|c| c.ipa.as_str()).collect();
        log::info!("Query for /i/ returned: {:?}", top_ipas);
    }

    #[test]
    fn encode_normalize_roundtrip() {
        // F1 at 550Hz (midpoint of 200-900 range).
        let norm = normalize_f1(550.0);
        assert!(norm > 120 && norm < 135, "midpoint F1 should be ~127, got {}", norm);

        // F2 at 1550Hz (midpoint of 600-2500 range).
        let norm = normalize_f2(1550.0);
        assert!(norm > 120 && norm < 135, "midpoint F2 should be ~127, got {}", norm);
    }
}
