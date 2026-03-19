//! Pronunciation trie — maps phoneme sequences to words using lexicon IPA data.
//!
//! Builds a prefix trie from the blackfall-lexicon's 134K IPA pronunciations.
//! Supports exact and fuzzy (edit-distance) lookup of phoneme sequences
//! recognized from audio.

use signal_oscillator::ipa::{self, PhonemeClass};
use std::collections::HashMap;

/// A word candidate returned from trie lookup.
#[derive(Debug, Clone)]
pub struct WordCandidate {
    /// The English word.
    pub word: String,
    /// Edit distance from the query phoneme sequence.
    pub edit_distance: usize,
    /// Fraction of query phonemes consumed by this match (0.0-1.0).
    pub phoneme_coverage: f32,
    /// Number of senses in the lexicon (proxy for word frequency).
    pub sense_count: u32,
}

/// A trie node in the pronunciation tree.
struct TrieNode {
    /// Children keyed by IPA phoneme string.
    children: HashMap<String, TrieNode>,
    /// Words that end at this node.
    words: Vec<TrieWord>,
}

/// A word entry at a trie leaf.
#[derive(Debug, Clone)]
struct TrieWord {
    word: String,
    sense_count: u32,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            words: Vec::new(),
        }
    }
}

/// Pronunciation trie built from lexicon IPA data.
pub struct PronunciationTrie {
    root: TrieNode,
    word_count: usize,
}

impl PronunciationTrie {
    /// Create an empty trie.
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            word_count: 0,
        }
    }

    /// Insert a word with its IPA pronunciation.
    ///
    /// The IPA string is tokenized into phonemes, suprasegmentals are stripped,
    /// and the phoneme sequence is inserted into the trie.
    pub fn insert(&mut self, word: &str, ipa_str: &str, sense_count: u32) {
        let phonemes = ipa_to_phoneme_sequence(ipa_str);
        if phonemes.is_empty() {
            return;
        }

        let mut node = &mut self.root;
        for phoneme in &phonemes {
            node = node.children.entry(phoneme.clone()).or_insert_with(TrieNode::new);
        }
        node.words.push(TrieWord {
            word: word.to_string(),
            sense_count,
        });
        self.word_count += 1;
    }

    /// Exact lookup: find words matching the exact phoneme sequence.
    pub fn lookup_exact(&self, phonemes: &[String]) -> Vec<WordCandidate> {
        let mut node = &self.root;
        for phoneme in phonemes {
            match node.children.get(phoneme) {
                Some(child) => node = child,
                None => return Vec::new(),
            }
        }
        node.words
            .iter()
            .map(|w| WordCandidate {
                word: w.word.clone(),
                edit_distance: 0,
                phoneme_coverage: 1.0,
                sense_count: w.sense_count,
            })
            .collect()
    }

    /// Fuzzy lookup: find words within `max_edits` edit distance of the query.
    ///
    /// Uses a recursive trie walk with Levenshtein-style edit tracking.
    /// Supports substitution, insertion, and deletion of phonemes.
    pub fn lookup_fuzzy(
        &self,
        phonemes: &[String],
        max_edits: usize,
    ) -> Vec<WordCandidate> {
        let mut results = Vec::new();
        self.fuzzy_walk(
            &self.root,
            phonemes,
            0,
            0,
            max_edits,
            phonemes.len(),
            &mut results,
        );
        results.sort_by(|a, b| {
            a.edit_distance
                .cmp(&b.edit_distance)
                .then(b.sense_count.cmp(&a.sense_count))
        });
        results
    }

    /// Recursive fuzzy trie walk.
    fn fuzzy_walk(
        &self,
        node: &TrieNode,
        query: &[String],
        query_pos: usize,
        edits: usize,
        max_edits: usize,
        total_phonemes: usize,
        results: &mut Vec<WordCandidate>,
    ) {
        // If we've consumed the whole query, collect words at this node.
        // Remaining trie depth counts as deletions.
        if query_pos >= query.len() {
            for w in &node.words {
                results.push(WordCandidate {
                    word: w.word.clone(),
                    edit_distance: edits,
                    phoneme_coverage: if total_phonemes > 0 {
                        query_pos as f32 / total_phonemes as f32
                    } else {
                        1.0
                    },
                    sense_count: w.sense_count,
                });
            }
            // Try deletions (skip trie edges) if budget allows.
            if edits < max_edits {
                for (_, child) in &node.children {
                    self.fuzzy_walk(
                        child,
                        query,
                        query_pos,
                        edits + 1,
                        max_edits,
                        total_phonemes,
                        results,
                    );
                }
            }
            return;
        }

        let current = &query[query_pos];

        // Exact match — follow matching edge.
        if let Some(child) = node.children.get(current) {
            self.fuzzy_walk(
                child,
                query,
                query_pos + 1,
                edits,
                max_edits,
                total_phonemes,
                results,
            );
        }

        if edits < max_edits {
            // Substitution — follow non-matching edges.
            for (edge_phoneme, child) in &node.children {
                if edge_phoneme != current {
                    self.fuzzy_walk(
                        child,
                        query,
                        query_pos + 1,
                        edits + 1,
                        max_edits,
                        total_phonemes,
                        results,
                    );
                }
            }

            // Insertion — skip a query phoneme (it was spurious).
            self.fuzzy_walk(
                node,
                query,
                query_pos + 1,
                edits + 1,
                max_edits,
                total_phonemes,
                results,
            );

            // Deletion — skip a trie edge (phoneme was missed in recognition).
            // Already handled above when query_pos >= query.len().
        }

        // Collect words at this node if query is close to exhausted.
        let remaining = query.len() - query_pos;
        if remaining <= max_edits - edits {
            for w in &node.words {
                let total_edits = edits + remaining;
                if total_edits <= max_edits {
                    results.push(WordCandidate {
                        word: w.word.clone(),
                        edit_distance: total_edits,
                        phoneme_coverage: query_pos as f32 / total_phonemes.max(1) as f32,
                        sense_count: w.sense_count,
                    });
                }
            }
        }
    }

    /// Number of words in the trie.
    pub fn len(&self) -> usize {
        self.word_count
    }

    /// Whether the trie is empty.
    pub fn is_empty(&self) -> bool {
        self.word_count == 0
    }
}

impl Default for PronunciationTrie {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert an IPA string to a sequence of phoneme IPA symbols, stripping suprasegmentals.
pub fn ipa_to_phoneme_sequence(ipa_str: &str) -> Vec<String> {
    ipa::tokenize(ipa_str)
        .into_iter()
        .filter(|t| t.class != PhonemeClass::Suprasegmental)
        .map(|t| t.ipa)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_trie() -> PronunciationTrie {
        let mut trie = PronunciationTrie::new();
        trie.insert("hello", "hɛloʊ", 5);
        trie.insert("delete", "dɪliːt", 3);
        trie.insert("cat", "kæt", 8);
        trie.insert("hat", "hæt", 4);
        trie.insert("bat", "bæt", 2);
        trie
    }

    #[test]
    fn exact_lookup_finds_word() {
        let trie = build_test_trie();
        let phonemes = ipa_to_phoneme_sequence("kæt");
        let results = trie.lookup_exact(&phonemes);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].word, "cat");
        assert_eq!(results[0].edit_distance, 0);
    }

    #[test]
    fn exact_lookup_miss_returns_empty() {
        let trie = build_test_trie();
        let phonemes = ipa_to_phoneme_sequence("dɒɡ");
        let results = trie.lookup_exact(&phonemes);
        assert!(results.is_empty());
    }

    #[test]
    fn fuzzy_lookup_with_substitution() {
        let trie = build_test_trie();
        // Query "hæt" (hat) — should also find "cat" (kæt) and "bat" (bæt)
        // with 1 edit (substitution of first phoneme).
        let phonemes = ipa_to_phoneme_sequence("hæt");
        let results = trie.lookup_fuzzy(&phonemes, 1);
        let words: Vec<&str> = results.iter().map(|r| r.word.as_str()).collect();
        assert!(words.contains(&"hat"), "should find exact match 'hat'");
        assert!(words.contains(&"cat"), "should find 'cat' with 1 sub");
        assert!(words.contains(&"bat"), "should find 'bat' with 1 sub");
    }

    #[test]
    fn fuzzy_results_sorted_by_distance() {
        let trie = build_test_trie();
        let phonemes = ipa_to_phoneme_sequence("hæt");
        let results = trie.lookup_fuzzy(&phonemes, 1);
        // Exact match should come first.
        if results.len() >= 2 {
            assert!(results[0].edit_distance <= results[1].edit_distance);
        }
    }

    #[test]
    fn trie_count() {
        let trie = build_test_trie();
        assert_eq!(trie.len(), 5);
    }

    #[test]
    fn ipa_to_phonemes_strips_stress() {
        // "dɪˈliːt" has primary stress marker — should be stripped.
        let phonemes = ipa_to_phoneme_sequence("dɪˈliːt");
        assert!(
            !phonemes.iter().any(|p| p == "ˈ"),
            "stress markers should be stripped"
        );
    }
}
