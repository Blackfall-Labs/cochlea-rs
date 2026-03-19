//! Transcript output types.

/// A complete transcription result.
#[derive(Debug, Clone)]
pub struct Transcript {
    /// The full transcribed text.
    pub text: String,
    /// Per-word details with timing and confidence.
    pub words: Vec<TranscribedWord>,
    /// Overall transcription confidence (0.0-1.0).
    pub confidence: f32,
}

/// A single transcribed word with metadata.
#[derive(Debug, Clone)]
pub struct TranscribedWord {
    /// The recognized word.
    pub word: String,
    /// Start time in milliseconds.
    pub start_ms: u32,
    /// End time in milliseconds.
    pub end_ms: u32,
    /// Recognition confidence (0.0-1.0).
    pub confidence: f32,
    /// Alternative word candidates.
    pub alternatives: Vec<String>,
}

impl Transcript {
    /// Create an empty transcript.
    pub fn empty() -> Self {
        Self {
            text: String::new(),
            words: Vec::new(),
            confidence: 0.0,
        }
    }

    /// Build a transcript from a sequence of transcribed words.
    pub fn from_words(words: Vec<TranscribedWord>) -> Self {
        if words.is_empty() {
            return Self::empty();
        }

        let text = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let confidence = words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32;

        Self {
            text,
            words,
            confidence,
        }
    }
}

impl std::fmt::Display for Transcript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}
