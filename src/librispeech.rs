//! LibriSpeech spool reader — extracts MFCC frames and transcripts from
//! hush's pre-processed `dev-clean.spool` file.
//!
//! The spool file contains 2,703 LibriSpeech samples, each stored as a
//! dataspool card with quantized 26-dim MFCC frames and a UTF-8 transcript.
//!
//! Card binary format:
//! ```text
//! [u32 n_frames]                  — number of MFCC frames
//! [n_frames × 26 i8]             — quantized MFCC coefficients
//! [u16 transcript_len]           — transcript byte length
//! [transcript_len × u8]          — UTF-8 transcript text
//! ```
//!
//! Dequantization: `f32_value = (i8_value as f32 / 127.0) * 50.0`

use std::path::Path;

/// A single LibriSpeech sample: pre-extracted MFCC frames + transcript.
pub struct LibriSpeechSample {
    /// Dequantized MFCC frames (each Vec<f32> has 26 coefficients).
    pub frames: Vec<Vec<f32>>,
    /// The transcript text (uppercased words, space-separated).
    pub transcript: String,
}

/// Number of MFCC coefficients per frame in the spool.
const SPOOL_MFCC_DIM: usize = 26;

/// Read all LibriSpeech samples from a spool file.
///
/// Returns an error string if the file can't be read or parsed.
pub fn read_spool(path: &Path) -> Result<Vec<LibriSpeechSample>, String> {
    let mut reader = dataspool_rs::SpoolReader::open(path)
        .map_err(|e| format!("failed to open spool: {}", e))?;

    let count = reader.card_count();
    let mut samples = Vec::with_capacity(count);

    for i in 0..count {
        let card_bytes = reader
            .read_card(i)
            .map_err(|e| format!("failed to read card {}: {}", i, e))?;

        match parse_card(&card_bytes) {
            Some(sample) => samples.push(sample),
            None => {
                log::warn!("Skipping malformed card {}", i);
            }
        }
    }

    log::info!(
        "Read {} LibriSpeech samples from spool ({} cards total)",
        samples.len(),
        count
    );

    Ok(samples)
}

/// Parse a single card's raw bytes into a LibriSpeechSample.
fn parse_card(data: &[u8]) -> Option<LibriSpeechSample> {
    if data.len() < 4 {
        return None;
    }

    // Read n_frames (u32 LE).
    let n_frames = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

    let mfcc_bytes = n_frames * SPOOL_MFCC_DIM;
    let mfcc_start = 4;
    let mfcc_end = mfcc_start + mfcc_bytes;

    if data.len() < mfcc_end + 2 {
        return None;
    }

    // Read MFCC frames and dequantize i8 → f32.
    let mut frames = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let frame_start = mfcc_start + f * SPOOL_MFCC_DIM;
        let mut coeffs = Vec::with_capacity(SPOOL_MFCC_DIM);
        for j in 0..SPOOL_MFCC_DIM {
            let quantized = data[frame_start + j] as i8;
            let dequantized = (quantized as f32 / 127.0) * 50.0;
            coeffs.push(dequantized);
        }
        frames.push(coeffs);
    }

    // Read transcript length (u16 LE).
    let transcript_len =
        u16::from_le_bytes([data[mfcc_end], data[mfcc_end + 1]]) as usize;

    let transcript_start = mfcc_end + 2;
    let transcript_end = transcript_start + transcript_len;

    if data.len() < transcript_end {
        return None;
    }

    let transcript = String::from_utf8_lossy(&data[transcript_start..transcript_end]).to_string();

    Some(LibriSpeechSample { frames, transcript })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn spool_path() -> PathBuf {
        // Try relative to project root first, then absolute.
        let paths = [
            PathBuf::from("../hush/data/dev-clean.spool"),
            PathBuf::from("E:/repos/blackfall-labs/hush/data/dev-clean.spool"),
        ];
        for p in &paths {
            if p.exists() {
                return p.clone();
            }
        }
        paths[0].clone() // will fail with descriptive error
    }

    #[test]
    fn read_spool_if_available() {
        let path = spool_path();
        if !path.exists() {
            eprintln!("Skipping: spool not found at {:?}", path);
            return;
        }

        let samples = read_spool(&path).expect("should read spool");
        assert!(
            samples.len() > 100,
            "expected many samples, got {}",
            samples.len()
        );

        // Check first sample has reasonable data.
        let s = &samples[0];
        assert!(!s.frames.is_empty(), "should have frames");
        assert_eq!(s.frames[0].len(), SPOOL_MFCC_DIM);
        assert!(!s.transcript.is_empty(), "should have transcript");

        eprintln!(
            "First sample: {} frames, transcript: {:?}",
            s.frames.len(),
            &s.transcript[..s.transcript.len().min(80)]
        );
    }

    #[test]
    fn dequantization_range() {
        let path = spool_path();
        if !path.exists() {
            return;
        }

        let samples = read_spool(&path).expect("should read spool");
        let sample = &samples[0];

        // Dequantized values should be in [-50, 50] range.
        for frame in &sample.frames {
            for &val in frame {
                assert!(
                    val >= -50.0 && val <= 50.0,
                    "dequantized value out of range: {}",
                    val
                );
            }
        }
    }
}
