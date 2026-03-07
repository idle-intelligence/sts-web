//! SentencePiece `.model` protobuf decoder.
//!
//! Parses a SentencePiece protobuf model file and decodes token IDs to text.

/// SentencePiece decoder: maps token IDs -> text.
pub struct SpmDecoder {
    /// vocab[i] = piece string for token ID `i`.
    vocab: Vec<String>,
}

impl SpmDecoder {
    /// Parse a SentencePiece `.model` protobuf from raw bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        let vocab = parse_sentencepiece_vocab(data);
        Self { vocab }
    }

    /// Decode a slice of token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        decode_tokens(&self.vocab, ids)
    }

    /// Number of vocabulary entries.
    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }
}

// ---------------------------------------------------------------------------
// Protobuf parsing
// ---------------------------------------------------------------------------

/// Extract piece strings from a SentencePiece protobuf.
/// Each piece's index in the returned Vec is its token ID.
fn parse_sentencepiece_vocab(data: &[u8]) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        let (tag, tag_len) = read_varint(data, pos);
        pos += tag_len;

        let field_num = tag >> 3;
        let wire_type = tag & 0x7;

        if field_num == 1 && wire_type == 2 {
            // Field 1 (repeated SentencePiece), length-delimited
            let (len, len_len) = read_varint(data, pos);
            pos += len_len;

            let piece_bytes = &data[pos..pos + len as usize];
            let piece = parse_sentence_piece(piece_bytes);
            pieces.push(piece);

            pos += len as usize;
        } else {
            pos = skip_field(data, pos, wire_type as u8);
        }
    }

    pieces
}

fn parse_sentence_piece(buf: &[u8]) -> String {
    let mut pos = 0;
    let mut piece = String::new();

    while pos < buf.len() {
        let (tag, tag_len) = read_varint(buf, pos);
        pos += tag_len;

        let field_num = tag >> 3;
        let wire_type = tag & 0x7;

        if field_num == 1 && wire_type == 2 {
            let (len, len_len) = read_varint(buf, pos);
            pos += len_len;
            piece = String::from_utf8_lossy(&buf[pos..pos + len as usize]).to_string();
            pos += len as usize;
        } else {
            pos = skip_field(buf, pos, wire_type as u8);
        }
    }

    piece
}

fn read_varint(buf: &[u8], start: usize) -> (u64, usize) {
    let mut value: u64 = 0;
    let mut shift = 0;
    let mut pos = start;

    while pos < buf.len() {
        let byte = buf[pos];
        pos += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            break;
        }
    }

    (value, pos - start)
}

fn skip_field(buf: &[u8], pos: usize, wire_type: u8) -> usize {
    match wire_type {
        0 => {
            // Varint
            let mut p = pos;
            while p < buf.len() && buf[p] & 0x80 != 0 {
                p += 1;
            }
            p + 1
        }
        1 => pos + 8,  // 64-bit
        2 => {
            // Length-delimited
            let (len, len_len) = read_varint(buf, pos);
            pos + len_len + len as usize
        }
        5 => pos + 4,  // 32-bit
        _ => pos + 1,  // Unknown, skip 1 byte
    }
}

// ---------------------------------------------------------------------------
// Token decoding
// ---------------------------------------------------------------------------

/// Decode token IDs to text using the SentencePiece vocabulary.
fn decode_tokens(vocab: &[String], token_ids: &[u32]) -> String {
    let mut pieces = Vec::new();
    let mut byte_buffer: Vec<u8> = Vec::new();

    let flush_bytes = |byte_buffer: &mut Vec<u8>, pieces: &mut Vec<String>| {
        if !byte_buffer.is_empty() {
            if let Ok(s) = String::from_utf8(byte_buffer.clone()) {
                pieces.push(s);
            }
            byte_buffer.clear();
        }
    };

    for &id in token_ids {
        // Skip special tokens (EOS=0, padding=3)
        if id == 0 || id == 3 {
            continue;
        }

        if let Some(piece) = vocab.get(id as usize) {
            // Detect byte fallback tokens: <0xHH>
            if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
                if let Ok(byte_val) = u8::from_str_radix(&piece[3..5], 16) {
                    byte_buffer.push(byte_val);
                    continue;
                }
            }

            flush_bytes(&mut byte_buffer, &mut pieces);
            pieces.push(piece.clone());
        }
    }

    flush_bytes(&mut byte_buffer, &mut pieces);

    // Join and convert SentencePiece underscores to spaces.
    // Do NOT trim: the leading space from \u{2581} is a word boundary signal
    // needed for correct streaming (per-token) decoding.
    let text = pieces.join("");
    text.replace('\u{2581}', " ")
}
