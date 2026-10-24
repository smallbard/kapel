# 9. Text: kapel-tokenizers

`kapel-tokenizers` turns text into token ids and back, and renders conversations with a model's chat template. It reads the `tokenizer.json` files of the Hugging Face `tokenizers` library and aims for the same ids on the same text; its parity tests compare 260 texts over 13 tokenizers. It depends on no other Kapel crate.

## 9.1 What is supported

Kapel implements the BPE model, which covers the GPT-2 family (byte-level BPE, as in GPT-2, Llama 3, Qwen) and SentencePiece-style tokenizers written as BPE with byte fallback (Llama 2, Mistral, Gemma). WordPiece and Unigram models are not supported, and a file that uses them is rejected with an error. A model without a `type` but with merges is a BPE, as `tokenizers` reads it: GPT-2's own file on the Hub is written so.

`Tokenizer::from_file` reads the six parts of a `tokenizer.json`: the model, the added tokens, the normalizer, the pre-tokenizer, the post-processor and the decoder.

## 9.2 The encoding pipeline

1. **Cut out added tokens.** Special tokens such as `<|im_start|>` must never be split by BPE. They are found with an Aho-Corasick automaton in leftmost-longest mode, which gives the same result as scanning for each token at each position, but in one pass. `lstrip` and `rstrip` options swallow surrounding whitespace.
2. **Normalize** each remaining segment: Unicode NFC, NFD, NFKC or NFKD, lowercasing, prepending, replacing, stripping, or a sequence of these.
3. **Pre-tokenize** the segment into pieces. *ByteLevel* applies the GPT-2 regular expression, then maps every byte to a printable character (a space becomes `Ġ`, a newline `Ċ`). *Split* cuts on a string or a regular expression, with the behaviours of `tokenizers` (removed, isolated, merged with the previous or next piece). *Metaspace* replaces spaces with `▁`. *Digits* isolates numbers.
4. **Apply BPE** to each piece (9.3).
5. **Post-process**: add the special tokens of the template, such as a beginning-of-sequence token, when the caller asks for special tokens.

`encode_ordinary` skips the first and the last steps: the whole text is plain text, and an added token it spells, such as `<|endoftext|>`, is encoded as its characters (tiktoken's `encode_ordinary`). Training corpora are encoded this way (section 15.13): a document that quotes a special token cannot inject it.

Regular expressions are compiled once, when the tokenizer is loaded, with `fancy-regex`, which supports the look-ahead that some tokenizers use. If an expression gives up on a pathological input because it hits its backtracking limit, the text is rejected with an error rather than silently cut in the wrong place.

## 9.3 Byte-pair encoding in O(n log n)

BPE starts from the characters of a piece and repeatedly merges the adjacent pair with the best rank in the merge list, until no listed pair remains. A naive loop rescans the whole piece after each merge, which is quadratic: a hostile text of one long word could stall it.

Kapel keeps the symbols in a linked list and the candidate pairs in a priority queue ordered by rank, then position:

```rust
// each symbol: its text, its id, alive or merged; next[] and prev[] link the live ones
let mut heap: BinaryHeap<Reverse<(rank, left, merged)>> = /* every adjacent pair */;
while let Some(Reverse((rank, left, merged))) = heap.pop() {
    // skip stale entries: the pair at `left` may have changed since it was pushed
    // merge, then push the new pairs with the left and right neighbours
}
```

Each merge costs a logarithmic heap operation, so a piece of n characters costs O(n log n). A regression test encodes a word of 300,000 characters in under five seconds.

When a symbol is not in the vocabulary, byte fallback encodes it as `<0xHH>` tokens; otherwise it becomes the unknown token, if the tokenizer has one. Results are cached per piece, up to 100,000 entries, because natural text repeats its words.

## 9.4 Decoding

Decoders undo the pre-tokenizer: ByteLevel maps characters back to bytes, Metaspace turns `▁` back into spaces, ByteFallback reassembles `<0xHH>` runs into UTF-8, and Replace, Fuse and Strip clean up. Invalid bytes become U+FFFD rather than an error, so decoding never fails. `decode(ids, skip_special_tokens)` optionally drops special tokens. The generation engine relies on this to stream text (chapter 11).

**The bytes of one token.** `token_bytes(id)` gives what a token adds to the decoded text when it stands in the middle of it, as bytes: a token of a byte-level vocabulary may be part of a character, so it is not always text. The decoder's steps run on the one token, at the level of bytes: ByteLevel maps its characters to bytes, ByteFallback turns `<0xHH>` into its byte, Metaspace and Replace turn `▁` into a space. The steps that act only at the ends of a whole text, Strip and the space Metaspace takes off a first token, leave the token unchanged. A special token gives its content as it is; `is_special(id)` says which tokens are. Constrained generation (chapter 11) builds its tree of tokens from these bytes. The parity test checks, for every case of every tokenizer, that the bytes of its tokens, one after the other, are the decoded text, but for a leading space a first-token rule removes.

## 9.5 Chat templates

A chat model expects its conversations in a precise format, written by its authors as a Jinja template in `chat_template.jinja` or in `tokenizer_config.json`. Kapel renders it with `minijinja`, plus the pieces that `transformers` templates assume:

- Python string methods (`.strip()`, `.startswith()` and so on), through `minijinja-contrib`'s compatibility layer.
- `raise_exception(message)`, which turns into a template error.
- `strftime_now(format)`, the current local date and time; Llama 3 templates print today's date.
- A `tojson` filter that writes JSON exactly as Python's `json.dumps` does, separators and all.
- The `{% generation %}` tags of training templates, rendered as plain blocks.

A template comes from the model's files, so it is untrusted code. Rendering is bounded by fuel (10 million steps) and by a cap of 1 MiB on the output; minijinja also refuses a single repeated string above 100 MB. `SECURITY.md` states what remains unbounded: a string built by repeated concatenation is limited by fuel, not by size.

For tests, `with_fixed_time` pins the clock, so renderings that contain a date compare equal to the reference produced by `tools/parity/chat_reference.py`, which pins the same date on the Python side.
