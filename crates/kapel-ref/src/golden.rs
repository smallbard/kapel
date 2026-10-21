//! Golden blocks for the GGML formats that the reference reads through
//! `Tensor::to_f64_vec`: one block per format, built by hand, decoded by a
//! transcription of GGML's own `dequantize_row_*` loops (which walk the output
//! in order, unlike the per-element formulas of `kapel-core`), plus values
//! worked out by hand.

use kapel_core::{DType, QuantFormat, Tensor};

use crate::rng::Rng;

/// The values of one block of `format`, as the reference reads them.
fn decoded(format: QuantFormat, bytes: &[u8]) -> Vec<f64> {
    let dtype = DType::Quantized(format);
    Tensor::from_bytes(dtype, &[dtype.block_len()], bytes)
        .expect("one whole block")
        .to_f64_vec()
        .expect("a decoded format")
}

/// Random bytes, then the given little-endian `f16` fields written at their
/// offsets.
fn block(len: usize, seed: u64, halves: &[(usize, u16)]) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut bytes: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
    for &(at, bits) in halves {
        bytes[at..at + 2].copy_from_slice(&bits.to_le_bytes());
    }
    bytes
}

// Exact f16 values: 0.5, 0.25, −1.5, 2.0.
const HALF: u16 = 0x3800;
const QUARTER: u16 = 0x3400;
const MINUS_ONE_AND_HALF: u16 = 0xbe00;
const TWO: u16 = 0x4000;

fn assert_same(format: QuantFormat, got: &[f64], expected: &[f64]) {
    assert_eq!(got.len(), expected.len(), "{format:?}");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(g, e, "{format:?}, element {i}");
    }
}

#[test]
fn q4_1_is_scale_times_code_plus_minimum() {
    // block_q4_1 { d, m, qs[16] }: y[j] = qs[j] & 15, y[j + 16] = qs[j] >> 4.
    let (d, m) = (0.5, -1.5);
    let bytes = block(20, 1, &[(0, HALF), (2, MINUS_ONE_AND_HALF)]);
    let qs = &bytes[4..];
    let mut y = vec![0.0; 32];
    for j in 0..16 {
        y[j] = f64::from(qs[j] & 0x0f) * d + m;
        y[j + 16] = f64::from(qs[j] >> 4) * d + m;
    }
    let got = decoded(QuantFormat::Q4_1, &bytes);
    assert_same(QuantFormat::Q4_1, &got, &y);
    let mut by_hand = [HALF.to_le_bytes(), MINUS_ONE_AND_HALF.to_le_bytes()].concat();
    by_hand.extend([0xf0, 0x21]);
    by_hand.extend([0; 14]);
    let got = decoded(QuantFormat::Q4_1, &by_hand);
    // Code 0 → −1.5; code 1 → −1; code 15 → 6; code 2 → −0.5.
    assert_eq!(
        (got[0], got[1], got[16], got[17], got[2]),
        (-1.5, -1.0, 6.0, -0.5, -1.5)
    );
}

/// GGML's `dequantize_row_q5_0` and `_q5_1`: the fifth bit of element `j` is
/// bit `j` of `qh`, read as `(qh >> j) << 4 & 0x10` for the first half and
/// `(qh >> (j + 12)) & 0x10` for the second.
fn q5(bytes: &[u8], d: f64, m: Option<f64>) -> Vec<f64> {
    let rest = if m.is_some() {
        &bytes[4..]
    } else {
        &bytes[2..]
    };
    let qh = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
    let qs = &rest[4..];
    let mut y = vec![0.0; 32];
    for j in 0..16 {
        let xh_0 = ((qh >> j) << 4) & 0x10;
        let xh_1 = (qh >> (j + 12)) & 0x10;
        let x0 = f64::from(u32::from(qs[j] & 0x0f) | xh_0);
        let x1 = f64::from(u32::from(qs[j] >> 4) | xh_1);
        match m {
            Some(m) => {
                y[j] = x0 * d + m;
                y[j + 16] = x1 * d + m;
            }
            None => {
                y[j] = (x0 - 16.0) * d;
                y[j + 16] = (x1 - 16.0) * d;
            }
        }
    }
    y
}

#[test]
fn q5_0_and_q5_1_take_their_fifth_bit_from_qh() {
    let bytes = block(22, 2, &[(0, HALF)]);
    assert_same(
        QuantFormat::Q5_0,
        &decoded(QuantFormat::Q5_0, &bytes),
        &q5(&bytes, 0.5, None),
    );
    let bytes = block(24, 3, &[(0, QUARTER), (2, MINUS_ONE_AND_HALF)]);
    assert_same(
        QuantFormat::Q5_1,
        &decoded(QuantFormat::Q5_1, &bytes),
        &q5(&bytes, 0.25, Some(-1.5)),
    );
    // By hand: qh with only bit 17 set; every code 0 but qs[1] = 0x00.
    let mut by_hand = HALF.to_le_bytes().to_vec();
    by_hand.extend((1u32 << 17).to_le_bytes());
    by_hand.extend([0; 16]);
    let got = decoded(QuantFormat::Q5_0, &by_hand);
    // Element 17 has code 16 → 0; the others code 0 → −8.
    assert_eq!((got[17], got[1], got[16]), (0.0, -8.0, -8.0));
}

#[test]
fn q2k_follows_ggml() {
    // block_q2_K { scales[16], qs[64], d, dmin }.
    let (d, dmin) = (0.5, 0.25);
    let bytes = block(84, 4, &[(80, HALF), (82, QUARTER)]);
    let (scales, qs) = (&bytes[..16], &bytes[16..80]);
    let mut y = Vec::new();
    let mut is = 0;
    for n in 0..2 {
        let q = &qs[32 * n..];
        for j in 0..4 {
            let shift = 2 * j;
            for half in [0, 16] {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * f64::from(sc & 0x0f), dmin * f64::from(sc >> 4));
                for l in 0..16 {
                    y.push(dl * f64::from((q[l + half] >> shift) & 3) - ml);
                }
            }
        }
    }
    assert_same(QuantFormat::Q2K, &decoded(QuantFormat::Q2K, &bytes), &y);
}

#[test]
fn q3k_follows_ggml() {
    // block_q3_K { hmask[32], qs[64], scales[12], d }.
    let d = 2.0;
    let bytes = block(110, 5, &[(108, TWO)]);
    let (hmask, qs, packed) = (&bytes[..32], &bytes[32..96], &bytes[96..108]);
    // The scale unpacking of GGML, on 32-bit words.
    let word =
        |i: usize| u32::from_le_bytes([packed[i], packed[i + 1], packed[i + 2], packed[i + 3]]);
    let (kmask1, kmask2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let mut aux = [word(0), word(4), word(8), 0];
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let scales: Vec<i8> = aux
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .map(|b| b as i8)
        .collect();
    let mut y = Vec::new();
    let (mut is, mut m) = (0, 1u8);
    for n in 0..2 {
        let q = &qs[32 * n..];
        for j in 0..4 {
            let shift = 2 * j;
            for half in [0, 16] {
                let dl = d * f64::from(i32::from(scales[is]) - 32);
                is += 1;
                for l in 0..16 {
                    let low = i32::from((q[l + half] >> shift) & 3);
                    let sub = if hmask[l + half] & m != 0 { 0 } else { 4 };
                    y.push(dl * f64::from(low - sub));
                }
            }
            m <<= 1;
        }
    }
    assert_same(QuantFormat::Q3K, &decoded(QuantFormat::Q3K, &bytes), &y);
}

/// GGML's `get_scale_min_k4`.
fn scale_min_k4(j: usize, q: &[u8]) -> (f64, f64) {
    if j < 4 {
        (f64::from(q[j] & 63), f64::from(q[j + 4] & 63))
    } else {
        (
            f64::from((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4)),
            f64::from((q[j + 4] >> 4) | ((q[j] >> 6) << 4)),
        )
    }
}

#[test]
fn q4k_follows_ggml() {
    // block_q4_K { d, dmin, scales[12], qs[128] }.
    let (d, dmin) = (0.25, 0.5);
    let bytes = block(144, 6, &[(0, QUARTER), (2, HALF)]);
    let (scales, qs) = (&bytes[4..16], &bytes[16..]);
    let mut y = Vec::new();
    for (chunk, is) in (0..4).map(|c| (&qs[32 * c..32 * c + 32], 2 * c)) {
        let (sc1, m1) = scale_min_k4(is, scales);
        let (sc2, m2) = scale_min_k4(is + 1, scales);
        y.extend(
            chunk
                .iter()
                .map(|q| d * sc1 * f64::from(q & 0x0f) - dmin * m1),
        );
        y.extend(
            chunk
                .iter()
                .map(|q| d * sc2 * f64::from(q >> 4) - dmin * m2),
        );
    }
    assert_same(QuantFormat::Q4K, &decoded(QuantFormat::Q4K, &bytes), &y);
}

#[test]
fn q5k_follows_ggml() {
    // block_q5_K { d, dmin, scales[12], qh[32], qs[128] }.
    let (d, dmin) = (0.5, 0.25);
    let bytes = block(176, 7, &[(0, HALF), (2, QUARTER)]);
    let (scales, qh, qs) = (&bytes[4..16], &bytes[16..48], &bytes[48..]);
    let mut y = Vec::new();
    let (mut u1, mut u2) = (1u8, 2u8);
    for c in 0..4 {
        let ql = &qs[32 * c..];
        let (sc1, m1) = scale_min_k4(2 * c, scales);
        let (sc2, m2) = scale_min_k4(2 * c + 1, scales);
        for l in 0..32 {
            let high = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            y.push(d * sc1 * (f64::from(ql[l] & 0x0f) + high) - dmin * m1);
        }
        for l in 0..32 {
            let high = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            y.push(d * sc2 * (f64::from(ql[l] >> 4) + high) - dmin * m2);
        }
        u1 <<= 2;
        u2 <<= 2;
    }
    assert_same(QuantFormat::Q5K, &decoded(QuantFormat::Q5K, &bytes), &y);
}

#[test]
fn q6k_follows_ggml() {
    // block_q6_K { ql[128], qh[64], scales[16] (int8), d }.
    let d = 0.5;
    let bytes = block(210, 8, &[(208, HALF)]);
    let (ql, qh, sc) = (&bytes[..128], &bytes[128..192], &bytes[192..208]);
    let mut y = vec![0.0; 256];
    for n in 0..2 {
        let (ql, qh, sc, y) = (
            &ql[64 * n..],
            &qh[32 * n..],
            &sc[8 * n..],
            &mut y[128 * n..],
        );
        for l in 0..32 {
            let is = l / 16;
            let code = |low: u8, high: u8| i32::from(low | (high & 3) << 4) - 32;
            let q1 = code(ql[l] & 0x0f, qh[l]);
            let q2 = code(ql[l + 32] & 0x0f, qh[l] >> 2);
            let q3 = code(ql[l] >> 4, qh[l] >> 4);
            let q4 = code(ql[l + 32] >> 4, qh[l] >> 6);
            let s = |i: usize| f64::from(sc[i] as i8);
            y[l] = d * s(is) * f64::from(q1);
            y[l + 32] = d * s(is + 2) * f64::from(q2);
            y[l + 64] = d * s(is + 4) * f64::from(q3);
            y[l + 96] = d * s(is + 6) * f64::from(q4);
        }
    }
    assert_same(QuantFormat::Q6K, &decoded(QuantFormat::Q6K, &bytes), &y);
    // By hand: all codes 0 (q = −32), scale 0 is −2, the others 1.
    let mut by_hand = vec![0u8; 210];
    by_hand[192..208].fill(1);
    by_hand[192] = (-2i8) as u8;
    by_hand[208..].copy_from_slice(&HALF.to_le_bytes());
    let got = decoded(QuantFormat::Q6K, &by_hand);
    assert_eq!(
        (got[0], got[15], got[16], got[255]),
        (32.0, 32.0, -16.0, -16.0)
    );
}
