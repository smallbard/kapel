//! The parameters of every kernel of `kernels.ptx`, as the backend passes them.
//!
//! The kernels are compiled apart (`kernels/build.sh`) and loaded by name at
//! run time: nothing checks at compile time that a launch passes what the
//! kernel expects. This table is the backend's side of that contract. Every
//! launch is checked against it, and a test checks it against the parameter
//! lists of the PTX, so that a kernel changed without rebuilding the PTX, or
//! the PTX without updating the backend, fails the tests.

// Without the feature `cuda`, only the test reads the table.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

/// One kernel argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Param {
    /// A `u32` by value.
    U32,
    /// An `f32` by value.
    F32,
    /// A slice (`&[T]` or `DisjointSlice<T>`): two parameters, its device
    /// pointer, then its length in elements, both 64 bits.
    Slice,
    /// A [`kapel_cuda_abi::View`] by value.
    View,
    /// A [`kapel_cuda_abi::Paged`] by value.
    Paged,
}

use Param::{F32, Paged, Slice, U32, View};

/// Every kernel of `kernels.ptx`, with its arguments in order.
pub(crate) const KERNELS: &[(&str, &[Param])] = &[
    ("copy_view_8", &[Slice, View, Slice]),
    ("copy_view_16", &[Slice, View, Slice]),
    ("copy_view_32", &[Slice, View, Slice]),
    ("copy_view_64", &[Slice, View, Slice]),
    ("copy_into_8", &[Slice, View, Slice, View, U32]),
    ("copy_into_16", &[Slice, View, Slice, View, U32]),
    ("copy_into_32", &[Slice, View, Slice, View, U32]),
    ("copy_into_64", &[Slice, View, Slice, View, U32]),
    ("widen_16", &[U32, Slice, View, Slice]),
    ("narrow_16", &[U32, Slice, Slice]),
    ("unary_forward", &[U32, Slice, Slice]),
    ("unary_backward", &[U32, Slice, Slice, Slice]),
    ("affine", &[F32, F32, Slice, Slice]),
    ("binary_forward", &[U32, Slice, View, Slice, View, Slice]),
    (
        "binary_backward",
        &[U32, U32, Slice, View, Slice, View, Slice, Slice],
    ),
    ("reduce_sum", &[Slice, Slice, Slice, F32, Slice]),
    ("spread", &[Slice, Slice, F32, Slice]),
    ("line_max", &[Slice, Slice, U32, U32, Slice, Slice]),
    ("place", &[Slice, Slice, Slice]),
    ("softmax_lines", &[U32, Slice, Slice, U32, U32, Slice]),
    (
        "softmax_lines_backward",
        &[U32, Slice, Slice, Slice, U32, U32, Slice],
    ),
    (
        "norm_rows",
        &[U32, Slice, Slice, Slice, F32, F32, U32, Slice],
    ),
    (
        "norm_rows_backward",
        &[U32, Slice, Slice, Slice, F32, F32, U32, Slice, Slice, Slice],
    ),
    ("reduce_sum_segments", &[Slice, Slice, Slice, U32, Slice]),
    ("cross_entropy_rows", &[U32, Slice, View, Slice, U32, Slice]),
    (
        "cross_entropy_rows_backward",
        &[U32, Slice, View, Slice, Slice, U32, U32, Slice],
    ),
    ("rope_table", &[Slice, U32, Slice, F32, Slice]),
    ("rope", &[Slice, Slice, Slice, U32, U32, U32, Slice]),
    ("gather_rows", &[Slice, Slice, U32, Slice]),
    ("gather_rows_16", &[Slice, Slice, U32, Slice]),
    ("scatter_rows", &[Slice, Slice, Slice, U32, Slice]),
    ("unary_any", &[U32, U32, Slice, View, U32, Slice]),
    ("affine_any", &[F32, F32, U32, Slice, View, U32, Slice]),
    (
        "binary_any",
        &[U32, U32, Slice, View, U32, Slice, View, U32, Slice],
    ),
    (
        "norm_any",
        &[U32, U32, Slice, Slice, Slice, F32, F32, U32, Slice],
    ),
    (
        "norm_block_any",
        &[U32, U32, Slice, Slice, Slice, F32, F32, U32, Slice],
    ),
    (
        "rope_any",
        &[U32, Slice, View, Slice, Slice, U32, U32, U32, U32, Slice],
    ),
    (
        "write_rows_any",
        &[U32, Slice, View, Slice, U32, U32, Slice],
    ),
    (
        "paged_attention",
        &[
            Paged, Slice, Slice, Slice, Slice, Slice, Slice, Slice, Slice,
        ],
    ),
    ("paged_combine", &[Paged, Slice, Slice, Slice]),
    (
        "norm_16",
        &[U32, U32, Slice, Slice, Slice, F32, F32, U32, Slice],
    ),
    ("glu_any", &[U32, U32, Slice, U32, U32, Slice]),
    ("glu_backward", &[U32, Slice, Slice, U32, Slice]),
    ("argmax_rows", &[U32, Slice, U32, Slice]),
    (
        "paged_attention_16",
        &[
            Paged, U32, Slice, Slice, Slice, Slice, Slice, Slice, Slice, Slice,
        ],
    ),
    (
        "qgemv",
        &[U32, U32, U32, Slice, U32, Slice, U32, U32, U32, U32, Slice],
    ),
    ("dequantize", &[U32, U32, U32, Slice, U32, U32, Slice]),
    (
        "dequantize_rows",
        &[U32, U32, U32, Slice, Slice, U32, Slice],
    ),
    ("argmax_parts", &[U32, Slice, U32, U32, Slice, Slice]),
    ("argmax_merge", &[U32, Slice, Slice, Slice]),
    (
        "qgemv_int8",
        &[U32, U32, Slice, U32, Slice, U32, U32, U32, Slice],
    ),
    (
        "qgemv_int8_1",
        &[U32, U32, Slice, U32, Slice, U32, U32, Slice],
    ),
    (
        "qgemv_int4",
        &[U32, U32, Slice, U32, Slice, U32, U32, U32, Slice],
    ),
    (
        "qgemv_int4_1",
        &[U32, U32, Slice, U32, Slice, U32, U32, Slice],
    ),
    (
        "qgemv_q8_0",
        &[U32, U32, Slice, U32, Slice, U32, U32, U32, Slice],
    ),
    (
        "qgemv_q8_0_1",
        &[U32, U32, Slice, U32, Slice, U32, U32, Slice],
    ),
    (
        "qgemv_q4_0",
        &[U32, U32, Slice, U32, Slice, U32, U32, U32, Slice],
    ),
    (
        "qgemv_q4_0_1",
        &[U32, U32, Slice, U32, Slice, U32, U32, Slice],
    ),
    (
        "qgemv_q4_1",
        &[U32, U32, Slice, U32, Slice, U32, U32, U32, Slice],
    ),
    (
        "qgemv_q4_1_1",
        &[U32, U32, Slice, U32, Slice, U32, U32, Slice],
    ),
    (
        "qgemv_q4k",
        &[U32, U32, Slice, U32, Slice, U32, U32, U32, Slice],
    ),
    (
        "qgemv_q4k_1",
        &[U32, U32, Slice, U32, Slice, U32, U32, Slice],
    ),
    (
        "qgemv_q6k",
        &[U32, U32, Slice, U32, Slice, U32, U32, U32, Slice],
    ),
    (
        "qgemv_q6k_1",
        &[U32, U32, Slice, U32, Slice, U32, U32, Slice],
    ),
];

/// The arguments of kernel `name`, if there is one.
pub(crate) fn of(name: &str) -> Option<&'static [Param]> {
    KERNELS
        .iter()
        .find(|(kernel, _)| *kernel == name)
        .map(|(_, params)| *params)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use kapel_cuda_abi::View as AbiView;

    use super::*;

    /// One `.param` of a PTX entry, as far as its layout goes.
    #[derive(Debug, PartialEq, Eq)]
    enum Ptx {
        U32,
        F32,
        Pointer,
        U64,
        Bytes { size: usize, align: usize },
    }

    /// What each argument becomes in PTX.
    fn expected(params: &[Param]) -> Vec<Ptx> {
        params
            .iter()
            .flat_map(|p| match p {
                Param::U32 => vec![Ptx::U32],
                Param::F32 => vec![Ptx::F32],
                Param::Slice => vec![Ptx::Pointer, Ptx::U64],
                Param::View => vec![Ptx::Bytes {
                    size: size_of::<AbiView>(),
                    align: align_of::<AbiView>(),
                }],
                Param::Paged => vec![Ptx::Bytes {
                    size: size_of::<kapel_cuda_abi::Paged>(),
                    align: align_of::<kapel_cuda_abi::Paged>(),
                }],
            })
            .collect()
    }

    fn parse_param(line: &str) -> Ptx {
        let words: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(words.first(), Some(&".param"), "not a parameter: {line}");
        match words[1] {
            ".u32" => Ptx::U32,
            ".f32" => Ptx::F32,
            ".u64" if words[2] == ".ptr" => Ptx::Pointer,
            ".u64" => Ptx::U64,
            ".align" => {
                let align = words[2].parse().expect("an alignment");
                assert_eq!(words[3], ".b8", "a byte array: {line}");
                let size = words[4]
                    .split_once('[')
                    .and_then(|(_, rest)| rest.strip_suffix(']'))
                    .and_then(|n| n.parse().ok())
                    .unwrap_or_else(|| panic!("a byte array size: {line}"));
                Ptx::Bytes { size, align }
            }
            other => panic!("unexpected parameter type {other}: {line}"),
        }
    }

    /// The parameters of every `.entry` of a PTX module.
    fn entries(ptx: &str) -> BTreeMap<String, Vec<Ptx>> {
        let mut entries = BTreeMap::new();
        let mut lines = ptx.lines();
        while let Some(line) = lines.next() {
            let Some(rest) = line.trim().strip_prefix(".visible .entry ") else {
                continue;
            };
            let name = rest.trim_end_matches('(').to_string();
            let mut params = Vec::new();
            for line in lines.by_ref() {
                let line = line.trim();
                if line.starts_with(')') {
                    break;
                }
                params.push(parse_param(line.trim_end_matches(',')));
            }
            entries.insert(name, params);
        }
        entries
    }

    #[test]
    fn the_table_matches_the_ptx() {
        let ptx = include_str!("../kernels.ptx");
        let entries = entries(ptx);
        let names: Vec<&str> = KERNELS.iter().map(|(name, _)| *name).collect();
        let in_ptx: Vec<&str> = entries.keys().map(String::as_str).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, in_ptx, "the kernels of the table and of the PTX");
        for (name, params) in KERNELS {
            assert_eq!(
                entries[*name],
                expected(params),
                "kernel {name}: the PTX parameters differ from the table; \
                 rebuild the PTX with kernels/build.sh or update the table"
            );
        }
    }

    #[test]
    fn the_parser_reads_every_kind_of_parameter() {
        let ptx = ".visible .entry k(\n\t.param .u32 k_param_0,\n\t.param .f32 k_param_1,\n\
                   \t.param .u64 .ptr .align 4 k_param_2,\n\t.param .u64 k_param_3,\n\
                   \t.param .align 4 .b8 k_param_4[68]\n)\n";
        assert_eq!(
            entries(ptx)["k"],
            [
                Ptx::U32,
                Ptx::F32,
                Ptx::Pointer,
                Ptx::U64,
                Ptx::Bytes { size: 68, align: 4 }
            ]
        );
    }

    #[test]
    fn a_view_is_68_bytes_aligned_on_4() {
        // The PTX hard-codes this layout; changing `View` needs new kernels.
        assert_eq!(size_of::<AbiView>(), 68);
        assert_eq!(align_of::<AbiView>(), 4);
    }
}
