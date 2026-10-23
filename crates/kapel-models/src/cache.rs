//! The cache of generation: the keys and values of the positions already
//! seen, for every sequence in progress, kept by pages.
//!
//! Each attention layer keeps two tables, `[lines, kv_heads, head_dim]`, on
//! the backend, written in place ([`kapel_ops::Write`]). A sequence does not
//! own neighbouring lines: lines are handed out by pages of [`PAGE`], taken
//! from a common reserve as the sequence grows and given back when it ends, as
//! an operating system hands out memory. Each sequence keeps the list of its
//! pages, and the attention finds its keys through it
//! ([`kapel_ops::PagedAttention`]). Writing a position is a write in place,
//! into a free line: nothing is copied as a sequence grows, and ending one
//! only touches lists.
//!
//! A cache made by [`KvCache::new`] grows when a sequence needs more pages
//! than are free: its tables are copied into tables twice as large, and the
//! pages keep their numbers. One made by [`KvCache::with_pages`] has a fixed
//! number of pages, as a server sizes it for the memory of its device, and
//! refuses positions beyond them ([`ModelError::CacheFull`]).

use std::sync::atomic::{AtomicU64, Ordering};

use kapel_core::{DType, Tensor};
use kapel_ops::{Transfer, Write};

use crate::ModelError;

/// Lines of a page.
pub const PAGE: usize = 16;

/// A sequence followed by a [`KvCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SequenceId(usize);

/// What the cache knows of a sequence.
#[derive(Clone, Debug, Default)]
struct Sequence {
    /// Its pages, in the order of its positions.
    pages: Vec<u32>,
    /// The positions it holds.
    len: usize,
}

/// The keys and values of every sequence in progress, by pages.
#[derive(Debug)]
pub struct KvCache {
    /// Per layer, the keys and values `[lines, kv_heads, head_dim]`, made at
    /// the first write, when their widths are known.
    layers: Vec<Option<(Tensor, Tensor)>>,
    /// The pages of the tables: they have `pages · PAGE` lines.
    pages: usize,
    /// The free pages; the next one given is the last.
    free: Vec<u32>,
    /// Indexed by [`SequenceId`]; `None` for a sequence that ended.
    sequences: Vec<Option<Sequence>>,
    /// Whether the cache grows when it is full.
    fixed: bool,
    /// The type of the tables; `None` for that of the keys.
    dtype: Option<DType>,
    /// What tells this cache's tables from those of any other cache, and from
    /// those it had before it last made or grew one: a recorded pass writes
    /// into the tables it was recorded with.
    identity: (u64, u64),
}

impl Default for KvCache {
    fn default() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self {
            layers: Vec::new(),
            pages: 0,
            free: Vec::new(),
            sequences: Vec::new(),
            fixed: false,
            dtype: None,
            identity: (NEXT.fetch_add(1, Ordering::Relaxed), 0),
        }
    }
}

/// The host side of a pass: what [`Step`] holds, before it is uploaded.
pub(crate) struct Plan {
    pub(crate) positions: Vec<u32>,
    pub(crate) lines: Vec<u32>,
    /// `[S, widest]`.
    pub(crate) pages: Vec<u32>,
    pub(crate) widest: usize,
    /// `[R, 2]`.
    pub(crate) rows: Vec<u32>,
}

/// What the attention layers of one pass need to know of its tokens: where to
/// write their keys and values, and which positions each one sees. The same
/// for every layer.
pub(crate) struct Step {
    /// `[R]`: the position of each new token in its sequence.
    pub(crate) positions: Tensor,
    /// `[R]`: the line of the tables where each new token is written.
    pub(crate) lines: Tensor,
    /// `[S, P]`: the pages of each sequence of the pass, in order.
    pub(crate) pages: Tensor,
    /// `[R, 2]`: the sequence of each new token in the pass, and the number
    /// of positions it sees, its own included.
    pub(crate) rows: Tensor,
    /// The lines the tables must have.
    pub(crate) lines_needed: usize,
}

impl KvCache {
    /// An empty cache, which grows as its sequences do.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty cache of `pages` pages of [`PAGE`] lines, which does not
    /// grow: what is left of them is [`KvCache::free_pages`].
    pub fn with_pages(pages: usize) -> Self {
        Self {
            pages,
            free: (0..pages as u32).rev().collect(),
            fixed: true,
            ..Self::default()
        }
    }

    /// The sequences in progress.
    pub fn sequences(&self) -> impl Iterator<Item = SequenceId> + '_ {
        self.sequences
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_some())
            .map(|(i, _)| SequenceId(i))
    }

    /// The same cache, with tables of type `dtype` (`f16` or `f8e4m3`, say)
    /// instead of that of the keys: the keys and values are rounded to it as
    /// they are written. Set it before the first pass.
    #[must_use]
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Starts a new, empty sequence.
    pub fn add_sequence(&mut self) -> SequenceId {
        let index = match self.sequences.iter().position(Option::is_none) {
            Some(index) => index,
            None => {
                self.sequences.push(None);
                self.sequences.len() - 1
            }
        };
        self.sequences[index] = Some(Sequence::default());
        SequenceId(index)
    }

    /// Ends a sequence: its pages go back to the reserve.
    ///
    /// # Panics
    ///
    /// If the sequence already ended.
    pub fn remove_sequence(&mut self, id: SequenceId) {
        let sequence = self.sequences[id.0].take().expect("a sequence in progress");
        self.free.extend(sequence.pages.iter().rev());
    }

    fn sequence(&self, id: SequenceId) -> &Sequence {
        self.sequences[id.0]
            .as_ref()
            .expect("a sequence in progress")
    }

    /// The positions the sequence holds.
    ///
    /// # Panics
    ///
    /// If the sequence ended.
    pub fn len(&self, id: SequenceId) -> usize {
        self.sequence(id).len
    }

    /// Whether the sequence holds no position.
    ///
    /// # Panics
    ///
    /// If the sequence ended.
    pub fn is_empty(&self, id: SequenceId) -> bool {
        self.len(id) == 0
    }

    /// The pages not given to any sequence.
    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    /// The pages of the tables.
    pub fn pages(&self) -> usize {
        self.pages
    }

    /// The pages that `count` more positions of the sequence would take from
    /// the reserve.
    ///
    /// # Panics
    ///
    /// If the sequence ended.
    pub fn pages_needed(&self, id: SequenceId, count: usize) -> usize {
        let sequence = self.sequence(id);
        (sequence.len + count)
            .div_ceil(PAGE)
            .saturating_sub(sequence.pages.len())
    }

    /// Brings the sequence back to its first `len` positions, as if the next
    /// ones had never been written: speculative decoding takes back the
    /// tokens it refused. The pages left empty go back to the reserve.
    ///
    /// # Panics
    ///
    /// If the sequence ended, or holds fewer than `len` positions.
    pub fn truncate(&mut self, id: SequenceId, len: usize) {
        let sequence = self.sequences[id.0]
            .as_mut()
            .expect("a sequence in progress");
        assert!(len <= sequence.len, "a sequence can only be shortened");
        sequence.len = len;
        let freed: Vec<u32> = sequence.pages.drain(len.div_ceil(PAGE)..).collect();
        self.free.extend(freed.iter().rev());
    }

    /// The bytes the tables hold, or will hold once a pass has made them:
    /// what each position costs times the lines.
    pub fn table_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|(k, v)| k.storage().len() + v.storage().len())
            .sum()
    }

    /// Whether the first `layers` layers have their tables.
    pub(crate) fn has_tables(&self, layers: usize) -> bool {
        self.layers.len() >= layers && self.layers[..layers].iter().all(Option::is_some)
    }

    /// What tells this cache and its present tables from any other.
    pub(crate) fn identity(&self) -> (u64, u64) {
        self.identity
    }

    /// The pages per sequence a pass of `batch` needs in its table of pages.
    fn widest(&self, batch: &[(SequenceId, usize)]) -> usize {
        batch
            .iter()
            .map(|&(id, count)| (self.len(id) + count).div_ceil(PAGE))
            .max()
            .unwrap_or(0)
            .max(1)
    }

    /// Whether a pass of `batch` fits a table of `widest` pages per sequence,
    /// with the free pages, without growing the tables.
    pub(crate) fn fits(&self, batch: &[(SequenceId, usize)], widest: usize) -> bool {
        let needed: usize = batch
            .iter()
            .map(|&(id, count)| self.pages_needed(id, count))
            .sum();
        batch
            .iter()
            .all(|&(id, _)| self.sequences.get(id.0).is_some_and(Option::is_some))
            && self.widest(batch) <= widest
            && needed <= self.free.len()
    }

    /// Makes room for `batch` (each sequence and the positions it will add):
    /// free pages for all of them, the cache grown if it may, and every table
    /// made so far grown to the pages. A recorded pass then never grows them.
    ///
    /// # Errors
    ///
    /// [`ModelError::CacheFull`] for a cache of fixed size without the room;
    /// those of the backend.
    pub(crate) fn reserve<B: Write + Transfer>(
        &mut self,
        backend: &B,
        batch: &[(SequenceId, usize)],
    ) -> Result<(), ModelError> {
        self.make_room(batch)?;
        let lines = self.pages * PAGE;
        for layer in 0..self.layers.len() {
            if let Some((k, v)) = &self.layers[layer] {
                let (k, v) = (k.clone(), v.clone());
                // The clones go before the tables are replaced.
                let like = (k.narrow(0, 0, 1)?, v.narrow(0, 0, 1)?);
                drop((k, v));
                self.tables(backend, layer, lines, &like.0, &like.1)?;
            }
        }
        Ok(())
    }

    /// Free pages for `batch`, the cache grown (in pages) if it may.
    fn make_room(&mut self, batch: &[(SequenceId, usize)]) -> Result<(), ModelError> {
        let needed: usize = batch
            .iter()
            .map(|&(id, count)| self.pages_needed(id, count))
            .sum();
        if needed > self.free.len() {
            if self.fixed {
                return Err(ModelError::CacheFull {
                    needed,
                    free: self.free.len(),
                });
            }
            let pages = (self.pages * 2)
                .max(self.pages + needed - self.free.len())
                .max(4);
            self.free
                .splice(0..0, (self.pages as u32..pages as u32).rev());
            self.pages = pages;
        }
        Ok(())
    }

    /// Plans a pass of the model over `batch`: for each sequence, the number
    /// of new tokens. Their pages are taken, their positions counted. The
    /// table of pages has `widest` pages per sequence, or as many as the
    /// widest sequence needs.
    ///
    /// The new tokens of all the sequences form one list of `R` rows, in the
    /// order of the batch.
    pub(crate) fn plan(
        &mut self,
        batch: &[(SequenceId, usize)],
        widest: Option<usize>,
    ) -> Result<Plan, ModelError> {
        for (i, &(id, _)) in batch.iter().enumerate() {
            if batch[..i].iter().any(|&(other, _)| other == id) {
                return Err(ModelError::Tensor(kapel_core::Error::InvalidArgument {
                    op: "step",
                    message: "a sequence appears twice in one pass".into(),
                }));
            }
        }
        let widest = widest
            .unwrap_or_else(|| self.widest(batch))
            .max(self.widest(batch));
        self.make_room(batch)?;
        for &(id, count) in batch {
            let take = self.pages_needed(id, count);
            let at = self.free.len() - take;
            let pages: Vec<u32> = self.free.drain(at..).rev().collect();
            self.sequences[id.0]
                .as_mut()
                .expect("a sequence in progress")
                .pages
                .extend(pages);
        }
        let (mut positions, mut lines, mut rows) = (Vec::new(), Vec::new(), Vec::new());
        let mut pages = vec![0u32; batch.len() * widest];
        for (s, &(id, count)) in batch.iter().enumerate() {
            let sequence = self.sequences[id.0]
                .as_mut()
                .expect("a sequence in progress");
            pages[s * widest..][..sequence.pages.len()].copy_from_slice(&sequence.pages);
            for j in sequence.len..sequence.len + count {
                let line = sequence.pages[j / PAGE] as usize * PAGE + j % PAGE;
                positions.push(
                    u32::try_from(j).map_err(|_| kapel_core::Error::Overflow { op: "step" })?,
                );
                lines.push(line as u32);
                rows.extend([s as u32, j as u32 + 1]);
            }
            sequence.len += count;
        }
        Ok(Plan {
            positions,
            lines,
            pages,
            widest,
            rows,
        })
    }

    /// Prepares a pass of the model over `batch` ([`KvCache::plan`]), and
    /// uploads what the attention needs to `backend`.
    pub(crate) fn begin<B: Transfer>(
        &mut self,
        backend: &B,
        batch: &[(SequenceId, usize)],
    ) -> Result<Step, ModelError> {
        let plan = self.plan(batch, None)?;
        let r = plan.positions.len();
        let upload = |values: &[u32], shape: &[usize]| -> Result<Tensor, ModelError> {
            Ok(backend.upload(&Tensor::from_slice(values, shape)?)?)
        };
        Ok(Step {
            positions: upload(&plan.positions, &[r])?,
            lines: upload(&plan.lines, &[r])?,
            pages: upload(&plan.pages, &[batch.len(), plan.widest])?,
            rows: upload(&plan.rows, &[r, 2])?,
            lines_needed: self.pages * PAGE,
        })
    }

    /// The lines the tables must have for the pages of the cache.
    pub(crate) fn lines(&self) -> usize {
        self.pages * PAGE
    }

    /// The tables of `layer`, made or grown on `backend` to `lines` lines, for
    /// keys like `k [R, kv_heads, head_dim]` and values like `v`.
    pub(crate) fn tables<B: Write + Transfer>(
        &mut self,
        backend: &B,
        layer: usize,
        lines: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<&mut (Tensor, Tensor), ModelError> {
        if self.layers.len() <= layer {
            self.layers.resize_with(layer + 1, || None);
        }
        let dtype = self.dtype.unwrap_or(k.dtype());
        let slot = &mut self.layers[layer];
        let make = |like: &Tensor| backend.zeros(&[lines, like.shape()[1], like.shape()[2]], dtype);
        if slot
            .as_ref()
            .is_none_or(|(keys, _)| keys.shape()[0] < lines)
        {
            self.identity.1 += 1;
        }
        match slot {
            None => *slot = Some((make(k)?, make(v)?)),
            Some((keys, _)) if keys.shape()[0] < lines => {
                // Twice as large, or more: the old lines keep their numbers.
                let (old_k, old_v) = slot.take().expect("matched above");
                let old = old_k.shape()[0];
                let index = backend.upload(&Tensor::from_slice(
                    &(0..old as u32).collect::<Vec<_>>(),
                    &[old],
                )?)?;
                let (mut new_k, mut new_v) = (make(k)?, make(v)?);
                backend.write_rows(&mut new_k, &index, &old_k)?;
                backend.write_rows(&mut new_v, &index, &old_v)?;
                *slot = Some((new_k, new_v));
            }
            Some(_) => {}
        }
        Ok(slot.as_mut().expect("made above"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(cache: &KvCache, id: SequenceId) -> Vec<usize> {
        let sequence = cache.sequence(id);
        (0..sequence.len)
            .map(|j| sequence.pages[j / PAGE] as usize * PAGE + j % PAGE)
            .collect()
    }

    fn grow(cache: &mut KvCache, id: SequenceId, count: usize) -> Result<Step, ModelError> {
        cache.begin(&kapel_ref::Reference, &[(id, count)])
    }

    #[test]
    fn sequences_take_whole_pages_and_give_them_back() {
        let mut cache = KvCache::with_pages(4);
        let (a, b) = (cache.add_sequence(), cache.add_sequence());
        grow(&mut cache, a, 2).unwrap();
        grow(&mut cache, b, 17).unwrap();
        assert_eq!(cache.free_pages(), 1);
        assert_eq!(lines(&cache, a), [0, 1]);
        assert_eq!(lines(&cache, b)[..2], [16, 17]);
        assert_eq!(lines(&cache, b)[16], 32);
        assert_eq!(cache.pages_needed(a, 14), 0);
        assert_eq!(cache.pages_needed(a, 15), 1);
        cache.remove_sequence(a);
        assert_eq!(cache.free_pages(), 2);
        let c = cache.add_sequence();
        assert_eq!(c, a, "the number of an ended sequence is used again");
        grow(&mut cache, c, 1).unwrap();
        assert_eq!(lines(&cache, c), [0]);
        cache.truncate(b, 16);
        assert_eq!(cache.free_pages(), 2);
        grow(&mut cache, c, 40).unwrap();
        assert_eq!(cache.free_pages(), 0);
    }

    #[test]
    fn a_fixed_cache_refuses_positions_beyond_its_pages() {
        let mut cache = KvCache::with_pages(2);
        let id = cache.add_sequence();
        assert!(matches!(
            grow(&mut cache, id, 33),
            Err(ModelError::CacheFull { needed: 3, free: 2 })
        ));
    }

    #[test]
    fn a_growing_cache_doubles_and_keeps_its_pages() {
        let mut cache = KvCache::new();
        let id = cache.add_sequence();
        grow(&mut cache, id, 3).unwrap();
        assert_eq!(cache.pages(), 4);
        let before = lines(&cache, id);
        grow(&mut cache, id, 4 * PAGE).unwrap();
        assert_eq!(cache.pages(), 8);
        assert_eq!(lines(&cache, id)[..3], before[..]);
    }

    #[test]
    fn a_pass_describes_its_tokens() {
        let mut cache = KvCache::with_pages(4);
        let (a, b) = (cache.add_sequence(), cache.add_sequence());
        grow(&mut cache, a, 3).unwrap();
        let step = cache
            .begin(&kapel_ref::Reference, &[(b, 2), (a, 1)])
            .unwrap();
        assert_eq!(step.positions.to_vec::<u32>().unwrap(), [0, 1, 3]);
        assert_eq!(step.lines.to_vec::<u32>().unwrap(), [16, 17, 3]);
        assert_eq!(step.pages.to_vec::<u32>().unwrap(), [1, 0]);
        assert_eq!(step.rows.to_vec::<u32>().unwrap(), [0, 1, 0, 2, 1, 4]);
        assert!(
            cache
                .begin(&kapel_ref::Reference, &[(a, 1), (a, 1)])
                .is_err()
        );
    }
}
