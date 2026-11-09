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
//!
//! With [`KvCache::with_prefix_sharing`], sequences share the pages of their
//! common beginning, as the requests of a server share a system prompt. Each
//! full page is indexed by the tokens it holds and those before it; a
//! sequence started by [`KvCache::add_sequence_sharing`] takes, rather than
//! computes, the pages of the longest beginning of its prompt the cache
//! knows. A page counts the sequences that use it; once none does, it stays
//! indexed, free to be taken back, until its room is needed (the page served
//! least recently goes first). Pages are written only past the end of their
//! sequence: a shared page is full, and never written again.
//!
//! The layers of linear attention (Gated DeltaNet) keep no keys nor values but
//! states, a fixed size per sequence: the last inputs of their convolution and
//! the matrices of their delta rule ([`kapel_ops::Recurrent`]). Each sequence
//! has its slot in these tables, the first slot left to the padding rows of
//! recorded passes. A state holds every position at once: it cannot be shared
//! between sequences nor brought back to an earlier length, so a cache that
//! holds states shares no pages and refuses to go on with a sequence it has
//! [truncated](KvCache::truncate).

use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use kapel_core::{DType, Tensor};
use kapel_ops::{Backend, Contiguous, Transfer, Write};

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
    /// The tokens of its positions, with prefix sharing.
    tokens: Vec<u32>,
    /// Its full pages that are indexed or known not to be: those before
    /// this one have been offered to the index.
    offered: usize,
    /// The hash of the tokens of its first `offered` pages ([`chain_of`]).
    chain: u64,
    /// Whether it was brought back to an earlier length while the cache held
    /// states, which cannot follow it.
    stale: bool,
}

/// The hash of `tokens`, whole pages, as the index chains it: `0` for none.
fn chain_of(tokens: &[u32]) -> u64 {
    tokens
        .as_chunks::<PAGE>()
        .0
        .iter()
        .fold(0, |parent, page| PageKey::new(parent, page).chain)
}

/// What the index knows of a full page: the tokens before it, by their
/// hash, and its own.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PageKey {
    /// The hash of every token before the page's own and of them.
    chain: u64,
    /// The hash of the tokens before the page (`0` for a first page).
    parent: u64,
    tokens: [u32; PAGE],
}

impl PageKey {
    fn new(parent: u64, tokens: &[u32]) -> Self {
        let tokens: [u32; PAGE] = tokens.try_into().expect("a full page");
        let mut hasher = DefaultHasher::new();
        parent.hash(&mut hasher);
        tokens.hash(&mut hasher);
        Self {
            // Never 0, which means "no page before".
            chain: hasher.finish() | 1,
            parent,
            tokens,
        }
    }
}

/// The pages that sequences can share, by the tokens they hold.
#[derive(Debug, Default)]
struct Prefixes {
    /// The indexed page of each chain.
    by_chain: HashMap<u64, u32>,
    /// Per page, its key if it is indexed.
    keys: Vec<Option<PageKey>>,
    /// The indexed pages no sequence uses, by the time they were let go:
    /// the first is the next taken back.
    idle: BTreeMap<u64, u32>,
    /// Per page, its key in `idle`.
    idle_since: Vec<Option<u64>>,
    clock: u64,
    /// Positions taken from the index rather than computed.
    hits: u64,
}

/// A copy of the first lines of one page into another, waiting for a
/// backend: a sequence cut inside a page it shared gets its own copy.
#[derive(Clone, Copy, Debug)]
struct PageCopy {
    from: u32,
    to: u32,
    lines: usize,
}

/// The keys and values of every sequence in progress, by pages.
#[derive(Debug)]
pub struct KvCache {
    /// Per layer, the keys and values `[lines, kv_heads, head_dim]`, made at
    /// the first write, when their widths are known.
    layers: Vec<Option<(Tensor, Tensor)>>,
    /// Per layer of linear attention, the states of its convolution `[slots,
    /// taps − 1, channels]` and of its delta rule `[slots, heads, key width,
    /// value width]` (`f32`), slot `1 + i` for sequence `i`.
    states: Vec<Option<(Tensor, Tensor)>>,
    /// The pages of the tables: they have `pages · PAGE` lines.
    pages: usize,
    /// The free pages, not indexed; the next one given is the last.
    free: Vec<u32>,
    /// Per page, the sequences that use it.
    refs: Vec<u32>,
    /// The index of shared pages, with prefix sharing.
    prefixes: Option<Prefixes>,
    /// The pages the padding rows of recorded passes write into, taken as
    /// they are needed and kept.
    scratch: Vec<u32>,
    /// Copies to make before the next pass.
    copies: Vec<PageCopy>,
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
            states: Vec::new(),
            pages: 0,
            free: Vec::new(),
            refs: Vec::new(),
            prefixes: None,
            scratch: Vec::new(),
            copies: Vec::new(),
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
    /// `[R]`: the slot of each row's sequence in the tables of states.
    pub(crate) slots: Vec<u32>,
    /// The slots of the sequences the pass starts: their states are zeroed
    /// first.
    pub(crate) fresh: Vec<u32>,
}

impl Plan {
    /// Adds rows up to `rows`, each one token that writes into a line of
    /// `scratch` of its own and sees it alone: rows that compute nothing
    /// anyone reads, so that a recorded pass serves smaller batches.
    pub(crate) fn pad(&mut self, rows: usize, scratch: &[u32]) {
        let real = self.positions.len();
        for r in real..rows {
            let line = scratch[r - real];
            self.positions.push(0);
            self.lines.push(line);
            let at = self.pages.len();
            self.pages.resize(at + self.widest, 0);
            self.pages[at] = line / PAGE as u32;
            // The line is the first position its row sees: the row's
            // positions start at the line's place in its page.
            self.rows.extend([r as u32, line % PAGE as u32 + 1]);
            // The padding's own slot, which no sequence has.
            self.slots.push(0);
        }
    }
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
    /// `[R]`: the slot of each new token's sequence in the tables of states.
    pub(crate) slots: Tensor,
    /// The slots whose states the pass zeroes first, on the host.
    pub(crate) fresh: Vec<u32>,
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
            refs: vec![0; pages],
            fixed: true,
            ..Self::default()
        }
    }

    /// The same cache, its sequences sharing the pages of their common
    /// beginnings (see the module's documentation). Set it before the first
    /// sequence.
    #[must_use]
    pub fn with_prefix_sharing(mut self) -> Self {
        self.prefixes = Some(Prefixes {
            keys: vec![None; self.pages],
            idle_since: vec![None; self.pages],
            ..Prefixes::default()
        });
        self
    }

    /// Whether the cache keeps its number of pages.
    pub fn is_fixed(&self) -> bool {
        self.fixed
    }

    /// Starts a new sequence that will continue `prompt`, its first positions
    /// taken from the pages of the longest beginning of `prompt` the cache
    /// holds, whole pages short of its last token (which is left to compute,
    /// for its logits). The sequence and the positions it holds already:
    /// what is left of `prompt` to compute. Without prefix sharing, a new
    /// empty sequence.
    pub fn add_sequence_sharing(&mut self, prompt: &[u32]) -> (SequenceId, usize) {
        let id = self.add_sequence();
        // A state holds a whole sequence: nothing of it can be shared.
        let holds_states = self.states.iter().any(Option::is_some);
        let Some(prefixes) = self.prefixes.as_mut().filter(|_| !holds_states) else {
            return (id, 0);
        };
        let mut parent = 0;
        let mut pages = Vec::new();
        for chunk in prompt[..prompt.len().saturating_sub(1)]
            .as_chunks::<PAGE>()
            .0
        {
            let key = PageKey::new(parent, chunk);
            let Some(&page) = prefixes.by_chain.get(&key.chain) else {
                break;
            };
            if prefixes.keys[page as usize].as_ref() != Some(&key) {
                break;
            }
            pages.push(page);
            parent = key.chain;
        }
        let len = pages.len() * PAGE;
        for &page in &pages {
            if self.refs[page as usize] == 0
                && let Some(since) = prefixes.idle_since[page as usize].take()
            {
                prefixes.idle.remove(&since);
            }
            self.refs[page as usize] += 1;
        }
        prefixes.hits += len as u64;
        let sequence = self.sequences[id.0].as_mut().expect("just added");
        sequence.offered = pages.len();
        sequence.chain = parent;
        sequence.pages = pages;
        sequence.len = len;
        sequence.tokens = prompt[..len].to_vec();
        (id, len)
    }

    /// The positions taken from shared pages rather than computed, since the
    /// cache was made.
    pub fn shared_positions(&self) -> u64 {
        self.prefixes.as_ref().map_or(0, |p| p.hits)
    }

    /// Forgets every shared page no sequence uses: after a pass that failed,
    /// what they hold may be wrong.
    pub fn clear_prefixes(&mut self) {
        let Some(prefixes) = self.prefixes.as_mut() else {
            return;
        };
        for (_, page) in std::mem::take(&mut prefixes.idle) {
            prefixes.idle_since[page as usize] = None;
            if let Some(key) = prefixes.keys[page as usize].take() {
                prefixes.by_chain.remove(&key.chain);
            }
            self.free.push(page);
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
        // The last pages first: they are taken back before the pages they
        // follow, which other sequences are likelier to share.
        for &page in sequence.pages.iter().rev() {
            self.release(page);
        }
    }

    /// One sequence fewer uses `page`: once none does, it is free, or idle
    /// if indexed.
    fn release(&mut self, page: u32) {
        let refs = &mut self.refs[page as usize];
        *refs -= 1;
        if *refs > 0 {
            return;
        }
        match self.prefixes.as_mut() {
            Some(p) if p.keys[page as usize].is_some() => {
                p.clock += 1;
                p.idle.insert(p.clock, page);
                p.idle_since[page as usize] = Some(p.clock);
            }
            _ => self.free.push(page),
        }
    }

    /// A page for a sequence: a free one, or else the idle page let go
    /// first, taken out of the index.
    fn take_page(&mut self) -> u32 {
        let page = match self.free.pop() {
            Some(page) => page,
            None => {
                let p = self.prefixes.as_mut().expect("pages were counted");
                let (_, page) = p.idle.pop_first().expect("pages were counted");
                p.idle_since[page as usize] = None;
                let key = p.keys[page as usize].take().expect("an indexed page");
                p.by_chain.remove(&key.chain);
                page
            }
        };
        self.refs[page as usize] = 1;
        page
    }

    /// The pages free or idle.
    fn available(&self) -> usize {
        self.free.len() + self.prefixes.as_ref().map_or(0, |p| p.idle.len())
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

    /// The pages no sequence uses (those that keep a shared beginning
    /// included: they are taken back when needed).
    pub fn free_pages(&self) -> usize {
        self.available()
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
        if len < sequence.len && self.states.iter().any(Option::is_some) {
            sequence.stale = true;
        }
        sequence.len = len;
        sequence.tokens.truncate(len);
        if sequence.offered > len / PAGE {
            sequence.offered = len / PAGE;
            sequence.chain = chain_of(&sequence.tokens[..sequence.offered * PAGE]);
        }
        let freed: Vec<u32> = sequence.pages.drain(len.div_ceil(PAGE)..).collect();
        let last = (!len.is_multiple_of(PAGE)).then(|| sequence.pages[len / PAGE]);
        for &page in freed.iter().rev() {
            self.release(page);
        }
        // The page now cut will be written again: no other sequence may see
        // it change, nor the index keep it.
        let Some(page) = last else { return };
        if self.refs[page as usize] > 1 {
            self.refs[page as usize] -= 1;
            let copy = self.take_page();
            self.copies.push(PageCopy {
                from: page,
                to: copy,
                lines: len % PAGE,
            });
            let sequence = self.sequences[id.0].as_mut().expect("in progress");
            sequence.pages[len / PAGE] = copy;
        } else if let Some(p) = self.prefixes.as_mut()
            && let Some(key) = p.keys[page as usize].take()
        {
            p.by_chain.remove(&key.chain);
        }
    }

    /// Makes the copies of pages that cut sequences wait for.
    fn copy_pages<B: Backend + Transfer>(&mut self, backend: &B) -> Result<(), ModelError> {
        for copy in std::mem::take(&mut self.copies) {
            let to: Vec<u32> = (0..copy.lines)
                .map(|j| copy.to * PAGE as u32 + j as u32)
                .collect();
            let index = backend.upload(&Tensor::from_slice(&to, &[to.len()])?)?;
            for (k, v) in self.layers.iter_mut().flatten() {
                for table in [k, v] {
                    let lines = table.narrow(0, copy.from as usize * PAGE, copy.lines)?;
                    let rows = backend.forward(&Contiguous, &[&lines])?;
                    drop(lines);
                    backend.write_rows(table, &index, &rows)?;
                }
            }
        }
        Ok(())
    }

    /// Takes now, from the reserve, the pages the padding of recorded passes
    /// of up to `rows` rows will write into, so that recording one never
    /// takes pages a pass was counting on. They are never given back.
    ///
    /// # Errors
    ///
    /// [`ModelError::CacheFull`] for a cache of fixed size without them.
    pub fn reserve_padding(&mut self, rows: usize) -> Result<(), ModelError> {
        self.scratch_lines(rows.saturating_sub(1)).map(drop)
    }

    /// `count` lines for the padding rows of recorded passes to write into
    /// and read, one each: lines of pages taken from the reserve the first
    /// time they are needed, and never given back.
    pub(crate) fn scratch_lines(&mut self, count: usize) -> Result<Vec<u32>, ModelError> {
        let pages = count.div_ceil(PAGE);
        let missing = pages.saturating_sub(self.scratch.len());
        if self.fixed && missing > self.available() {
            // All or nothing: a pass may count on the pages left.
            return Err(ModelError::CacheFull {
                needed: missing,
                free: self.available(),
            });
        }
        while self.scratch.len() < pages {
            if self.available() == 0 {
                self.grow_to(self.pages * 2 + 1);
            }
            let page = self.take_page();
            self.scratch.push(page);
        }
        Ok((0..count)
            .map(|i| self.scratch[i / PAGE] * PAGE as u32 + (i % PAGE) as u32)
            .collect())
    }

    /// The bytes the tables hold, or will hold once a pass has made them:
    /// what each position costs times the lines.
    pub fn table_bytes(&self) -> usize {
        self.layers
            .iter()
            .chain(&self.states)
            .flatten()
            .map(|(k, v)| k.storage().len() + v.storage().len())
            .sum()
    }

    /// Whether the first `layers` layers have their tables, or their states.
    pub(crate) fn has_tables(&self, layers: usize) -> bool {
        (0..layers).all(|l| {
            self.layers.get(l).is_some_and(Option::is_some)
                || self.states.get(l).is_some_and(Option::is_some)
        })
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
            && needed <= self.available()
            && self.copies.is_empty()
    }

    /// Makes room for `batch` (each sequence and the positions it will add):
    /// free pages for all of them, the cache grown if it may, and every table
    /// made so far grown to the pages. A recorded pass then never grows them.
    ///
    /// # Errors
    ///
    /// [`ModelError::CacheFull`] for a cache of fixed size without the room;
    /// those of the backend.
    pub(crate) fn reserve<B: Backend + Transfer>(
        &mut self,
        backend: &B,
        batch: &[(SequenceId, usize)],
    ) -> Result<(), ModelError> {
        self.make_room(batch)?;
        self.copy_pages(backend)?;
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
        for layer in 0..self.states.len() {
            if let Some((conv, delta)) = &self.states[layer] {
                let shapes = (
                    conv.shape()[1..].to_vec(),
                    conv.dtype(),
                    delta.shape()[1..].to_vec(),
                );
                self.states(backend, layer, (&shapes.0, shapes.1), &shapes.2, &[])?;
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
        let available = self.available();
        if needed > available {
            if self.fixed {
                return Err(ModelError::CacheFull {
                    needed,
                    free: available,
                });
            }
            self.grow_to((self.pages * 2).max(self.pages + needed - available).max(4));
        }
        Ok(())
    }

    /// Grows the cache to `pages` pages; the new ones are given after the
    /// free ones.
    fn grow_to(&mut self, pages: usize) {
        self.free
            .splice(0..0, (self.pages as u32..pages as u32).rev());
        self.refs.resize(pages, 0);
        if let Some(p) = self.prefixes.as_mut() {
            p.keys.resize(pages, None);
            p.idle_since.resize(pages, None);
        }
        self.pages = pages;
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
        batch: &[(SequenceId, &[u32])],
        widest: Option<usize>,
    ) -> Result<Plan, ModelError> {
        let counts: Vec<(SequenceId, usize)> =
            batch.iter().map(|&(id, ids)| (id, ids.len())).collect();
        let plan = self.plan_counts(&counts, widest)?;
        if self.prefixes.is_some() {
            for &(id, ids) in batch {
                self.offer(id, ids);
            }
        }
        Ok(plan)
    }

    /// Notes the new tokens of a sequence, and indexes the pages they fill.
    fn offer(&mut self, id: SequenceId, ids: &[u32]) {
        let sequence = self.sequences[id.0].as_mut().expect("in progress");
        sequence.tokens.extend_from_slice(ids);
        let prefixes = self.prefixes.as_mut().expect("prefix sharing");
        let full = sequence.len / PAGE;
        while sequence.offered < full {
            let at = sequence.offered;
            let key = PageKey::new(sequence.chain, &sequence.tokens[at * PAGE..][..PAGE]);
            sequence.chain = key.chain;
            let page = sequence.pages[at];
            if !prefixes.by_chain.contains_key(&key.chain) && prefixes.keys[page as usize].is_none()
            {
                prefixes.by_chain.insert(key.chain, page);
                prefixes.keys[page as usize] = Some(key);
            }
            sequence.offered += 1;
        }
    }

    fn plan_counts(
        &mut self,
        batch: &[(SequenceId, usize)],
        widest: Option<usize>,
    ) -> Result<Plan, ModelError> {
        for (i, &(id, _)) in batch.iter().enumerate() {
            if self.sequence(id).stale {
                return Err(ModelError::Tensor(kapel_core::Error::InvalidArgument {
                    op: "step",
                    message: "a sequence of a model with linear attention was shortened: \
                              its states cannot follow"
                        .into(),
                }));
            }
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
            let pages: Vec<u32> = (0..take).map(|_| self.take_page()).collect();
            self.sequences[id.0]
                .as_mut()
                .expect("a sequence in progress")
                .pages
                .extend(pages);
        }
        let (mut positions, mut lines, mut rows) = (Vec::new(), Vec::new(), Vec::new());
        let (mut slots, mut fresh) = (Vec::new(), Vec::new());
        let mut pages = vec![0u32; batch.len() * widest];
        for (s, &(id, count)) in batch.iter().enumerate() {
            let sequence = self.sequences[id.0]
                .as_mut()
                .expect("a sequence in progress");
            let slot =
                u32::try_from(id.0 + 1).map_err(|_| kapel_core::Error::Overflow { op: "step" })?;
            if sequence.len == 0 {
                fresh.push(slot);
            }
            slots.extend(std::iter::repeat_n(slot, count));
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
            slots,
            fresh,
        })
    }

    /// Prepares a pass of the model over `batch` ([`KvCache::plan`]), and
    /// uploads what the attention needs to `backend`.
    pub(crate) fn begin<B: Backend + Transfer>(
        &mut self,
        backend: &B,
        batch: &[(SequenceId, &[u32])],
    ) -> Result<Step, ModelError> {
        self.copy_pages(backend)?;
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
            slots: upload(&plan.slots, &[r])?,
            fresh: plan.fresh,
            lines_needed: self.pages * PAGE,
        })
    }

    /// The slots the tables of states must have: one per sequence ever
    /// started, and the padding's.
    fn slots(&self) -> usize {
        self.sequences.len() + 1
    }

    /// The states of `layer`, made or grown on `backend` for every sequence:
    /// a convolution's `[slots, conv...]` of type `conv.1`, a delta rule's
    /// `[slots, delta...]` in `f32`; the slots of `fresh` zeroed.
    pub(crate) fn states<B: Write + Transfer>(
        &mut self,
        backend: &B,
        layer: usize,
        conv: (&[usize], DType),
        delta: &[usize],
        fresh: &[u32],
    ) -> Result<&mut (Tensor, Tensor), ModelError> {
        if self.states.len() <= layer {
            self.states.resize_with(layer + 1, || None);
        }
        let needed = self.slots();
        let shaped = |slots: usize, rest: &[usize]| {
            let mut shape = vec![slots];
            shape.extend_from_slice(rest);
            shape
        };
        let made = self.states[layer].as_ref().map(|(c, _)| c.shape()[0]);
        if made.is_none_or(|slots| slots < needed) {
            self.identity.1 += 1;
            // Room for twice the sequences, or more: few regrowths.
            let slots = needed.max(made.unwrap_or(0) * 2).max(4);
            let mut c = backend.zeros(&shaped(slots, conv.0), conv.1)?;
            let mut d = backend.zeros(&shaped(slots, delta), DType::F32)?;
            if let Some((old_c, old_d)) = self.states[layer].take() {
                let old = old_c.shape()[0];
                let index = backend.upload(&Tensor::from_slice(
                    &(0..old as u32).collect::<Vec<_>>(),
                    &[old],
                )?)?;
                backend.write_rows(&mut c, &index, &old_c)?;
                backend.write_rows(&mut d, &index, &old_d)?;
            }
            self.states[layer] = Some((c, d));
        }
        let (c, d) = self.states[layer].as_mut().expect("made above");
        if !fresh.is_empty() {
            let index = backend.upload(&Tensor::from_slice(fresh, &[fresh.len()])?)?;
            let zeros_c = backend.zeros(&shaped(fresh.len(), conv.0), c.dtype())?;
            let zeros_d = backend.zeros(&shaped(fresh.len(), delta), DType::F32)?;
            backend.write_rows(c, &index, &zeros_c)?;
            backend.write_rows(d, &index, &zeros_d)?;
        }
        Ok(self.states[layer].as_mut().expect("made above"))
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
        cache.begin(&kapel_ref::Reference, &[(id, &vec![0; count])])
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
            .begin(&kapel_ref::Reference, &[(b, &[1, 2]), (a, &[3])])
            .unwrap();
        assert_eq!(step.positions.to_vec::<u32>().unwrap(), [0, 1, 3]);
        assert_eq!(step.lines.to_vec::<u32>().unwrap(), [16, 17, 3]);
        assert_eq!(step.pages.to_vec::<u32>().unwrap(), [1, 0]);
        assert_eq!(step.rows.to_vec::<u32>().unwrap(), [0, 1, 0, 2, 1, 4]);
        assert!(
            cache
                .begin(&kapel_ref::Reference, &[(a, &[1]), (a, &[2])])
                .is_err()
        );
    }

    fn write(cache: &mut KvCache, id: SequenceId, tokens: &[u32]) {
        cache.begin(&kapel_ref::Reference, &[(id, tokens)]).unwrap();
    }

    #[test]
    fn sequences_share_the_pages_of_a_common_beginning() {
        let mut cache = KvCache::with_pages(8).with_prefix_sharing();
        let prompt: Vec<u32> = (0..40).collect();
        let (a, held) = cache.add_sequence_sharing(&prompt);
        assert_eq!(held, 0);
        write(&mut cache, a, &prompt);
        assert_eq!(cache.free_pages(), 5);
        // Two full pages in common, then another token.
        let mut other = prompt[..32].to_vec();
        other.push(99);
        let (b, held) = cache.add_sequence_sharing(&other);
        assert_eq!(held, 32);
        assert_eq!(lines(&cache, b), lines(&cache, a)[..32]);
        assert_eq!(cache.free_pages(), 5, "nothing taken for shared pages");
        write(&mut cache, b, &[99]);
        assert_eq!(cache.free_pages(), 4);
        // A prompt of exactly two pages keeps its last token to compute.
        let (c, held) = cache.add_sequence_sharing(&prompt[..32]);
        assert_eq!(held, 16);
        cache.remove_sequence(c);
        assert_eq!(cache.shared_positions(), 48);
        // Once every sequence has ended, the pages stay, idle but known.
        cache.remove_sequence(a);
        cache.remove_sequence(b);
        assert_eq!(cache.free_pages(), 8);
        let (d, held) = cache.add_sequence_sharing(&prompt);
        assert_eq!(held, 32);
        cache.remove_sequence(d);
    }

    #[test]
    fn idle_pages_are_taken_back_last_page_first() {
        let mut cache = KvCache::with_pages(4).with_prefix_sharing();
        let prompt: Vec<u32> = (0..48).collect();
        let (a, _) = cache.add_sequence_sharing(&prompt);
        write(&mut cache, a, &prompt);
        cache.remove_sequence(a);
        // Three pages: the free one, then the idle ones, the last page of
        // the prompt first; its first page stays.
        let (b, _) = cache.add_sequence_sharing(&[7]);
        write(&mut cache, b, &[7; 48]);
        let (c, held) = cache.add_sequence_sharing(&prompt);
        assert_eq!(held, 16);
        cache.remove_sequence(c);
    }

    #[test]
    fn a_sequence_cut_in_a_shared_page_writes_into_its_own_copy() {
        let mut cache = KvCache::with_pages(4).with_prefix_sharing();
        let prompt: Vec<u32> = (0..33).collect();
        let (a, _) = cache.add_sequence_sharing(&prompt);
        write(&mut cache, a, &prompt);
        let (b, held) = cache.add_sequence_sharing(&prompt);
        assert_eq!(held, 32);
        let shared = lines(&cache, a)[16];
        cache.truncate(b, 20);
        assert_ne!(lines(&cache, b)[16], shared, "a page of its own");
        assert_eq!(lines(&cache, b)[..16], lines(&cache, a)[..16]);
        assert_eq!(cache.copies.len(), 1);
        assert!(!cache.fits(&[(b, 1)], 4), "the copy comes first");
        write(&mut cache, b, &[5]);
        assert!(cache.copies.is_empty());
        // The first sequence still finds its pages.
        cache.remove_sequence(b);
        let (c, held) = cache.add_sequence_sharing(&prompt);
        assert_eq!(held, 32);
        assert_eq!(lines(&cache, c)[16], shared);
    }
}
