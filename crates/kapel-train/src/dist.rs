//! Collective communications between the ranks of a run.
//!
//! Every collective is built on one exchange: each rank gives its bytes and
//! receives those of every rank, in the order of the ranks. A sum is then
//! computed by every rank alike, rank 0's values first, then rank 1's
//! added, and so on: the result does not depend on which rank arrives
//! first, and every rank holds the same bits. Two implementations: ranks as
//! threads of one process ([`Threads`]), and processes joined by TCP
//! ([`Tcp`]), rank 0 gathering and sending back. Neither is fast; they are
//! what the numbers of several ranks are checked with on one machine, before
//! NCCL takes their place on several cards.
//!
//! A rank that fails calls [`Collective::abort`]: the others' exchanges
//! then fail instead of waiting for it.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A collective operation that failed: another rank stopped, or the
/// connection to it broke.
#[derive(Debug)]
pub struct DistError(pub String);

impl std::fmt::Display for DistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DistError {}

/// What a collective operation returns.
pub type Result<T> = std::result::Result<T, DistError>;

/// The ranks of a run, from one of them.
pub trait Collective: Send + Sync {
    /// This rank, from 0.
    fn rank(&self) -> usize;

    /// The number of ranks.
    fn size(&self) -> usize;

    /// Gives `mine` and receives every rank's bytes, in the order of the
    /// ranks (this rank's among them). Every rank must call it, as many
    /// times.
    ///
    /// # Errors
    ///
    /// If another rank stopped or cannot be reached.
    fn exchange(&self, mine: &[u8]) -> Result<Vec<Vec<u8>>>;

    /// Gives `mine` to rank 0, which receives every rank's bytes in the
    /// order of the ranks; the others receive `None`.
    ///
    /// # Errors
    ///
    /// As [`Collective::exchange`].
    fn gather(&self, mine: &[u8]) -> Result<Option<Vec<Vec<u8>>>>;

    /// Tells the other ranks that this one stops: their exchanges fail.
    fn abort(&self);

    /// Waits for every rank.
    ///
    /// # Errors
    ///
    /// As [`Collective::exchange`].
    fn barrier(&self) -> Result<()> {
        self.exchange(&[]).map(drop)
    }

    /// The sum over the ranks of `values`, written into them on every
    /// rank: rank 0's values, then each next rank's added in order.
    ///
    /// # Errors
    ///
    /// As [`Collective::exchange`], and if the ranks give different lengths.
    fn all_reduce_sum(&self, values: &mut [f32]) -> Result<()> {
        let all = self.exchange(&f32_bytes(values))?;
        sum_into(values, &all, 0)
    }

    /// The part `rank` of the sum over the ranks of `values` (its
    /// `values.len() / size` elements, the last part shorter), each rank
    /// receiving its own: what ZeRO's reduce-scatter gives.
    ///
    /// # Errors
    ///
    /// As [`Collective::all_reduce_sum`].
    fn reduce_scatter_sum(&self, values: &[f32]) -> Result<Vec<f32>> {
        let all = self.exchange(&f32_bytes(values))?;
        let (start, end) = part(values.len(), self.size(), self.rank());
        let mut out = values[start..end].to_vec();
        sum_into(&mut out, &all, start)?;
        Ok(out)
    }

    /// Every rank's `part`, one after the other in the order of the ranks.
    ///
    /// # Errors
    ///
    /// As [`Collective::exchange`].
    fn all_gather(&self, part: &[f32]) -> Result<Vec<f32>> {
        let all = self.exchange(&f32_bytes(part))?;
        Ok(all.iter().flat_map(|b| f32s(b)).collect())
    }

    /// Every rank's `values` of `f64`, one after the other in the order of
    /// the ranks.
    ///
    /// # Errors
    ///
    /// As [`Collective::exchange`].
    fn all_gather_f64(&self, values: &[f64]) -> Result<Vec<f64>> {
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let all = self.exchange(&bytes)?;
        Ok(all
            .iter()
            .flat_map(|b| {
                b.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| f64::from_le_bytes(*c))
                    .collect::<Vec<_>>()
            })
            .collect())
    }
}

/// The part of `n` elements that rank `rank` of `size` holds:
/// `⌈n / size⌉` each, the last ones shorter or empty.
pub fn part(n: usize, size: usize, rank: usize) -> (usize, usize) {
    let each = n.div_ceil(size.max(1));
    ((rank * each).min(n), ((rank + 1) * each).min(n))
}

/// The rank that owns each parameter when the optimizer's state, the
/// gradients or the weights are shared out: the largest first, each to the
/// rank that holds the fewest elements so far (the lowest at a tie). The same
/// on every rank.
pub fn owners(
    sizes: &[(String, usize)],
    ranks: usize,
) -> std::collections::BTreeMap<String, usize> {
    let mut order: Vec<&(String, usize)> = sizes.iter().collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut load = vec![0usize; ranks.max(1)];
    let mut out = std::collections::BTreeMap::new();
    for (name, n) in order {
        let rank = (0..load.len())
            .min_by_key(|&r| (load[r], r))
            .expect("a rank");
        load[rank] += n;
        out.insert(name.clone(), rank);
    }
    out
}

/// Tensors in host memory as the bytes of a safetensors image, for an
/// exchange.
///
/// # Errors
///
/// If a tensor is not in host memory, or of a type safetensors cannot hold.
pub fn pack(tensors: &[(String, kapel_core::Tensor)]) -> Result<Vec<u8>> {
    let refs: Vec<(&str, &kapel_core::Tensor)> =
        tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    kapel_hub::safetensors::to_bytes(&refs, &Default::default())
        .map_err(|e| DistError(e.to_string()))
}

/// The tensors of [`pack`]'s bytes.
///
/// # Errors
///
/// If the bytes are not such an image.
pub fn unpack(bytes: Vec<u8>) -> Result<Vec<(String, kapel_core::Tensor)>> {
    let file = kapel_hub::safetensors::SafeTensors::from_bytes(bytes)
        .map_err(|e| DistError(e.to_string()))?;
    let names: Vec<String> = file.names().map(str::to_string).collect();
    names
        .into_iter()
        .map(|n| {
            let t = file.tensor(&n).map_err(|e| DistError(e.to_string()))?;
            Ok((n, t))
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32s(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
}

/// `out` = the sum, in the order of the ranks, of the elements
/// `from..from + out.len()` of every rank's `f32` values.
fn sum_into(out: &mut [f32], all: &[Vec<u8>], from: usize) -> Result<()> {
    let n = all.first().map_or(0, |b| b.len());
    if all.iter().any(|b| b.len() != n) {
        return Err(DistError("ranks gave values of different lengths".into()));
    }
    for (k, bytes) in all.iter().enumerate() {
        let values = f32s(&bytes[from * 4..(from + out.len()) * 4]);
        if k == 0 {
            for (o, v) in out.iter_mut().zip(values) {
                *o = v;
            }
        } else {
            for (o, v) in out.iter_mut().zip(values) {
                *o += v;
            }
        }
    }
    Ok(())
}

/// The ranks of one process, one thread each: [`Threads::group`] gives each
/// its handle.
pub struct Threads {
    rank: usize,
    shared: Arc<Shared>,
}

struct Shared {
    size: usize,
    state: Mutex<State>,
    turn: Condvar,
}

struct State {
    /// The bytes given to the exchange in progress, by rank.
    given: Vec<Option<Vec<u8>>>,
    /// The exchanges done.
    generation: u64,
    /// The last exchange's bytes.
    done: Arc<Vec<Vec<u8>>>,
    aborted: bool,
}

impl Threads {
    /// The handles of `size` ranks, one for each thread.
    pub fn group(size: usize) -> Vec<Self> {
        let shared = Arc::new(Shared {
            size,
            state: Mutex::new(State {
                given: vec![None; size],
                generation: 0,
                done: Arc::new(Vec::new()),
                aborted: false,
            }),
            turn: Condvar::new(),
        });
        (0..size)
            .map(|rank| Self {
                rank,
                shared: Arc::clone(&shared),
            })
            .collect()
    }
}

fn stopped() -> DistError {
    DistError("another rank stopped".into())
}

impl Collective for Threads {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.shared.size
    }

    fn exchange(&self, mine: &[u8]) -> Result<Vec<Vec<u8>>> {
        self.meet(mine, true).map(|all| all.unwrap_or_default())
    }

    fn gather(&self, mine: &[u8]) -> Result<Option<Vec<Vec<u8>>>> {
        self.meet(mine, self.rank == 0)
    }

    fn abort(&self) {
        if let Ok(mut state) = self.shared.state.lock() {
            state.aborted = true;
        }
        self.shared.turn.notify_all();
    }
}

impl Threads {
    /// Gives `mine` and waits for every rank's; returns them all if
    /// `receive` (the others copy nothing).
    fn meet(&self, mine: &[u8], receive: bool) -> Result<Option<Vec<Vec<u8>>>> {
        let mut state = self.shared.state.lock().map_err(|_| stopped())?;
        if state.aborted {
            return Err(stopped());
        }
        state.given[self.rank] = Some(mine.to_vec());
        if state.given.iter().all(Option::is_some) {
            let all: Vec<Vec<u8>> = state
                .given
                .iter_mut()
                .map(|g| g.take().unwrap_or_default())
                .collect();
            state.done = Arc::new(all);
            state.generation += 1;
            self.shared.turn.notify_all();
            return Ok(receive.then(|| state.done.as_ref().clone()));
        }
        let generation = state.generation;
        while state.generation == generation && !state.aborted {
            state = self.shared.turn.wait(state).map_err(|_| stopped())?;
        }
        if state.generation == generation {
            return Err(stopped());
        }
        Ok(receive.then(|| state.done.as_ref().clone()))
    }
}

/// The ranks of separate processes, joined by TCP: rank 0 listens at the
/// address of the run, the others connect to it.
pub struct Tcp {
    rank: usize,
    size: usize,
    /// Rank 0: a connection to each other rank, by rank (its own `None`).
    /// The others: their connection to rank 0.
    peers: Mutex<Vec<Option<TcpStream>>>,
}

/// How long a rank waits for the others to connect.
const CONNECT: Duration = Duration::from_secs(300);

impl Tcp {
    /// Rank `rank` of `size`, rank 0 at `address` (`host:port`).
    ///
    /// # Errors
    ///
    /// If rank 0 cannot listen at `address`, or the others cannot reach it
    /// within five minutes.
    pub fn connect(address: &str, rank: usize, size: usize) -> Result<Self> {
        let failed = |e: std::io::Error| DistError(format!("{address}: {e}"));
        let mut peers: Vec<Option<TcpStream>> = (0..size).map(|_| None).collect();
        if rank == 0 {
            let listener = TcpListener::bind(address).map_err(failed)?;
            for _ in 1..size {
                let (mut stream, _) = listener.accept().map_err(failed)?;
                stream.set_nodelay(true).map_err(failed)?;
                let mut id = [0u8; 8];
                stream.read_exact(&mut id).map_err(failed)?;
                let r = u64::from_le_bytes(id) as usize;
                if r == 0 || r >= size || peers[r].is_some() {
                    return Err(DistError(format!("{address}: a rank {r} unexpected")));
                }
                peers[r] = Some(stream);
            }
        } else {
            let start = std::time::Instant::now();
            let mut stream = loop {
                match TcpStream::connect(address) {
                    Ok(s) => break s,
                    Err(e) if start.elapsed() > CONNECT => return Err(failed(e)),
                    Err(_) => std::thread::sleep(Duration::from_millis(100)),
                }
            };
            stream.set_nodelay(true).map_err(failed)?;
            stream
                .write_all(&(rank as u64).to_le_bytes())
                .map_err(failed)?;
            peers = vec![Some(stream)];
        }
        Ok(Self {
            rank,
            size,
            peers: Mutex::new(peers),
        })
    }
}

fn send(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    stream.write_all(&(bytes.len() as u64).to_le_bytes())?;
    stream.write_all(bytes)
}

fn receive(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len = [0u8; 8];
    stream.read_exact(&mut len)?;
    let mut bytes = vec![0u8; u64::from_le_bytes(len) as usize];
    stream.read_exact(&mut bytes)?;
    Ok(bytes)
}

impl Collective for Tcp {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn exchange(&self, mine: &[u8]) -> Result<Vec<Vec<u8>>> {
        let broken = |e: std::io::Error| DistError(format!("a rank stopped: {e}"));
        let mut peers = self.peers.lock().map_err(|_| stopped())?;
        if self.rank == 0 {
            let mut all = vec![mine.to_vec()];
            for stream in peers.iter_mut().skip(1) {
                let stream = stream.as_mut().ok_or_else(stopped)?;
                all.push(receive(stream).map_err(broken)?);
            }
            let mut packed = Vec::new();
            for bytes in &all {
                packed.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                packed.extend_from_slice(bytes);
            }
            for stream in peers.iter_mut().skip(1) {
                let stream = stream.as_mut().ok_or_else(stopped)?;
                send(stream, &packed).map_err(broken)?;
            }
            Ok(all)
        } else {
            let stream = peers[0].as_mut().ok_or_else(stopped)?;
            send(stream, mine).map_err(broken)?;
            let packed = receive(stream).map_err(broken)?;
            let mut all = Vec::with_capacity(self.size);
            let mut at = 0;
            while at < packed.len() {
                let len =
                    u64::from_le_bytes(packed[at..at + 8].try_into().expect("8 bytes")) as usize;
                all.push(packed[at + 8..at + 8 + len].to_vec());
                at += 8 + len;
            }
            if all.len() != self.size {
                return Err(DistError("an exchange cut short".into()));
            }
            Ok(all)
        }
    }

    fn gather(&self, mine: &[u8]) -> Result<Option<Vec<Vec<u8>>>> {
        let broken = |e: std::io::Error| DistError(format!("a rank stopped: {e}"));
        let mut peers = self.peers.lock().map_err(|_| stopped())?;
        if self.rank == 0 {
            let mut all = vec![mine.to_vec()];
            for stream in peers.iter_mut().skip(1) {
                let stream = stream.as_mut().ok_or_else(stopped)?;
                all.push(receive(stream).map_err(broken)?);
            }
            Ok(Some(all))
        } else {
            let stream = peers[0].as_mut().ok_or_else(stopped)?;
            send(stream, mine).map_err(broken)?;
            Ok(None)
        }
    }

    fn abort(&self) {
        if let Ok(mut peers) = self.peers.lock() {
            for stream in peers.iter_mut().flatten() {
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
            peers.iter_mut().for_each(|p| *p = None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn each<C: Collective + 'static>(
        ranks: Vec<C>,
        f: impl Fn(&C) -> Vec<f32> + Send + Sync + Copy + 'static,
    ) -> Vec<Vec<f32>> {
        let handles: Vec<_> = ranks
            .into_iter()
            .map(|c| std::thread::spawn(move || f(&c)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    }

    fn values(rank: usize) -> Vec<f32> {
        (0..10)
            .map(|i| 0.1 * (i as f32) + 1e7 * rank as f32 + 0.3)
            .collect()
    }

    fn reduce(c: &impl Collective) -> Vec<f32> {
        let mut v = values(c.rank());
        c.all_reduce_sum(&mut v).unwrap();
        let mut scattered = c.reduce_scatter_sum(&values(c.rank())).unwrap();
        scattered = c.all_gather(&scattered).unwrap();
        assert_eq!(scattered, v);
        v
    }

    /// The sum of every rank is rank 0's values, then each next rank's
    /// added, the same bits on every rank, whichever arrives first.
    #[test]
    fn every_rank_holds_the_sum_in_the_order_of_the_ranks() {
        let mut expected = values(0);
        for r in 1..4 {
            for (e, v) in expected.iter_mut().zip(values(r)) {
                *e += v;
            }
        }
        for got in each(Threads::group(4), reduce) {
            assert_eq!(got, expected);
        }
        let ranks: Vec<Tcp> = {
            let address = "127.0.0.1:29711";
            let handles: Vec<_> = (0..4)
                .map(|r| std::thread::spawn(move || Tcp::connect(address, r, 4).unwrap()))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        };
        for got in each(ranks, reduce) {
            assert_eq!(got, expected);
        }
    }

    /// Each parameter has one owner, the elements shared out within the
    /// largest parameter of an equal part.
    #[test]
    fn owners_share_the_elements_out() {
        let sizes: Vec<(String, usize)> = (0..40)
            .map(|i| (format!("p{i}"), 1000 + 37 * (i * i % 23)))
            .collect();
        let owners = owners(&sizes, 4);
        assert_eq!(owners.len(), 40);
        let total: usize = sizes.iter().map(|s| s.1).sum();
        let largest = sizes.iter().map(|s| s.1).max().unwrap();
        for r in 0..4 {
            let load: usize = sizes
                .iter()
                .filter(|s| owners[&s.0] == r)
                .map(|s| s.1)
                .sum();
            assert!(
                load.abs_diff(total / 4) <= largest,
                "rank {r}: {load} of {total}"
            );
        }
    }

    /// A rank that stops makes the others' exchanges fail, not wait.
    #[test]
    fn a_rank_that_stops_stops_the_others() {
        let got = each(Threads::group(3), |c| {
            if c.rank() == 1 {
                c.abort();
                return vec![];
            }
            let mut v = vec![1.0];
            vec![if c.all_reduce_sum(&mut v).is_err() {
                -1.0
            } else {
                v[0]
            }]
        });
        assert_eq!(got[0], vec![-1.0]);
        assert_eq!(got[2], vec![-1.0]);
    }
}
