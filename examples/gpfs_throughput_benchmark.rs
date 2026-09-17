//! Standalone, database-free benchmark for large-file sequential and
//! random-access I/O throughput on a shared/clustered filesystem (e.g.
//! GPFS).
//!
//! This complements `fs_benchmark.rs`, which focuses on small-file/metadata
//! operations (the size of a single SQLite commit or a directory entry).
//! Neither of those probes ever moves more than a few KiB at once, so
//! neither can say anything about whether *sustained large-file* I/O -- the
//! pattern used when localizing/delocalizing task inputs/outputs -- behaves
//! well on the target filesystem. This tool answers that separately.
//!
//! Has **no dependency on `sprocket`, SQLite, or `sqlx`** -- pure `std::fs`
//! plus `libc` for `posix_fadvise` and `rand` for random-offset selection.
//!
//! Each round performs, against one target file of `file-size-mb` (default
//! 256 MiB):
//!
//! - **sequential write**: write the whole file in fixed 1MiB chunks, fsync
//!   once at the end, report MB/s.
//! - **cache drop**: best-effort `posix_fadvise(..., POSIX_FADV_DONTNEED)`
//!   (Linux-specific; a no-op warning is printed if unavailable/ineffective
//!   on this platform) so the subsequent read measures real filesystem
//!   latency rather than a page-cache hit.
//! - **sequential read**: read the whole file back in fixed 1MiB chunks,
//!   report MB/s.
//! - **random-offset I/O**: `iops-samples` random-offset reads and writes
//!   (small ~64KiB buffer) at random positions within the file, report
//!   median/p95/max latency and aggregate IOPS.
//!
//! After the timed rounds, a one-time **buffer-size sweep** (4KiB / 64KiB /
//! 1MiB) writes and reads a fixed amount of data with each buffer size, to
//! compare small vs. large buffer throughput -- relevant since SQLite's
//! page size is 32KiB (see the DB-performance investigation this tool
//! accompanies).
//!
//! Usage: `cargo run --release --example gpfs_throughput_benchmark -- \
//! <target-dir> [file-size-mb] [rounds]`
//!
//! `file-size-mb` defaults to 256, `rounds` defaults to 5 (large-file
//! rounds are expensive, so this default is much lower than
//! `fs_benchmark.rs`'s 40).

use std::env;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Result as IoResult;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use rand::RngExt;

/// Chunk size used for sequential write/read.
const SEQ_CHUNK_SIZE: usize = 1024 * 1024;

/// Buffer size used for the random-offset I/O probe.
const RANDOM_IO_BUFFER_SIZE: usize = 64 * 1024;

/// Number of random-offset read/write samples per round.
const RANDOM_IO_SAMPLES: usize = 50;

/// Buffer sizes compared in the post-loop sweep.
const SWEEP_BUFFER_SIZES: &[usize] = &[4 * 1024, 64 * 1024, 1024 * 1024];

/// Total bytes moved per buffer size in the sweep (kept modest since this
/// runs once per buffer size, not once per round).
const SWEEP_TOTAL_BYTES: usize = 64 * 1024 * 1024;

fn main() -> IoResult<()> {
    let mut args = env::args().skip(1);
    let dir: PathBuf = args
        .next()
        .unwrap_or_else(|| {
            eprintln!(
                "usage: gpfs_throughput_benchmark <target-dir> [file-size-mb=256] [rounds=5]"
            );
            std::process::exit(1);
        })
        .into();
    let file_size_mb: u64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let rounds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    fs::create_dir_all(&dir)?;
    let file_size = file_size_mb * 1024 * 1024;
    println!(
        "probing large-file I/O at {} ({file_size_mb} MiB file, {rounds} rounds)",
        dir.display()
    );

    let target_path = dir.join("throughput-target.dat");
    // Pre-sized once, up front, via `set_len` (ftruncate) -- overwriting its
    // already-allocated blocks in place should not require the filesystem to
    // perform any new block/extent allocation, unlike `target_path` above
    // which is freshly created and grown every round. Comparing the two
    // isolates "extending a file" (allocation-manager cost) from raw
    // sustained write bandwidth.
    let prealloc_path = dir.join("throughput-prealloc.dat");
    {
        let file = File::create(&prealloc_path)?;
        file.set_len(file_size)?;
        file.sync_all()?;
    }

    let mut write_mbps = Vec::with_capacity(rounds);
    let mut prealloc_write_mbps = Vec::with_capacity(rounds);
    let mut read_mbps = Vec::with_capacity(rounds);
    let mut random_read_latencies = Vec::with_capacity(rounds * RANDOM_IO_SAMPLES);
    let mut random_write_latencies = Vec::with_capacity(rounds * RANDOM_IO_SAMPLES);
    let mut fadvise_ever_attempted = false;
    let mut fadvise_ever_failed = false;

    for round in 0..rounds {
        let write_buf = vec![(round as u8).wrapping_add(1); SEQ_CHUNK_SIZE];
        let start = Instant::now();
        {
            let mut file = File::create(&target_path)?;
            let mut written = 0u64;
            while written < file_size {
                let remaining = (file_size - written).min(SEQ_CHUNK_SIZE as u64) as usize;
                file.write_all(&write_buf[..remaining])?;
                written += remaining as u64;
            }
            file.sync_all()?;
        }
        let write_elapsed = start.elapsed();
        let write_mb_per_sec = mb_per_sec(file_size, write_elapsed);
        write_mbps.push(write_mb_per_sec);

        let start = Instant::now();
        {
            let mut file = fs::OpenOptions::new().write(true).open(&prealloc_path)?;
            file.seek(SeekFrom::Start(0))?;
            let mut written = 0u64;
            while written < file_size {
                let remaining = (file_size - written).min(SEQ_CHUNK_SIZE as u64) as usize;
                file.write_all(&write_buf[..remaining])?;
                written += remaining as u64;
            }
            file.sync_all()?;
        }
        let prealloc_write_elapsed = start.elapsed();
        let prealloc_write_mb_per_sec = mb_per_sec(file_size, prealloc_write_elapsed);
        prealloc_write_mbps.push(prealloc_write_mb_per_sec);

        let (attempted, failed) = drop_cache(&target_path)?;
        fadvise_ever_attempted |= attempted;
        fadvise_ever_failed |= failed;

        let start = Instant::now();
        {
            let mut file = File::open(&target_path)?;
            let mut read_buf = vec![0u8; SEQ_CHUNK_SIZE];
            let mut read_total = 0u64;
            while read_total < file_size {
                let remaining = (file_size - read_total).min(SEQ_CHUNK_SIZE as u64) as usize;
                file.read_exact(&mut read_buf[..remaining])?;
                read_total += remaining as u64;
            }
        }
        let read_elapsed = start.elapsed();
        let read_mb_per_sec = mb_per_sec(file_size, read_elapsed);
        read_mbps.push(read_mb_per_sec);

        let (round_read_latencies, round_write_latencies) =
            random_offset_probe(&target_path, file_size)?;
        let read_median = median_duration(&round_read_latencies);
        let write_median = median_duration(&round_write_latencies);
        random_read_latencies.extend(round_read_latencies);
        random_write_latencies.extend(round_write_latencies);

        println!(
            "round {round}: seq-write(grow) {write_mb_per_sec:.1} MB/s ({write_elapsed:.3?}) | \
             seq-write(prealloc) {prealloc_write_mb_per_sec:.1} MB/s \
             ({prealloc_write_elapsed:.3?}) | seq-read {read_mb_per_sec:.1} MB/s \
             ({read_elapsed:.3?}) | random-read median {read_median:.3?} | random-write median \
             {write_median:.3?}",
        );
    }

    println!();
    print_stats("sequential write (growing file)", &write_mbps, "MB/s");
    print_stats(
        "sequential write (pre-allocated file)",
        &prealloc_write_mbps,
        "MB/s",
    );
    print_stats("sequential read", &read_mbps, "MB/s");
    print_latency_stats("random-offset read", &random_read_latencies);
    print_latency_stats("random-offset write", &random_write_latencies);
    print_comparison_mbps(
        "block allocation overhead",
        "growing file",
        &write_mbps,
        "pre-allocated file",
        &prealloc_write_mbps,
    );

    if fadvise_ever_attempted {
        if fadvise_ever_failed {
            println!(
                "note: posix_fadvise(DONTNEED) was attempted but failed/unavailable at least \
                 once -- sequential-read numbers above may partly reflect page-cache hits rather \
                 than real filesystem latency"
            );
        }
    } else {
        println!(
            "note: posix_fadvise(DONTNEED) is not attempted on this platform -- sequential-read \
             numbers above may partly reflect page-cache hits rather than real filesystem latency"
        );
    }

    run_buffer_size_sweep(&dir)?;

    fs::remove_file(&target_path)?;
    fs::remove_file(&prealloc_path)?;

    Ok(())
}

/// Prints a paired-comparison summary for two series of throughput (MB/s)
/// samples, mirroring `fs_benchmark.rs`'s `print_comparison` but for
/// higher-is-better throughput figures instead of lower-is-better
/// durations.
fn print_comparison_mbps(
    label: &str,
    a_name: &str,
    a: &[f64],
    b_name: &str,
    b: &[f64],
) {
    if a.is_empty() || a.len() != b.len() {
        return;
    }
    let mut b_wins = 0usize;
    for (av, bv) in a.iter().zip(b.iter()) {
        if bv > av {
            b_wins += 1;
        }
    }
    let mut a_sorted = a.to_vec();
    a_sorted.sort_by(|x, y| x.partial_cmp(y).expect("no NaNs in throughput samples"));
    let mut b_sorted = b.to_vec();
    b_sorted.sort_by(|x, y| x.partial_cmp(y).expect("no NaNs in throughput samples"));
    let a_median = a_sorted[a_sorted.len() / 2].max(f64::EPSILON);
    let b_median = b_sorted[b_sorted.len() / 2];
    println!(
        "{label}: {b_name} beat {a_name} in {b_wins}/{} rounds; median {b_name} is {:.2}x \
         {a_name}'s median throughput",
        a.len(),
        b_median / a_median,
    );
}

/// Best-effort attempt to drop this file's pages from the OS page cache via
/// `posix_fadvise(..., POSIX_FADV_DONTNEED)` (Linux-specific; harmless
/// no-op-ish on platforms where it isn't effective). Returns `(attempted,
/// failed)` so the caller can print an accurate caveat about whether
/// subsequent read timings reflect real filesystem latency.
#[cfg(target_os = "linux")]
fn drop_cache(path: &Path) -> IoResult<(bool, bool)> {
    use std::os::unix::io::AsRawFd;

    let file = File::open(path)?;
    let fd = file.as_raw_fd();
    // SAFETY: `fd` is a valid, open file descriptor for the duration of
    // this call; `len = 0` means "to the end of the file" per POSIX.
    let rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
    Ok((true, rc != 0))
}

/// No-op stub on non-Linux platforms, where `posix_fadvise` either doesn't
/// exist or doesn't reliably evict pages (e.g. macOS). Always reports
/// `attempted = false` so the caller prints the "not attempted" caveat.
#[cfg(not(target_os = "linux"))]
fn drop_cache(_path: &Path) -> IoResult<(bool, bool)> {
    Ok((false, false))
}

/// Performs [`RANDOM_IO_SAMPLES`] random-offset reads and the same number
/// of random-offset writes against `path` (assumed to already be
/// `file_size` bytes long), returning the per-sample latencies.
fn random_offset_probe(
    path: &Path,
    file_size: u64,
) -> IoResult<(Vec<Duration>, Vec<Duration>)> {
    let mut rng = rand::rng();
    let max_start = file_size.saturating_sub(RANDOM_IO_BUFFER_SIZE as u64).max(1);

    let mut read_latencies = Vec::with_capacity(RANDOM_IO_SAMPLES);
    {
        let mut file = File::open(path)?;
        let mut buf = vec![0u8; RANDOM_IO_BUFFER_SIZE];
        for _ in 0..RANDOM_IO_SAMPLES {
            let offset = rng.random_range(0..max_start);
            let start = Instant::now();
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut buf)?;
            read_latencies.push(start.elapsed());
        }
    }

    let mut write_latencies = Vec::with_capacity(RANDOM_IO_SAMPLES);
    {
        let mut file = fs::OpenOptions::new().write(true).open(path)?;
        let buf = vec![0xABu8; RANDOM_IO_BUFFER_SIZE];
        for _ in 0..RANDOM_IO_SAMPLES {
            let offset = rng.random_range(0..max_start);
            let start = Instant::now();
            file.seek(SeekFrom::Start(offset))?;
            file.write_all(&buf)?;
            file.sync_data()?;
            write_latencies.push(start.elapsed());
        }
    }

    Ok((read_latencies, write_latencies))
}

/// One-time comparison of write/read throughput at three buffer sizes
/// (4KiB / 64KiB / 1MiB), moving a fixed [`SWEEP_TOTAL_BYTES`] of data at
/// each size. Run once (not per round) since it targets a scaling
/// question, not paired noise cancellation.
fn run_buffer_size_sweep(dir: &Path) -> IoResult<()> {
    println!();
    println!(
        "buffer-size sweep ({} MiB moved per size):",
        SWEEP_TOTAL_BYTES / (1024 * 1024)
    );

    let sweep_path = dir.join("sweep-target.dat");
    for &buf_size in SWEEP_BUFFER_SIZES {
        let buf = vec![0x5Au8; buf_size];
        let iterations = SWEEP_TOTAL_BYTES / buf_size;

        let start = Instant::now();
        {
            let mut file = File::create(&sweep_path)?;
            for _ in 0..iterations {
                file.write_all(&buf)?;
            }
            file.sync_all()?;
        }
        let write_elapsed = start.elapsed();
        let write_mb_per_sec = mb_per_sec((iterations * buf_size) as u64, write_elapsed);

        drop_cache(&sweep_path)?;

        let start = Instant::now();
        {
            let mut file = File::open(&sweep_path)?;
            let mut read_buf = vec![0u8; buf_size];
            for _ in 0..iterations {
                file.read_exact(&mut read_buf)?;
            }
        }
        let read_elapsed = start.elapsed();
        let read_mb_per_sec = mb_per_sec((iterations * buf_size) as u64, read_elapsed);

        println!(
            "  {buf_size:>7} byte buffer: write {write_mb_per_sec:.1} MB/s ({write_elapsed:.3?}) \
             | read {read_mb_per_sec:.1} MB/s ({read_elapsed:.3?})",
        );
    }

    fs::remove_file(&sweep_path)?;
    Ok(())
}

/// Converts `bytes` moved over `elapsed` into a MB/s throughput figure.
fn mb_per_sec(bytes: u64, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64().max(f64::EPSILON);
    (bytes as f64 / (1024.0 * 1024.0)) / secs
}

/// Prints median/min/max for a series of throughput samples (in MB/s or
/// similar).
fn print_stats(label: &str, samples: &[f64], unit: &str) {
    if samples.is_empty() {
        return;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs in throughput samples"));
    let median = sorted[sorted.len() / 2];
    let min = sorted.first().copied().unwrap_or_default();
    let max = sorted.last().copied().unwrap_or_default();
    println!("{label}: median {median:.1} {unit}, min {min:.1} {unit}, max {max:.1} {unit}");
}

/// Prints median/p95/max for a series of latencies, plus an aggregate IOPS
/// figure derived from the mean latency.
fn print_latency_stats(label: &str, samples: &[Duration]) {
    if samples.is_empty() {
        return;
    }
    let mut sorted = samples.to_vec();
    sorted.sort();
    let median = sorted[sorted.len() / 2];
    let p95_index = ((sorted.len() as f64) * 0.95) as usize;
    let p95 = sorted[p95_index.min(sorted.len() - 1)];
    let max = sorted.last().copied().unwrap_or_default();
    let mean_secs =
        sorted.iter().map(Duration::as_secs_f64).sum::<f64>() / sorted.len() as f64;
    let iops = 1.0 / mean_secs.max(f64::EPSILON);
    println!(
        "{label}: median {median:.3?}, p95 {p95:.3?}, max {max:.3?}, ~{iops:.0} IOPS \
         ({} samples)",
        sorted.len(),
    );
}

/// Median of a `Duration` slice (upper-median for even-length slices,
/// matching `fs_benchmark.rs`'s convention).
fn median_duration(values: &[Duration]) -> Duration {
    if values.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = values.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}
