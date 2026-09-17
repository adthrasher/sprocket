//! Standalone, database-free benchmark for characterizing raw filesystem
//! latency and noise on a shared/clustered filesystem (e.g. GPFS) under
//! real, possibly "ridiculous," concurrent load from other users/jobs.
//!
//! This has **no dependency on `sprocket`, SQLite, or `sqlx`** -- it exists
//! to answer a narrower question than `gpfs_db_benchmark.rs`: is the
//! variability/latency we see coming from the filesystem itself (contended
//! locking, slow metadata operations, expensive fsync), or is it somehow
//! specific to how SQLite or `sprocket` uses it? Run both benchmarks
//! side-by-side when investigating a performance report: if this benchmark
//! alone reproduces the same wild swings, that confirms the filesystem (not
//! sprocket's code) is the source, which is what our prior GPFS
//! investigation found.
//!
//! Like `gpfs_db_benchmark.rs`, this interleaves several probes within each
//! round so they all sample nearly the same contention window, rather than
//! running each probe as a separate phase -- an earlier version of the DB
//! benchmark that ran phases sequentially produced results that swung wildly
//! (0.4x-459x) between runs purely from time-varying external contention,
//! not from anything the code did differently. Each round performs:
//!
//! - **stat canary**: `fs::metadata` on a small existing file -- the
//!   cheapest possible filesystem operation, used as a noise floor. If this
//!   alone is slow or highly variable, the filesystem is under enough load
//!   that no application-level change can fully compensate.
//! - **fsync-per-write**: write a small chunk and `fsync` the file five
//!   times in a row against one open file handle -- mirrors doing five
//!   separate single-statement SQLite commits (one fsync each).
//! - **single-fsync batch**: write five small chunks to one open file
//!   handle, then `fsync` once -- mirrors batching five SQLite writes into
//!   one transaction/commit.
//! - **create+delete cycle**: create a new file, write, fsync, close, then
//!   remove it -- mirrors SQLite's `DELETE` journal mode, which creates and
//!   deletes a rollback-journal file on every commit; SQLite's own docs call
//!   this out as often the slowest part of a commit on network filesystems.
//! - **reuse-file cycle**: truncate and rewrite the same file every round
//!   without ever deleting it -- mirrors `PERSIST` journal mode, which keeps
//!   the journal file present between commits instead of recreating it.
//! - **advisory-lock cycle** (Unix only): acquires and releases a
//!   `fcntl(F_SETLKW)` byte-range write lock at the same file offset SQLite
//!   uses for its `PENDING_BYTE`/`RESERVED_BYTE` rollback-journal locking
//!   protocol (`0x40000000`, i.e. just past the 1GiB mark) -- no data is
//!   written or synced at all here. Note this is *uncontended* (only one
//!   process ever holds it), since POSIX `fcntl` locks are associated with
//!   the owning process/inode pair, not a file descriptor -- a second lock
//!   request from the *same* process never conflicts with its own earlier
//!   lock.
//! - **contended-lock wait** (Unix only): spawns a fresh child process that
//!   acquires the same `fcntl` lock and holds it for a fixed, short
//!   duration, then times how long this process's own blocking lock
//!   request on the same byte range takes to succeed. This is the genuine
//!   cross-process lock-handoff cost -- the closest single-machine proxy
//!   for what happens when multiple concurrent SQLite writers (or
//!   processes on different cluster nodes) actually contend for the same
//!   database lock, as opposed to the uncontended `advisory-lock cycle`
//!   probe above. If raw I/O and the uncontended lock stay fast/stable
//!   while *this* probe shows large or wildly variable excess over its
//!   artificial hold duration, that would confirm the cost is specifically
//!   in cross-process/cross-node lock arbitration (e.g. GPFS's distributed
//!   lock manager revoking/transferring a lock token) rather than in raw
//!   filesystem I/O or even uncontended locking.
//! - **rename cycle**: `fs::rename` a pre-existing temp file into place --
//!   the standard atomic-durable-write pattern (write to a temp file, then
//!   rename over the target), isolated from the write/fsync cost already
//!   measured above so it reflects pure rename metadata cost.
//! - **mkdir+rmdir cycle**: create then remove an empty directory -- pure
//!   directory metadata cost, with no file data involved at all.
//! - **chmod+utime cycle** (Unix only): toggles a file's permission bits
//!   and touches its mtime/atime -- pure attribute-metadata cost, with no
//!   data write.
//!
//! After the interleaved rounds, a separate one-time **metadata scaling**
//! section measures how create/stat/readdir cost changes as a directory's
//! entry count grows (10 / 1,000 / 5,000 files) -- a classic GPFS weak
//! point, since metadata operations often degrade with directory fan-out on
//! network/clustered filesystems even when raw I/O throughput is fine -- and
//! compares spreading many small files across one directory vs. many
//! separate directories, to isolate per-directory lock contention from pure
//! file-count cost.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example fs_benchmark -- <target-directory> [rounds]
//! ```
//!
//! `<target-directory>` should be a path on the filesystem you want to
//! measure (e.g. a directory under a GPFS mount); it will be created if it
//! does not already exist, and all files this benchmark creates live inside
//! it (cleaned up at the end, on success). `rounds` defaults to 40 if
//! omitted.

use std::env;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Result as IoResult;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

const CHUNK: &[u8] = b"benchmark payload line\n";
const WRITES_PER_ROUND: usize = 5;

/// How long the child process in the contended-lock probe holds the lock
/// before releasing it, in milliseconds. The signal of interest is not this
/// value itself but how much *more* than this the parent's wait ends up
/// being -- that excess is lock-handoff overhead.
#[cfg(unix)]
const CONTENDED_HOLD_MS: u64 = 5;

/// Hidden subcommand used internally to spawn a child process that holds
/// the SQLite-style advisory lock for a fixed duration, so the parent can
/// measure genuine cross-process lock-contention wait time (POSIX `fcntl`
/// locks are associated with the owning *process*, not a file descriptor --
/// a second lock request from the *same* process on the same file never
/// conflicts with its own earlier lock, so contention can only be observed
/// against a genuinely different process).
#[cfg(unix)]
const HOLD_LOCK_ARG: &str = "__hold_lock__";

/// The byte offset SQLite's Unix VFS uses for its `PENDING_BYTE` /
/// `RESERVED_BYTE` rollback-journal advisory locks: one byte past the 1GiB
/// mark (`sqlite3.c`'s `os_unix.c`, `PENDING_BYTE = 0x40000000`). Locking
/// this same offset (rather than, say, offset 0) reproduces SQLite's exact
/// locking pattern, including whatever cost a network filesystem's lock
/// manager attaches to advisory locks at large/sparse offsets specifically.
#[cfg(unix)]
const SQLITE_PENDING_BYTE: libc::off_t = 0x4000_0000;

fn main() -> IoResult<()> {
    #[cfg(unix)]
    {
        // Hidden child-process entry point for the contended-lock probe --
        // see `HOLD_LOCK_ARG`'s doc comment. Must be checked before the
        // normal argument parsing below.
        let mut raw_args = env::args().skip(1);
        if raw_args.next().as_deref() == Some(HOLD_LOCK_ARG) {
            let path: PathBuf = raw_args.next().expect("missing lock path").into();
            let hold_ms: u64 = raw_args
                .next()
                .expect("missing hold duration")
                .parse()
                .expect("hold duration must be an integer");
            return hold_lock_child(&path, hold_ms);
        }
    }

    let mut args = env::args().skip(1);
    let dir: PathBuf = match args.next() {
        Some(dir) => dir.into(),
        None => {
            eprintln!("usage: fs_benchmark <target-directory> [rounds]");
            std::process::exit(1);
        }
    };
    let rounds: usize = args
        .next()
        .map(|s| s.parse().expect("rounds must be a non-negative integer"))
        .unwrap_or(40);

    println!("probing filesystem at {}", dir.display());
    fs::create_dir_all(&dir)?;

    // A stable, pre-existing file for the stat canary, so that probe never
    // has to pay file-creation cost itself.
    let canary_path = dir.join("canary.txt");
    fs::write(&canary_path, b"canary\n")?;

    // The `reuse-file` probe's file is created once, up front, then
    // truncated and rewritten every round -- never deleted -- mirroring
    // `PERSIST` journal mode.
    let reuse_path = dir.join("reuse.dat");
    fs::write(&reuse_path, b"")?;

    // The advisory-lock probe's file is created once, up front, and only
    // ever locked/unlocked -- never written to.
    #[cfg(unix)]
    let lock_path = dir.join("lock.dat");
    #[cfg(unix)]
    fs::write(&lock_path, b"")?;
    #[cfg(unix)]
    let lock_file = OpenOptions::new().read(true).write(true).open(&lock_path)?;

    // The contended-lock probe reuses a separate file (and spawns a fresh
    // child process each round) so it never interferes with the
    // uncontended advisory-lock probe above.
    #[cfg(unix)]
    let contended_lock_path = dir.join("contended-lock.dat");
    #[cfg(unix)]
    fs::write(&contended_lock_path, b"")?;
    #[cfg(unix)]
    let current_exe = env::current_exe()?;

    // The rename-cycle probe: a fresh temp source file is created (outside
    // the timer, each round) then renamed over this fixed target path, so
    // the timed portion reflects pure rename metadata cost, not the cost of
    // creating the source file.
    let rename_target_path = dir.join("rename-target.dat");
    fs::write(&rename_target_path, b"")?;

    // The chmod+utime-cycle probe's target file, created once up front and
    // reused every round.
    #[cfg(unix)]
    let chmod_path = dir.join("chmod-target.dat");
    #[cfg(unix)]
    fs::write(&chmod_path, b"")?;

    let mut stats = Vec::with_capacity(rounds);
    let mut per_write_fsync = Vec::with_capacity(rounds);
    let mut single_fsync_batch = Vec::with_capacity(rounds);
    let mut create_delete = Vec::with_capacity(rounds);
    let mut reuse = Vec::with_capacity(rounds);
    #[cfg(unix)]
    let mut locks = Vec::with_capacity(rounds);
    #[cfg(unix)]
    let mut contended_locks = Vec::with_capacity(rounds);
    let mut renames = Vec::with_capacity(rounds);
    let mut mkdir_rmdirs = Vec::with_capacity(rounds);
    #[cfg(unix)]
    let mut chmod_utimes = Vec::with_capacity(rounds);

    for i in 0..rounds {
        let start = Instant::now();
        fs::metadata(&canary_path)?;
        let stat_elapsed = start.elapsed();

        let start = Instant::now();
        fsync_per_write(&dir, i)?;
        let per_write_elapsed = start.elapsed();

        let start = Instant::now();
        single_fsync(&dir, i)?;
        let single_fsync_elapsed = start.elapsed();

        let start = Instant::now();
        create_delete_cycle(&dir, i)?;
        let create_delete_elapsed = start.elapsed();

        let start = Instant::now();
        reuse_cycle(&reuse_path)?;
        let reuse_elapsed = start.elapsed();

        let rename_tmp_path = dir.join(format!("rename-tmp-{i}.dat"));
        fs::write(&rename_tmp_path, CHUNK)?;
        let start = Instant::now();
        fs::rename(&rename_tmp_path, &rename_target_path)?;
        let rename_elapsed = start.elapsed();

        let mkdir_path = dir.join(format!("mkdir-{i}"));
        let start = Instant::now();
        fs::create_dir(&mkdir_path)?;
        fs::remove_dir(&mkdir_path)?;
        let mkdir_rmdir_elapsed = start.elapsed();

        #[cfg(unix)]
        let chmod_utime_elapsed = {
            let start = Instant::now();
            chmod_utime_cycle(&chmod_path, i)?;
            start.elapsed()
        };

        #[cfg(unix)]
        let lock_elapsed = {
            let start = Instant::now();
            advisory_lock_cycle(&lock_file)?;
            start.elapsed()
        };
        #[cfg(unix)]
        let contended_lock_elapsed =
            contended_lock_probe(&current_exe, &contended_lock_path)?;
        #[cfg(unix)]
        let lock_suffix = format!(
            " | advisory-lock {lock_elapsed:.3?} | contended-lock {contended_lock_elapsed:.3?} | \
             chmod+utime {chmod_utime_elapsed:.3?}"
        );
        #[cfg(not(unix))]
        let lock_suffix = String::new();

        println!(
            "round {i}: stat {stat_elapsed:.3?} | fsync-per-write {per_write_elapsed:.3?} | \
             single-fsync {single_fsync_elapsed:.3?} ({:.2}x) | create+delete \
             {create_delete_elapsed:.3?} | reuse {reuse_elapsed:.3?} ({:.2}x) | rename \
             {rename_elapsed:.3?} | mkdir+rmdir {mkdir_rmdir_elapsed:.3?}{lock_suffix}",
            per_write_elapsed.as_secs_f64() / single_fsync_elapsed.as_secs_f64().max(f64::EPSILON),
            create_delete_elapsed.as_secs_f64() / reuse_elapsed.as_secs_f64().max(f64::EPSILON),
        );

        stats.push(stat_elapsed);
        per_write_fsync.push(per_write_elapsed);
        single_fsync_batch.push(single_fsync_elapsed);
        create_delete.push(create_delete_elapsed);
        reuse.push(reuse_elapsed);
        #[cfg(unix)]
        locks.push(lock_elapsed);
        #[cfg(unix)]
        contended_locks.push(contended_lock_elapsed);
        renames.push(rename_elapsed);
        mkdir_rmdirs.push(mkdir_rmdir_elapsed);
        #[cfg(unix)]
        chmod_utimes.push(chmod_utime_elapsed);
    }

    println!();
    println!(
        "stat canary (noise floor):        median {:.3?}, min {:.3?}, max {:.3?}",
        median(&stats),
        stats.iter().min().copied().unwrap_or_default(),
        stats.iter().max().copied().unwrap_or_default(),
    );
    report("fsync-per-write (5 fsyncs)", &per_write_fsync);
    report("single-fsync batch (1 fsync)", &single_fsync_batch);
    report("create+delete cycle (DELETE-like)", &create_delete);
    report("reuse-file cycle (PERSIST-like)", &reuse);
    report("rename cycle (metadata only)", &renames);
    report("mkdir+rmdir cycle (metadata only)", &mkdir_rmdirs);
    #[cfg(unix)]
    report("chmod+utime cycle (metadata only)", &chmod_utimes);
    #[cfg(unix)]
    report("advisory-lock cycle (fcntl only, no I/O)", &locks);
    #[cfg(unix)]
    report(
        &format!("contended-lock wait (vs {CONTENDED_HOLD_MS}ms hold)"),
        &contended_locks,
    );

    println!();
    print_comparison(
        "fsync batching",
        "fsync-per-write",
        &per_write_fsync,
        "single-fsync",
        &single_fsync_batch,
    );
    print_comparison(
        "avoiding create+delete",
        "create+delete",
        &create_delete,
        "reuse-file",
        &reuse,
    );
    #[cfg(unix)]
    {
        let stat_med = median(&stats).as_secs_f64().max(f64::EPSILON);
        let lock_med = median(&locks).as_secs_f64();
        let lock_max = locks.iter().max().copied().unwrap_or_default();
        println!(
            "advisory-lock overhead: median {:.2}x the stat noise floor (lock max {lock_max:.3?}); \
             if this is disproportionately slow/variable compared to the raw-I/O probes above, \
             that points at lock arbitration (e.g. GPFS's distributed lock manager) rather than \
             I/O throughput as the source of SQLite-specific latency",
            lock_med / stat_med,
        );

        let hold = Duration::from_millis(CONTENDED_HOLD_MS);
        let excess_med = median(&contended_locks).saturating_sub(hold);
        let excess_max = contended_locks
            .iter()
            .map(|d| d.saturating_sub(hold))
            .max()
            .unwrap_or_default();
        println!(
            "contended-lock overhead: median excess over the {CONTENDED_HOLD_MS}ms artificial \
             hold is {excess_med:.3?} (max excess {excess_max:.3?}); this is the genuine \
             cross-process lock-handoff cost -- large or wildly variable excess here (unlike the \
             uncontended advisory-lock probe above) would confirm GPFS's lock manager is slow \
             specifically when two processes actually contend for the same byte range, which is \
             what happens under real concurrent SQLite writers",
        );
    }

    run_metadata_scaling_probes(&dir)?;

    fs::remove_dir_all(&dir)?;

    Ok(())
}

/// Directory entry counts used by [`run_metadata_scaling_probes`] to see how
/// create/stat/readdir cost scales with a directory's fan-out.
const FANOUT_SIZES: &[usize] = &[10, 1_000, 5_000];

/// Number of timed trials averaged for each per-size measurement in
/// [`run_metadata_scaling_probes`].
const FANOUT_TRIALS: usize = 10;

/// Number of files used by the single-directory-vs-many-directories
/// comparison in [`run_metadata_scaling_probes`].
const FANOUT_COMPARISON_COUNT: usize = 200;

/// One-time (not interleaved-round) measurements of how metadata operation
/// cost scales with directory fan-out, and whether spreading files across
/// many directories avoids contention that a single large directory
/// experiences. Printed as a separate report section after the main
/// interleaved rounds, since these inherently vary the scale of a fixed
/// setup rather than comparing two things at one moment in time.
fn run_metadata_scaling_probes(dir: &Path) -> IoResult<()> {
    println!();
    println!("metadata scaling (directory fan-out):");

    for &size in FANOUT_SIZES {
        let fanout_dir = dir.join(format!("fanout-{size}"));
        fs::create_dir_all(&fanout_dir)?;
        let mut existing = Vec::with_capacity(size);
        for n in 0..size {
            let path = fanout_dir.join(format!("existing-{n}.dat"));
            fs::write(&path, CHUNK)?;
            existing.push(path);
        }

        let mut creates = Vec::with_capacity(FANOUT_TRIALS);
        let mut stats = Vec::with_capacity(FANOUT_TRIALS);
        for t in 0..FANOUT_TRIALS {
            let new_path = fanout_dir.join(format!("new-{t}.dat"));
            let start = Instant::now();
            let mut file = File::create(&new_path)?;
            file.write_all(CHUNK)?;
            file.sync_all()?;
            drop(file);
            fs::remove_file(&new_path)?;
            creates.push(start.elapsed());

            let probe_target = &existing[t % existing.len()];
            let start = Instant::now();
            fs::metadata(probe_target)?;
            stats.push(start.elapsed());
        }

        let start = Instant::now();
        let scanned = fs::read_dir(&fanout_dir)?.count();
        let scan_elapsed = start.elapsed();

        println!(
            "  {size:>5} entries: create+fsync+delete median {:.3?} | stat median {:.3?} | \
             readdir({scanned} entries) {scan_elapsed:.3?}",
            median(&creates),
            median(&stats),
        );

        fs::remove_dir_all(&fanout_dir)?;
    }

    println!();
    println!(
        "single-directory vs. many-directories ({FANOUT_COMPARISON_COUNT} files, create+delete \
         each):"
    );

    let single_dir = dir.join("single-dir");
    fs::create_dir_all(&single_dir)?;
    let start = Instant::now();
    for n in 0..FANOUT_COMPARISON_COUNT {
        let path = single_dir.join(format!("{n}.dat"));
        fs::write(&path, CHUNK)?;
        fs::remove_file(&path)?;
    }
    let single_dir_elapsed = start.elapsed();
    fs::remove_dir_all(&single_dir)?;

    let many_dirs = dir.join("many-dirs");
    fs::create_dir_all(&many_dirs)?;
    let start = Instant::now();
    for n in 0..FANOUT_COMPARISON_COUNT {
        let sub = many_dirs.join(format!("dir-{n}"));
        fs::create_dir(&sub)?;
        let path = sub.join("file.dat");
        fs::write(&path, CHUNK)?;
        fs::remove_file(&path)?;
        fs::remove_dir(&sub)?;
    }
    let many_dirs_elapsed = start.elapsed();
    fs::remove_dir_all(&many_dirs)?;

    println!(
        "  single directory: total {single_dir_elapsed:.3?} | many directories (own dir per \
         file): total {many_dirs_elapsed:.3?} ({:.2}x)",
        single_dir_elapsed.as_secs_f64() / many_dirs_elapsed.as_secs_f64().max(f64::EPSILON),
    );
    println!(
        "  (many-directories also pays extra mkdir/rmdir cost per file, so a ratio near 1x or \
         below doesn't rule out single-directory contention -- it means that extra cost roughly \
         offset any contention avoided)"
    );

    Ok(())
}

/// Child-process entry point for the contended-lock probe (see
/// [`HOLD_LOCK_ARG`]): opens `path`, takes the SQLite-style advisory lock,
/// signals readiness on stdout, holds the lock for `hold_ms`, then releases
/// it and exits.
#[cfg(unix)]
fn hold_lock_child(path: &Path, hold_ms: u64) -> IoResult<()> {
    use std::os::unix::io::AsRawFd;

    let file = OpenOptions::new().read(true).write(true).open(path)?;
    let fd = file.as_raw_fd();
    let mut lock = libc::flock {
        l_type: libc::F_WRLCK as libc::c_short,
        l_whence: libc::SEEK_SET as libc::c_short,
        l_start: SQLITE_PENDING_BYTE,
        l_len: 1,
        l_pid: 0,
    };

    // SAFETY: `fd` is a valid, open file descriptor for the duration of this
    // call, and `lock` is a properly initialized `flock` struct passed as
    // `fcntl` expects for `F_SETLKW`.
    let rc = unsafe { libc::fcntl(fd, libc::F_SETLKW, &mut lock as *mut libc::flock) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }

    // Signal the parent that the lock is now held, so it only starts timing
    // its own (blocking) lock attempt once contention is guaranteed.
    println!("READY");
    std::io::stdout().flush()?;

    std::thread::sleep(Duration::from_millis(hold_ms));

    lock.l_type = libc::F_UNLCK as libc::c_short;
    // SAFETY: same as above -- releasing the lock this process just held.
    let rc = unsafe { libc::fcntl(fd, libc::F_SETLK, &mut lock as *mut libc::flock) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }

    Ok(())
}

/// Spawns a fresh child process that holds the SQLite-style advisory lock
/// for [`CONTENDED_HOLD_MS`], waits for its "lock held" signal, then times
/// how long this (parent) process's own blocking `F_SETLKW` request on the
/// same byte range takes to succeed -- the genuine cross-process
/// lock-contention wait, as opposed to `advisory_lock_cycle`'s uncontended
/// acquire/release.
#[cfg(unix)]
fn contended_lock_probe(exe: &Path, path: &Path) -> IoResult<Duration> {
    use std::os::unix::io::AsRawFd;

    let mut child = Command::new(exe)
        .arg(HOLD_LOCK_ARG)
        .arg(path)
        .arg(CONTENDED_HOLD_MS.to_string())
        .stdout(Stdio::piped())
        .spawn()?;

    // Block until the child confirms it holds the lock, so the timing below
    // only measures genuine contention, not process-startup latency.
    let mut ready_line = String::new();
    BufReader::new(child.stdout.take().expect("piped stdout"))
        .read_line(&mut ready_line)?;

    let file = OpenOptions::new().read(true).write(true).open(path)?;
    let fd = file.as_raw_fd();
    let mut lock = libc::flock {
        l_type: libc::F_WRLCK as libc::c_short,
        l_whence: libc::SEEK_SET as libc::c_short,
        l_start: SQLITE_PENDING_BYTE,
        l_len: 1,
        l_pid: 0,
    };

    let start = Instant::now();
    // SAFETY: `fd` is a valid, open file descriptor for the duration of this
    // call, and `lock` is a properly initialized `flock` struct passed as
    // `fcntl` expects for the blocking `F_SETLKW` command.
    let rc = unsafe { libc::fcntl(fd, libc::F_SETLKW, &mut lock as *mut libc::flock) };
    let elapsed = start.elapsed();
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }

    lock.l_type = libc::F_UNLCK as libc::c_short;
    // SAFETY: same as above -- releasing the lock we just acquired.
    let rc = unsafe { libc::fcntl(fd, libc::F_SETLK, &mut lock as *mut libc::flock) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }

    child.wait()?;
    Ok(elapsed)
}

/// Toggles a file's permission bits and touches its mtime/atime -- mirrors
/// the pure-metadata `chmod(2)`/`utimensat(2)` operations SQLite (and other
/// tools) occasionally issue against the database file, independent of any
/// data write.
#[cfg(unix)]
fn chmod_utime_cycle(path: &Path, round: usize) -> IoResult<()> {
    use std::os::unix::fs::PermissionsExt;

    let mode = if round % 2 == 0 { 0o600 } else { 0o644 };
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    touch_now(path)
}

/// Sets both mtime and atime of `path` to "now" via `utimensat(2)`, without
/// reading or writing any file content.
#[cfg(unix)]
fn touch_now(path: &Path) -> IoResult<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let now = libc::timespec {
        tv_sec: 0,
        tv_nsec: libc::UTIME_NOW,
    };
    let times = [now, now];
    // SAFETY: `c_path` is a valid NUL-terminated C string for the lifetime
    // of this call, and `times` points to a valid two-element array as
    // `utimensat` requires.
    let rc = unsafe { libc::utimensat(libc::AT_FDCWD, c_path.as_ptr(), times.as_ptr(), 0) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Writes and fsyncs [`WRITES_PER_ROUND`] times against one open file
/// handle -- mirrors five separate single-statement SQLite commits.
fn fsync_per_write(dir: &Path, round: usize) -> IoResult<()> {
    let path = dir.join(format!("per-write-{round}.dat"));
    let mut file = File::create(&path)?;
    for _ in 0..WRITES_PER_ROUND {
        file.write_all(CHUNK)?;
        file.sync_all()?;
    }
    fs::remove_file(&path)
}

/// Writes [`WRITES_PER_ROUND`] chunks to one open file handle, then fsyncs
/// once -- mirrors batching five SQLite writes into one commit.
fn single_fsync(dir: &Path, round: usize) -> IoResult<()> {
    let path = dir.join(format!("single-fsync-{round}.dat"));
    let mut file = File::create(&path)?;
    for _ in 0..WRITES_PER_ROUND {
        file.write_all(CHUNK)?;
    }
    file.sync_all()?;
    fs::remove_file(&path)
}

/// Acquires and immediately releases a `fcntl(F_SETLKW)` write lock on a
/// single byte at [`SQLITE_PENDING_BYTE`] -- the same offset and lock type
/// SQLite takes (and blocks on) during its rollback-journal commit
/// protocol. No data is read or written; this isolates the cost of
/// cross-process/cross-node advisory-lock arbitration from the cost of I/O
/// itself.
#[cfg(unix)]
fn advisory_lock_cycle(file: &File) -> IoResult<()> {
    use std::os::unix::io::AsRawFd;

    let fd = file.as_raw_fd();
    let mut lock = libc::flock {
        l_type: libc::F_WRLCK as libc::c_short,
        l_whence: libc::SEEK_SET as libc::c_short,
        l_start: SQLITE_PENDING_BYTE,
        l_len: 1,
        l_pid: 0,
    };

    // SAFETY: `fd` is a valid, open file descriptor owned by `file` for the
    // duration of this call, and `lock` is a properly initialized `flock`
    // struct passed by reference as `fcntl` expects for `F_SETLKW`.
    let rc = unsafe { libc::fcntl(fd, libc::F_SETLKW, &mut lock as *mut libc::flock) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }

    lock.l_type = libc::F_UNLCK as libc::c_short;
    // SAFETY: same as above -- releasing the lock we just acquired on the
    // same fd/range.
    let rc = unsafe { libc::fcntl(fd, libc::F_SETLK, &mut lock as *mut libc::flock) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }

    Ok(())
}

/// Creates a new file, writes and fsyncs it, then deletes it -- mirrors
/// `DELETE` journal mode's per-commit journal-file create/delete cycle.
fn create_delete_cycle(dir: &Path, round: usize) -> IoResult<()> {
    let path = dir.join(format!("create-delete-{round}.dat"));
    let mut file = File::create(&path)?;
    file.write_all(CHUNK)?;
    file.sync_all()?;
    drop(file);
    fs::remove_file(&path)
}

/// Truncates and rewrites an already-existing file without deleting it --
/// mirrors `PERSIST` journal mode reusing the same on-disk journal file.
fn reuse_cycle(path: &Path) -> IoResult<()> {
    let mut file = OpenOptions::new().write(true).truncate(true).open(path)?;
    file.write_all(CHUNK)?;
    file.sync_all()
}

/// Prints median/total for a probe's samples.
fn report(label: &str, samples: &[Duration]) {
    println!(
        "{label:<34} total {:.3?}, median/round {:.3?}",
        sum(samples),
        median(samples),
    );
}

/// Prints a paired win-count/median-speedup comparison between two probes'
/// samples, analogous to the batched-vs-unbatched summary in
/// `gpfs_db_benchmark.rs`.
fn print_comparison(
    title: &str,
    baseline_label: &str,
    baseline: &[Duration],
    improved_label: &str,
    improved: &[Duration],
) {
    let ratios: Vec<f64> = baseline
        .iter()
        .zip(improved)
        .map(|(b, i)| b.as_secs_f64() / i.as_secs_f64().max(f64::EPSILON))
        .collect();
    let wins = ratios.iter().filter(|&&r| r > 1.0).count();
    println!(
        "{title}: {improved_label} won {wins}/{} rounds over {baseline_label}; median per-round \
         speedup {:.2}x; total-time speedup {:.2}x",
        ratios.len(),
        median_f64(&ratios),
        sum(baseline).as_secs_f64() / sum(improved).as_secs_f64().max(f64::EPSILON),
    );
}

/// Returns the median of a slice of durations (average of the two middle
/// elements for an even-length slice), without mutating the input.
fn median(values: &[Duration]) -> Duration {
    let mut sorted: Vec<Duration> = values.to_vec();
    sorted.sort();
    match sorted.len() {
        0 => Duration::ZERO,
        len if len % 2 == 1 => sorted[len / 2],
        len => (sorted[len / 2 - 1] + sorted[len / 2]) / 2,
    }
}

/// Returns the median of a slice of `f64` ratios (average of the two middle
/// elements for an even-length slice), without mutating the input.
fn median_f64(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    match sorted.len() {
        0 => 0.0,
        len if len % 2 == 1 => sorted[len / 2],
        len => (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0,
    }
}

/// Sums a slice of durations.
fn sum(values: &[Duration]) -> Duration {
    values.iter().sum()
}
