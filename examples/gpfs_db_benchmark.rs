//! Standalone benchmark for measuring `sprocket`'s SQLite commit latency
//! against a real deployment filesystem (e.g. GPFS or another clustered/
//! network filesystem).
//!
//! This is **not** part of the test suite or CI: local disks and CI runners
//! don't reproduce the per-commit network round-trip cost that motivated the
//! journal-mode/mmap/connection-pool tuning and the task-monitor write
//! batching in `src/system/v1/db` and `src/system/v1/exec/svc/task_monitor.rs`.
//! Run this manually against the target filesystem to measure real
//! before/after commit latency for those changes.
//!
//! It opens the database exactly the way `sprocket` does in production (via
//! [`SqliteDatabase::new`], so it exercises the same pragmas: journal mode,
//! `busy_timeout`, `mmap_size`, and connection pool size).
//!
//! # Why this interleaves batched and unbatched rounds
//!
//! An earlier version of this benchmark ran the entire unbatched workload
//! first and the entire batched workload second. On a shared, contended
//! filesystem that produced wildly inconsistent results run-to-run (total-
//! time ratios from 0.4x to 459x favoring batching, observed across five
//! back-to-back runs on real GPFS), because each phase sampled a
//! *different*, independently-varying slice of real-time contention -- a
//! load spike or lull during one phase but not the other completely swamps
//! any signal from batching itself. The tell was that even the isolated
//! single-row heartbeat canary swung by three orders of magnitude between
//! runs with no code change involved.
//!
//! This version instead alternates a small "round" of unbatched writes with
//! an equivalent round of batched writes (plus a heartbeat canary), so both
//! see nearly the same contention window a few milliseconds apart. This
//! makes a *single* run's paired comparison meaningful, rather than needing
//! many repeated runs and a geometric mean to see past the noise.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example gpfs_db_benchmark -- <path-to-db-file> [rounds]
//! ```
//!
//! `<path-to-db-file>` should point at a location on the filesystem you want
//! to measure (e.g. a path under a GPFS mount); it will be created if it does
//! not already exist. `rounds` defaults to 40 if omitted; each round performs
//! one heartbeat write, one task's worth of unbatched lifecycle writes (5
//! commits), and one task's worth of batched lifecycle writes (1 commit).

use std::env;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use chrono::Utc;
use sprocket::system::v1::db::Database;
use sprocket::system::v1::db::LogSource;
use sprocket::system::v1::db::NewTask;
use sprocket::system::v1::db::SprocketCommand;
use sprocket::system::v1::db::TaskStatus;
use sprocket::system::v1::db::TaskWrite;
use sprocket::system::v1::db::sqlite::SqliteDatabase;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let db_path: PathBuf = args
        .next()
        .context("usage: gpfs_db_benchmark <path-to-db-file> [rounds]")?
        .into();
    let rounds: usize = args
        .next()
        .map(|s| s.parse())
        .transpose()
        .context("rounds must be a non-negative integer")?
        .unwrap_or(40);

    println!("opening database at {}", db_path.display());
    let db = SqliteDatabase::new(&db_path)
        .await
        .context("failed to open database")?;

    let session_id = Uuid::new_v4();
    db.create_session(session_id, SprocketCommand::Run, "gpfs-benchmark")
        .await
        .context("failed to create session")?;

    let run_id = Uuid::new_v4();
    db.create_run(
        run_id,
        session_id,
        "gpfs-benchmark",
        "benchmark.wdl",
        Some("t"),
        "{}",
    )
    .await
    .context("failed to create run")?;

    let mut heartbeats = Vec::with_capacity(rounds);
    let mut unbatched = Vec::with_capacity(rounds);
    let mut batched = Vec::with_capacity(rounds);

    for i in 0..rounds {
        let start = Instant::now();
        db.heartbeat_session(session_id, Utc::now())
            .await
            .context("failed to record heartbeat")?;
        let heartbeat_elapsed = start.elapsed();

        let start = Instant::now();
        unbatched_task_writes(&db, run_id, &format!("unbatched-task-{i}")).await?;
        let unbatched_elapsed = start.elapsed();

        let start = Instant::now();
        batched_task_writes(&db, run_id, &format!("batched-task-{i}")).await?;
        let batched_elapsed = start.elapsed();

        println!(
            "round {i}: heartbeat {heartbeat_elapsed:.3?} | unbatched {unbatched_elapsed:.3?} | \
             batched {batched_elapsed:.3?} ({:.2}x)",
            unbatched_elapsed.as_secs_f64() / batched_elapsed.as_secs_f64().max(f64::EPSILON)
        );

        heartbeats.push(heartbeat_elapsed);
        unbatched.push(unbatched_elapsed);
        batched.push(batched_elapsed);
    }

    println!();
    println!(
        "heartbeat canary (environmental noise floor): median {:.3?}, min {:.3?}, max {:.3?}",
        median(&heartbeats),
        heartbeats.iter().min().copied().unwrap_or_default(),
        heartbeats.iter().max().copied().unwrap_or_default(),
    );
    println!(
        "unbatched task writes: total {:.3?}, median/round {:.3?}",
        sum(&unbatched),
        median(&unbatched),
    );
    println!(
        "batched task writes:   total {:.3?}, median/round {:.3?}",
        sum(&batched),
        median(&batched),
    );

    let ratios: Vec<f64> = unbatched
        .iter()
        .zip(&batched)
        .map(|(u, b)| u.as_secs_f64() / b.as_secs_f64().max(f64::EPSILON))
        .collect();
    let wins = ratios.iter().filter(|&&r| r > 1.0).count();
    println!(
        "batching won {wins}/{rounds} rounds; median per-round speedup {:.2}x; total-time \
         speedup {:.2}x",
        median_f64(&ratios),
        sum(&unbatched).as_secs_f64() / sum(&batched).as_secs_f64().max(f64::EPSILON),
    );

    Ok(())
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

/// Performs one task's worth of lifecycle writes (create, start, two log
/// chunks, complete) as five separate commits -- the behavior prior to the
/// task-monitor batching redesign.
async fn unbatched_task_writes(db: &SqliteDatabase, run_id: Uuid, name: &str) -> Result<()> {
    db.create_task(NewTask {
        name,
        run_id,
        status: TaskStatus::Initializing,
        call_id: Some(name),
        attempt: 0,
    })
    .await
    .context("failed to create task")?;

    db.update_task_started(name, Utc::now())
        .await
        .context("failed to mark task started")?;

    db.insert_task_log(name, LogSource::Stdout, b"line one\n")
        .await
        .context("failed to insert log")?;

    db.insert_task_log(name, LogSource::Stdout, b"line two\n")
        .await
        .context("failed to insert log")?;

    db.update_task_completed(name, Some(0), Utc::now())
        .await
        .context("failed to mark task completed")?;

    Ok(())
}

/// Performs the same task lifecycle writes as [`unbatched_task_writes`], but
/// flushed in a single transaction via [`Database::apply_task_writes`] --
/// the current, batched behavior.
async fn batched_task_writes(db: &SqliteDatabase, run_id: Uuid, name: &str) -> Result<()> {
    db.apply_task_writes(vec![
        TaskWrite::Create {
            name: name.to_string(),
            run_id,
            status: TaskStatus::Initializing,
            call_id: Some(name.to_string()),
            attempt: 0,
        },
        TaskWrite::Started {
            name: name.to_string(),
            started_at: Utc::now(),
        },
        TaskWrite::Log {
            name: name.to_string(),
            source: LogSource::Stdout,
            chunk: b"line one\n".to_vec(),
        },
        TaskWrite::Log {
            name: name.to_string(),
            source: LogSource::Stdout,
            chunk: b"line two\n".to_vec(),
        },
        TaskWrite::Completed {
            name: name.to_string(),
            exit_status: Some(0),
            completed_at: Utc::now(),
        },
    ])
    .await
    .context("failed to apply batched task writes")?;

    Ok(())
}
