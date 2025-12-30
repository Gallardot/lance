// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! UringReader implementation.

use super::future::UringReadFuture;
use super::requests::IoRequest;
use super::thread::{SUBMITTED_COUNTER, THREAD_SELECTOR, URING_THREADS};
use super::{DEFAULT_URING_BLOCK_SIZE, DEFAULT_URING_IO_PARALLELISM};
use crate::local::to_local_path;
use crate::traits::Reader;
use crate::uring::requests::RequestState;
use crate::utils::tracking_store::IOTracker;
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use object_store::path::Path;
use snafu::location;
use std::fs::File;
use std::future::Future;
use std::io::{self, ErrorKind};
use std::ops::Range;
use std::os::unix::io::{AsRawFd, RawFd};
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;
use tracing::instrument;

/// File handle for io_uring operations.
///
/// Keeps the file alive and provides the raw file descriptor.
#[derive(Debug)]
struct UringFileHandle {
    /// The file (kept alive via Arc)
    file: Arc<File>,

    /// Raw file descriptor for io_uring
    fd: RawFd,

    /// Object store path
    path: Path,
}

impl UringFileHandle {
    fn new(file: File, path: Path) -> Self {
        let fd = file.as_raw_fd();
        Self {
            file: Arc::new(file),
            fd,
            path,
        }
    }
}

/// io_uring-based reader for local files.
///
/// This reader uses a dedicated process-wide thread running an io_uring event loop
/// for high-performance asynchronous I/O.
#[derive(Debug)]
pub struct UringReader {
    /// File handle
    handle: Arc<UringFileHandle>,

    /// Block size for I/O operations
    block_size: usize,

    /// Cached file size
    size: OnceCell<usize>,

    /// I/O tracker for monitoring operations
    io_tracker: Arc<IOTracker>,
}

impl DeepSizeOf for UringReader {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        // Skip file handle (just a system resource)
        // Only count the path's deep size
        self.handle.path.as_ref().deep_size_of_children(context)
    }
}

impl UringReader {
    /// Open a file with io_uring.
    ///
    /// This is the internal constructor used by ObjectStore.
    #[instrument(level = "debug")]
    pub(crate) async fn open(
        path: &Path,
        block_size: usize,
        known_size: Option<usize>,
        io_tracker: Arc<IOTracker>,
    ) -> Result<Box<dyn Reader>> {
        let path = path.clone();
        let local_path = to_local_path(&path);

        // Open file in spawn_blocking (same pattern as LocalObjectReader)
        let handle = tokio::task::spawn_blocking(move || {
            let file = File::open(&local_path).map_err(|e| match e.kind() {
                ErrorKind::NotFound => Error::NotFound {
                    uri: path.to_string(),
                    location: location!(),
                },
                _ => e.into(),
            })?;

            Ok::<_, Error>(Arc::new(UringFileHandle::new(file, path)))
        })
        .await??;

        let size = OnceCell::new_with(known_size);

        // Determine block size (env var or default)
        let block_size = std::env::var("LANCE_URING_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(block_size.max(DEFAULT_URING_BLOCK_SIZE));

        Ok(Box::new(Self {
            handle,
            block_size,
            size,
            io_tracker,
        }))
    }

    /// Submit a read request to the io_uring thread via channel and return a future.
    fn submit_read(
        &self,
        offset: u64,
        length: usize,
    ) -> Pin<Box<dyn Future<Output = object_store::Result<Bytes>> + Send>> {
        let mut buffer = BytesMut::with_capacity(length);
        unsafe {
            buffer.set_len(length);
        }

        // Create IoRequest with all data
        let request = Arc::new(IoRequest {
            fd: self.handle.fd,
            offset,
            length,
            state: Mutex::new(RequestState {
                completed: false,
                waker: None,
                err: None,
                buffer,
            }),
        });

        // Increment submitted counter before sending to channel
        SUBMITTED_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Select thread in round-robin fashion
        let thread_idx =
            (THREAD_SELECTOR.fetch_add(1, Ordering::Relaxed) as usize) % URING_THREADS.len();

        // Send to selected thread via channel
        match URING_THREADS[thread_idx]
            .request_tx
            .send(Arc::clone(&request))
        {
            Ok(()) => {
                // Return future that will be woken when operation completes
                Box::pin(UringReadFuture { request })
            }
            Err(_) => {
                // Thread died - decrement counter and return error future
                SUBMITTED_COUNTER.fetch_sub(1, Ordering::Relaxed);
                Box::pin(async move {
                    Err(object_store::Error::Generic {
                        store: "UringReader",
                        source: Box::new(io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            "io_uring thread died",
                        )),
                    })
                })
            }
        }
    }
}

#[async_trait]
impl Reader for UringReader {
    fn path(&self) -> &Path {
        &self.handle.path
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn io_parallelism(&self) -> usize {
        std::env::var("LANCE_URING_IO_PARALLELISM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_URING_IO_PARALLELISM)
    }

    /// Returns the file size.
    async fn size(&self) -> object_store::Result<usize> {
        let file = self.handle.file.clone();
        self.size
            .get_or_try_init(|| async move {
                let metadata = tokio::task::spawn_blocking(move || {
                    file.metadata().map_err(|err| object_store::Error::Generic {
                        store: "UringReader",
                        source: Box::new(err),
                    })
                })
                .await??;
                Ok(metadata.len() as usize)
            })
            .await
            .cloned()
    }

    /// Read a range of bytes using io_uring.
    #[instrument(level = "debug", skip(self))]
    async fn get_range(&self, range: Range<usize>) -> object_store::Result<Bytes> {
        let io_tracker = self.io_tracker.clone();
        let path = self.handle.path.clone();
        let num_bytes = range.len() as u64;
        let range_u64 = (range.start as u64)..(range.end as u64);

        let result = self.submit_read(range.start as u64, range.len()).await;

        if result.is_ok() {
            io_tracker.record_read("get_range", path, num_bytes, Some(range_u64));
        }

        result
    }

    /// Read the entire file using io_uring.
    #[instrument(level = "debug", skip(self))]
    async fn get_all(&self) -> object_store::Result<Bytes> {
        let size = self.size().await?;
        let io_tracker = self.io_tracker.clone();
        let path = self.handle.path.clone();

        let result = self.submit_read(0, size).await;

        if let Ok(ref bytes) = result {
            io_tracker.record_read("get_all", path, bytes.len() as u64, None);
        }

        result
    }
}
