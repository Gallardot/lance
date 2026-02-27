# Environment Variables

This guide documents the `LANCE_*` environment variables that affect runtime behavior in production workloads.

## How to set environment variables

Set variables before starting your application process.

```bash
export LANCE_CPU_THREADS=8
export LANCE_IO_THREADS=64
python your_app.py
```

You can also set variables inline for one command:

```bash
LANCE_CPU_THREADS=8 LANCE_IO_THREADS=64 python your_app.py
```

## Execution and scan behavior

| Variable | Default | Description | Typical use |
|---|---|---|---|
| `LANCE_CPU_THREADS` | Auto-detected | Number of compute-intensive worker threads. | Limit CPU usage per process, or increase parallelism. |
| `LANCE_IO_CORE_RESERVATION` | `2` | Number of CPU cores reserved for I/O. | Tune balance between I/O and compute threads. |
| `LANCE_DEFAULT_BATCH_SIZE` | Dynamic, minimum `8192` | Default scan batch size in rows. | Reduce memory pressure for wide rows; increase throughput for narrow rows. |
| `LANCE_DEFAULT_FRAGMENT_READAHEAD` | Dynamic (unset: typically `io_parallelism * 2` for v2 scans; legacy paths use `4`) | Fragment read-ahead depth for scans. | Increase for high-latency storage, reduce for constrained memory. |
| `LANCE_DEFAULT_IO_BUFFER_SIZE` | `2147483648` (2 GiB) | Scan I/O buffer size in bytes. | Increase when using many I/O threads or large pages. |
| `LANCE_XTR_OVERFETCH` | `10` | Over-fetch factor for multivector ANN search. | Trade compute cost for recall. |
| `LANCE_MEM_POOL_SIZE` | `104857600` (100 MiB) | DataFusion spill memory pool size in bytes. | Increase for larger in-memory query working sets. |
| `LANCE_MAX_TEMP_DIRECTORY_SIZE` | `107374182400` (100 GiB) | Max spill temp directory size in bytes. | Cap temporary disk usage during heavy queries. |
| `LANCE_BYPASS_SPILLING` | Unset | If set, disables spilling. | Force in-memory execution (use with caution). |
| `LANCE_SESSION_CACHE_SIZE` | `4` | Maximum cached query session contexts. | Increase for repeated queries with similar options. |
| `LANCE_AUTO_MIGRATION` | `true` | Auto-migrate outdated metadata when needed. | Set `false` to disable automatic migration behavior. |

## I/O and object store behavior

| Variable | Default | Description | Typical use |
|---|---|---|---|
| `LANCE_MAX_IOP_SIZE` | `16777216` (16 MiB) | Max size of a single I/O operation in bytes. | Tune read chunking strategy. |
| `LANCE_IO_THREADS` | Store-specific default | Number of object store I/O threads. | Increase for cloud bandwidth saturation. |
| `LANCE_PROCESS_IO_THREADS_LIMIT` | `128` | Global process-level IOPS concurrency cap. Set `<=0` to disable. | Prevent cross-query I/O contention, or remove cap for peak throughput. |
| `LANCE_USE_LITE_SCHEDULER` | Unset | If set, enables lite I/O scheduler. | Compare scheduler behavior in your workload. |
| `LANCE_UPLOAD_CONCURRENCY` | `10` | Multipart upload concurrency. | Increase write throughput to remote object stores. |
| `LANCE_CONN_RESET_RETRIES` | `20` | Retry limit for connection reset during upload. | Improve robustness on unstable networks. |
| `LANCE_INITIAL_UPLOAD_SIZE` | `5242880` (5 MiB) | Initial multipart upload part size. Valid range: `5 MiB` to `5 GiB`. | Tune upload behavior for large object writes. |

## Encoding and file behavior

| Variable | Default | Description | Typical use |
|---|---|---|---|
| `LANCE_FILE_WRITER_MAX_PAGE_BYTES` | `33554432` (32 MiB) | Max page size in file writer. | Tune write/read trade-off and file layout. |
| `LANCE_STRUCTURAL_BATCH_DECODE_SPAWN_MODE` | Heuristic / mode-dependent when unset | Structural decode task mode: `always`, `never`, or internal default behavior when unset. | Tune decode parallelism strategy. |
| `LANCE_DICT_ENCODING_THRESHOLD` | `100` | Legacy dictionary encoding threshold. | Tune dictionary encoding aggressiveness on older paths. |
| `LANCE_ENCODING_DICT_TOO_SMALL` | `100` | Minimum value count before dictionary encoding is considered. | Avoid dictionary overhead on small arrays. |
| `LANCE_ENCODING_DICT_DIVISOR` | `2` | Divisor used in dictionary cardinality thresholding. | Tune cardinality heuristic. |
| `LANCE_ENCODING_DICT_MAX_CARDINALITY` | `100000` | Upper bound for dictionary cardinality thresholding. | Cap dictionary estimate complexity. |
| `LANCE_ENCODING_DICT_SIZE_RATIO` | `0.8` | Maximum encoded size ratio for dictionary encoding. | Control compression-vs-CPU trade-off. |
| `LANCE_BINARY_MINIBLOCK_CHUNK_SIZE` | `4096` | Chunk size for binary miniblock encoding. | Tune binary encoding characteristics. |

## Index and search behavior

| Variable | Default | Description | Typical use |
|---|---|---|---|
| `LANCE_USE_HNSW_SPEEDUP_INDEXING` | `auto` | Controls HNSW-assisted index build path (`enabled`, `disabled`, `auto`). | Force or disable acceleration for benchmarked workloads. |
| `LANCE_FTS_NUM_SHARDS` | Compute-thread count | Number of FTS build shards. | Increase FTS build parallelism. |
| `LANCE_FTS_PARTITION_SIZE` | `256` MiB | FTS partition size threshold. | Control memory/merge behavior in FTS build. |
| `LANCE_FTS_TARGET_SIZE` | `4096` MiB | Target partition size after FTS merge. | Tune final partition layout. |
| `LANCE_LANGUAGE_MODEL_HOME` | Platform local data dir fallback | Directory for language model assets used by tokenization. | Point to a managed model directory. |
| `LANCE_FLAT_SEARCH_PERCENT_THRESHOLD` | `10` | Threshold for switching flat search behavior in inverted search. | Tune latency/quality behavior. |
| `LANCE_NGRAM_TOKENS_PER_SPILL` | `1000000000` | N-gram spill threshold. | Reduce peak memory by spilling earlier. |
| `LANCE_NGRAM_NUM_PARTITIONS` | `max(cpu * 4, 128)` | N-gram build partition count. | Tune parallel build and memory profile. |
| `LANCE_NGRAM_TOKENIZE_PARALLELISM` | `8` | Tokenization parallelism for N-gram build. | Increase tokenizer throughput. |
| `LANCE_ZONEMAP_DEFAULT_ROWS_PER_ZONE` | `8192` | Default rows per zone for zonemap index. | Tune zonemap granularity. |
| `LANCE_BLOOMFILTER_DEFAULT_NUMBER_OF_ITEMS` | `8192` | Default expected item count for Bloom filter index. | Tune Bloom filter sizing. |
| `LANCE_BLOOMFILTER_DEFAULT_PROBABILITY` | `0.00057` | Default Bloom filter false-positive probability. | Tune space-vs-false-positive trade-off. |
