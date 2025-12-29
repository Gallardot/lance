#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def build_query(query: str, columns: str | None) -> dict[str, object]:
    full_text_query: dict[str, object] = {"query": query}
    if columns:
        full_text_query["columns"] = [col.strip() for col in columns.split(",") if col.strip()]
    return full_text_query


def ensure_local_lance_on_path() -> None:
    script_dir = Path(__file__).resolve().parent
    python_root = script_dir.parent
    sys.path.insert(0, str(python_root))


def run_single(args: argparse.Namespace) -> int:
    ensure_local_lance_on_path()
    import lance

    session_kwargs = {}
    if args.index_cache_bytes is not None:
        session_kwargs["index_cache_size_bytes"] = args.index_cache_bytes
    session = lance.Session(**session_kwargs)
    dataset = lance.LanceDataset(args.uri, session=session)

    query = build_query(args.query, args.columns)
    scanner = dataset.scan(full_text_query=query, limit=args.limit)
    table = scanner.to_table()

    result = {
        "prefix": int(os.environ.get("LANCE_FTS_PREFETCH_PREFIX_NUM", "0")),
        "top": int(os.environ.get("LANCE_FTS_PREFETCH_TOP_NUM", "0")),
        "rows": table.num_rows,
        "index_cache_entries": dataset.index_cache_entry_count(),
        "index_cache_hit_rate": dataset.index_cache_hit_rate(),
    }
    print(json.dumps(result))
    return 0


def parse_json_output(stdout: str) -> dict[str, object]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("no output from child process")
    return json.loads(lines[-1])


def run_matrix(args: argparse.Namespace) -> int:
    prefixes = parse_int_list(args.prefix)
    tops = parse_int_list(args.top)
    if not prefixes or not tops:
        raise ValueError("prefix and top lists must be non-empty")

    results: list[dict[str, object]] = []
    for prefix, top in itertools.product(prefixes, tops):
        env = os.environ.copy()
        env["LANCE_FTS_PREFETCH_PREFIX_NUM"] = str(prefix)
        env["LANCE_FTS_PREFETCH_TOP_NUM"] = str(top)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            "--uri",
            args.uri,
            "--query",
            args.query,
            "--limit",
            str(args.limit),
        ]
        if args.columns:
            cmd.extend(["--columns", args.columns])
        if args.index_cache_bytes is not None:
            cmd.extend(["--index-cache-bytes", str(args.index_cache_bytes)])

        proc = subprocess.run(
            cmd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise RuntimeError(
                f"child process failed for prefix={prefix} top={top} (exit {proc.returncode})"
            )

        result = parse_json_output(proc.stdout)
        results.append(result)
        print(
            "prefix={prefix} top={top} hit_rate={hit_rate:.4f} entries={entries} rows={rows}".format(
                prefix=result["prefix"],
                top=result["top"],
                hit_rate=result["index_cache_hit_rate"],
                entries=result["index_cache_entries"],
                rows=result["rows"],
            )
        )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full text search with different prefetch settings and report index cache hit rate."
        )
    )
    parser.add_argument("--uri", required=True, help="Dataset URI to query.")
    parser.add_argument("--query", required=True, help="Full text search query.")
    parser.add_argument("--columns", help="Comma-separated columns for full text search.")
    parser.add_argument("--limit", type=int, default=100, help="Limit for the full text search.")
    parser.add_argument(
        "--prefix",
        default="16",
        help="Comma-separated prefix prefetch sizes to test.",
    )
    parser.add_argument(
        "--top",
        default="32",
        help="Comma-separated top-score prefetch sizes to test.",
    )
    parser.add_argument(
        "--index-cache-bytes",
        type=int,
        help="Index cache size in bytes for the session.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON results.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.single:
        return run_single(args)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
