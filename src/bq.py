"""SQL -> Parquet, cached in the bucket.

* BigQuery: `EXPORT DATA` writes the query result straight into your GCS bucket as Parquet. That is much faster
  than pulling millions of rows through the API, and the result stays cached there: a re-run (or a new Kaggle
  session) skips BigQuery completely and only downloads the files.
* Tests: the same BigQuery SQL is transpiled to DuckDB (sqlglot) and run on small synthetic tables.

SQL uses placeholders such as `{hosp}` / `{icu}` / `{derived}` / `{eicu}` for the PhysioNet datasets.
Only numeric / text columns are exported (times are turned into hours inside SQL), so Parquet types are simple.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class BigQuerySource:
    kind = "bigquery"

    def __init__(self, cfg, store, datasets: dict):
        """datasets: {placeholder: (override, [candidate dataset names], probe_table)}"""
        from .gcp import bq_client, resolve_dataset
        self.client = bq_client(cfg.project_id)
        self.store = store
        self.ds = {k: resolve_dataset(self.client, o, c, p) for k, (o, c, p) in datasets.items()}
        for k, v in self.ds.items():
            log.info("{%s} -> %s", k, v)

    def fmt(self, sql: str) -> str:
        return sql.format(**{k: f"`{v}`" for k, v in self.ds.items()})

    def export(self, sql: str, key: str):
        from .store import GCSStore
        sql = self.fmt(sql)
        if isinstance(self.store, GCSStore):
            uri = self.store.uri(f"{key}/part-*.parquet")
            q = f"EXPORT DATA OPTIONS(uri='{uri}', format='PARQUET', compression='SNAPPY', overwrite=true) AS\n{sql}"
            try:
                job = self.client.query(q)
                job.result()
                log.info("BigQuery processed %.2f GB -> %s", (job.total_bytes_processed or 0) / 1e9, uri)
                return
            except Exception as e:                     # e.g. the token may not write to GCS -> stream instead
                log.warning("EXPORT DATA failed (%s). Streaming the result through the Storage API instead.",
                            str(e)[:200])
        self._stream(sql, key)

    def _stream(self, sql: str, key: str, rows_per_file: int = 2_000_000):
        """Run the query and write the result in Parquet parts (Storage Read API if allowed, else REST pages)."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        job = self.client.query(sql)
        it = job.result(page_size=100_000)
        log.info("BigQuery processed %.2f GB, %s rows", (job.total_bytes_processed or 0) / 1e9, it.total_rows)
        bqs = None
        try:
            from google.cloud import bigquery_storage
            from .gcp import credentials
            bqs = bigquery_storage.BigQueryReadClient(credentials=credentials("bigquery"))
        except Exception:
            pass
        tmp = Path("/tmp") / key.replace("/", "_")
        tmp.mkdir(parents=True, exist_ok=True)
        buf, n, part = [], 0, 0

        def flush():
            nonlocal buf, n, part
            if buf:
                f = tmp / f"part-{part:05d}.parquet"
                pq.write_table(pa.concat_tables(buf, promote_options="default"), f, compression="snappy")
                self.store.put(f, f"{key}/{f.name}")
                f.unlink()
                buf, n, part = [], 0, part + 1
        try:
            batches = it.to_arrow_iterable(bqstorage_client=bqs)
        except TypeError:
            batches = it.to_arrow_iterable()
        for b in batches:
            t = b if isinstance(b, pa.Table) else pa.Table.from_batches([b])
            buf.append(t)
            n += t.num_rows
            if n >= rows_per_file:
                flush()
        flush()
        if part == 0:                                   # empty result: still write one (empty) file
            f = tmp / "part-00000.parquet"
            it.to_arrow().to_pandas().to_parquet(f, index=False)
            self.store.put(f, f"{key}/{f.name}")

    def df(self, sql: str) -> pd.DataFrame:
        from .gcp import query_df
        return query_df(self.client, self.fmt(sql))


class DuckDBSource:
    """Runs the BigQuery SQL on DuckDB (unit tests / CI with synthetic tables)."""
    kind = "duckdb"

    def __init__(self, con, schemas: dict, store):
        self.con, self.ds, self.store = con, schemas, store

    def fmt(self, sql: str) -> str:
        import sqlglot
        return sqlglot.transpile(sql.format(**self.ds), read="bigquery", write="duckdb")[0]

    def export(self, sql: str, key: str):
        tmp = Path(self.store.uri(f"{key}/part-000.parquet"))
        tmp.parent.mkdir(parents=True, exist_ok=True)
        self.con.execute(f"COPY ({self.fmt(sql)}) TO '{tmp.as_posix()}' (FORMAT PARQUET)")

    def df(self, sql: str) -> pd.DataFrame:
        return self.con.execute(self.fmt(sql)).df()


def materialize(src, store, name: str, sql: str, cache_dir) -> list[Path]:
    """Run `sql` once (result cached in the bucket under data/<name>-<hash>/) and return local Parquet paths."""
    key = f"data/{name}-{hashlib.sha1(src.fmt(sql).encode()).hexdigest()[:10]}"
    if store.exists(f"{key}/_SUCCESS"):
        log.info("[%s] cached in bucket (%s) -> BigQuery skipped", name, store.uri(key))
    else:
        t = time.time()
        src.export(sql, key)
        store.put_text(json.dumps({"seconds": round(time.time() - t, 1)}), f"{key}/_SUCCESS")
        log.info("[%s] query done in %.0f s", name, time.time() - t)
    parts = sorted(k for k in store.list(key) if k.endswith(".parquet"))
    local = Path(cache_dir) / key
    local.mkdir(parents=True, exist_ok=True)

    def fetch(k):
        dst = local / Path(k).name
        if not dst.exists():
            store.get(k, dst)
        return dst
    with ThreadPoolExecutor(8) as ex:
        paths = list(ex.map(fetch, parts))
    log.info("[%s] %d parquet file(s) ready in %s", name, len(paths), local)
    return paths


def read(paths, columns=None) -> pd.DataFrame:
    import pyarrow.parquet as pq
    import pyarrow as pa
    tabs = [pq.read_table(p, columns=columns) for p in paths]
    tabs = [t for t in tabs if t.num_rows] or tabs[:1]
    return pa.concat_tables(tabs, promote_options="default").to_pandas() if tabs else pd.DataFrame()


def n_rows(paths) -> int:
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
