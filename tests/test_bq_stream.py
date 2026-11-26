"""BigQuery fallback path (no EXPORT DATA): results are streamed into Parquet parts in the store."""
import pyarrow as pa

from src.bq import BigQuerySource, materialize, read
from src.store import LocalStore


class _Iter:
    total_rows = 25

    def to_arrow_iterable(self, bqstorage_client=None):
        for s in range(0, 25, 10):
            yield pa.record_batch({"a": list(range(s, min(s + 10, 25)))})

    def to_arrow(self):
        return pa.table({"a": pa.array([], pa.int64())})


class _Job:
    total_bytes_processed = 0

    def result(self, page_size=None):
        return _Iter()


class _Client:
    def query(self, sql):
        return _Job()


def test_stream_fallback(tmp_path):
    store = LocalStore(tmp_path / "bucket", "p")
    src = object.__new__(BigQuerySource)
    src.client, src.store, src.ds = _Client(), store, {}
    src._stream("SELECT 1", "data/x", rows_per_file=10)
    parts = sorted(k for k in store.list("data/x") if k.endswith(".parquet"))
    assert len(parts) == 3
    paths = materialize(src, store, "y", "SELECT 1", tmp_path / "cache")      # LocalStore -> streaming path
    assert read(paths).a.tolist() == list(range(25))
