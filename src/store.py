"""Checkpoint store: a private GCS bucket (auto-created) or a local folder (tests / no GCP).
Everything needed to resume after a Kaggle session stops lives here."""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def default_bucket_name(project_id: str) -> str:
    name = re.sub(r"[^a-z0-9-]", "-", f"{project_id}-mimic-ckpt".lower()).strip("-")
    return name[:63]


class GCSStore:
    def __init__(self, project_id: str, bucket: str, prefix: str, location: str = "US"):
        from google.cloud import storage

        from .gcp import credentials
        self.client = storage.Client(project=project_id, credentials=credentials("gcs"))
        self.name = (bucket or default_bucket_name(project_id)).replace("gs://", "").strip("/")
        self.prefix = prefix.strip("/")
        self.bucket = self._ensure(project_id, location)

    def _ensure(self, project_id, location):
        from google.api_core import exceptions as gexc
        b = self.client.bucket(self.name)
        try:
            if b.exists():
                return b
        except gexc.Forbidden:
            return b           # exists but owned elsewhere / no get permission; writes will tell
        try:
            b.iam_configuration.uniform_bucket_level_access_enabled = True
            b.iam_configuration.public_access_prevention = "enforced"   # never public
            b = self.client.create_bucket(b, project=project_id, location=location)
            log.info("Created private bucket gs://%s (%s)", self.name, location)
        except gexc.Conflict:
            raise RuntimeError(f"Bucket name '{self.name}' is taken globally. Set cfg.bucket to another name.")
        except Exception as e:
            raise RuntimeError(
                f"Could not create bucket gs://{self.name} ({e}). Create it once in Cloud Shell:\n"
                f"  gcloud storage buckets create gs://{self.name} --project={project_id} --location={location} "
                f"--uniform-bucket-level-access --public-access-prevention\nthen set cfg.bucket='{self.name}'.")
        return b

    def uri(self, key=""):
        return f"gs://{self.name}/{self.prefix}/{key}".rstrip("/")

    def _k(self, key):
        return f"{self.prefix}/{key}"

    def put(self, local, key):
        self.bucket.blob(self._k(key)).upload_from_filename(str(local), timeout=600)

    def get(self, key, local) -> bool:
        bl = self.bucket.blob(self._k(key))
        if not bl.exists():
            return False
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(local) + ".part")
        bl.download_to_filename(str(tmp), timeout=600)
        tmp.replace(local)                        # never leave a half-downloaded file behind
        return True

    def exists(self, key) -> bool:
        return self.bucket.blob(self._k(key)).exists()

    def list(self, sub) -> list[str]:
        p = self._k(sub).rstrip("/") + "/"
        return [b.name[len(self.prefix) + 1:] for b in self.client.list_blobs(self.bucket, prefix=p)]

    def put_text(self, text, key):
        self.bucket.blob(self._k(key)).upload_from_string(text)

    def get_text(self, key):
        bl = self.bucket.blob(self._k(key))
        return bl.download_as_text() if bl.exists() else None

    def put_dir(self, local_dir, sub):
        n = 0
        for p in sorted(Path(local_dir).rglob("*")):
            if p.is_file():
                self.put(p, f"{sub.strip('/')}/{p.relative_to(local_dir).as_posix()}")
                n += 1
        log.info("Uploaded %d files -> %s", n, self.uri(sub))


class LocalStore:
    """Same interface backed by a folder (unit tests, or running without GCP)."""

    def __init__(self, root, prefix=""):
        self.root = Path(root) / prefix
        self.root.mkdir(parents=True, exist_ok=True)
        self.name = str(self.root)

    def uri(self, key=""):
        return str(self.root / key)

    def put(self, local, key):
        dst = self.root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dst)

    def get(self, key, local) -> bool:
        src = self.root / key
        if not src.exists():
            return False
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        if Path(local).resolve() != src.resolve():
            shutil.copyfile(src, local)
        return True

    def exists(self, key) -> bool:
        return (self.root / key).exists()

    def list(self, sub) -> list[str]:
        d = self.root / sub
        return [p.relative_to(self.root).as_posix() for p in d.rglob("*") if p.is_file()] if d.exists() else []

    def put_text(self, text, key):
        dst = self.root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)

    def get_text(self, key):
        f = self.root / key
        return f.read_text() if f.exists() else None

    def put_dir(self, local_dir, sub):
        for p in Path(local_dir).rglob("*"):
            if p.is_file():
                self.put(p, f"{sub.strip('/')}/{p.relative_to(local_dir).as_posix()}")


def make_store(cfg):
    if getattr(cfg, "local_store", ""):
        return LocalStore(cfg.local_store, cfg.gcs_prefix)
    from .gcp import validate_project_id
    cfg.project_id = validate_project_id(cfg.project_id)
    s = GCSStore(cfg.project_id, cfg.bucket, cfg.gcs_prefix, cfg.bucket_location)
    cfg.bucket = s.name
    log.info("Checkpoints -> %s", s.uri())
    return s
