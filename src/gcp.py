"""BigQuery helpers. Works with Kaggle's "Google Cloud Services" add-on
(linked account) and with normal ADC credentials elsewhere."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)
SOURCE_PROJECT = "physionet-data"


def _kaggle_integrations() -> str:
    return os.environ.get("KAGGLE_KERNEL_INTEGRATIONS", "")


def credentials(target: str):
    """Credentials for 'bigquery' or 'gcs'.
    On Kaggle: the account attached in Add-ons -> Google Cloud Services (passed explicitly, so we never
    fall back to the GCE metadata server). Elsewhere: Application Default Credentials (None)."""
    if "KAGGLE_USER_SECRETS_TOKEN" not in os.environ:
        return None                                   # not on Kaggle -> ADC
    want = {"bigquery": "BIGQUERY", "gcs": "GCS"}[target]
    have = [x.upper() for x in _kaggle_integrations().split(":") if x]
    if want in have:
        from kaggle_secrets import GcpTarget
        try:
            from kaggle_gcp import KaggleKernelCredentials
        except ImportError:                           # minimal re-implementation
            from google.auth import credentials as gac
            from kaggle_secrets import UserSecretsClient

            class KaggleKernelCredentials(gac.Credentials):
                def __init__(self, target):
                    super().__init__()
                    self.target = target

                def refresh(self, request):
                    self.token, self.expiry = UserSecretsClient()._get_access_token(self.target)
        return KaggleKernelCredentials(target=GcpTarget[want])
    try:                                              # Add-ons -> "Google Cloud SDK" also works
        from google.oauth2.credentials import Credentials
        from kaggle_secrets import UserSecretsClient
        info = json.loads(UserSecretsClient().get_gcloud_credential())
        log.info("Using Google Cloud SDK add-on credential for %s", target)
        return Credentials.from_authorized_user_info(info)
    except Exception:
        pass
    raise RuntimeError(
        f"Kaggle has no {want} integration attached (KAGGLE_KERNEL_INTEGRATIONS='{_kaggle_integrations()}').\n"
        "Fix: Add-ons -> Google Cloud Services -> attach your Google account and tick BOTH 'BigQuery' and "
        "'Cloud Storage' -> Save -> then Run -> Restart & clear outputs -> Run All.")


def validate_project_id(project_id: str) -> str:
    pid = (project_id or "").strip()
    if not pid or "YOUR_" in pid.upper() or " " in pid:
        raise ValueError(
            f"cfg.project_id='{project_id}' is not a real GCP project ID.\n"
            "Find it: console.cloud.google.com -> project picker (top bar) -> copy the 'ID' column "
            "(e.g. 'mimic-research-123456', NOT the display name). Use the same Google account/project "
            "that is attached in Kaggle Add-ons and has billing/BigQuery enabled.")
    return pid


def check_access(project_id: str, probes=()):
    """Quick pre-flight check printed in the notebook."""
    try:
        validate_project_id(project_id)
        print("project_id:", project_id, "OK")
    except ValueError as e:
        print("project_id: FAILED ->", e)
        return
    print("Kaggle integrations:", _kaggle_integrations() or "(none)")
    for t in ("bigquery", "gcs"):
        try:
            c = credentials(t)
            if c is not None:
                from google.auth.transport.requests import Request
                c.refresh(Request())
            print(f"  {t:8s}: OK")
        except Exception as e:
            print(f"  {t:8s}: FAILED -> {e}")
            return
    try:
        client = bq_client(project_id)
        from google.cloud import bigquery
        client.query("SELECT 1", job_config=bigquery.QueryJobConfig(dry_run=True))
        print("  BigQuery project/billing: OK")
    except Exception as e:
        print("  BigQuery project/billing: FAILED ->", str(e)[:300])
        return
    for t in probes:
        ok = _dry_run_ok(client, f"SELECT 1 FROM `{SOURCE_PROJECT}.{t}` LIMIT 1")
        print(f"  PhysioNet {t}: {'OK' if ok else 'not readable (other version name is tried automatically)'}")


def bq_client(project_id: str):
    from google.cloud import bigquery
    project_id = validate_project_id(project_id)
    return bigquery.Client(project=project_id, credentials=credentials("bigquery"))


def _dry_run_ok(client, sql: str) -> bool:
    from google.api_core import exceptions as gexc
    from google.cloud import bigquery
    try:
        client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
        return True
    except (gexc.NotFound, gexc.Forbidden, gexc.BadRequest):
        return False            # dataset absent or no PhysioNet access -> try next candidate


def resolve_dataset(client, override: str, candidates: list[str], probe_table: str) -> str:
    """Return 'physionet-data.<dataset>' that the account can read (free dry-run probe)."""
    names = [override] if override else candidates
    for name in names:
        fq = name if "." in name else f"{SOURCE_PROJECT}.{name}"
        if _dry_run_ok(client, f"SELECT 1 FROM `{fq}.{probe_table}` LIMIT 1"):
            log.info("Using dataset %s", fq)
            return fq
    raise RuntimeError(
        f"None of {names} is readable (probe table '{probe_table}'). Open BigQuery console -> "
        f"'{SOURCE_PROJECT}' and set the dataset name in the config.")


def query_df(client, sql: str):
    job = client.query(sql)
    job.result()
    log.info("BigQuery processed %.2f GB", (job.total_bytes_processed or 0) / 1e9)
    try:
        return job.to_dataframe(create_bqstorage_client=True)
    except Exception as e:  # Storage Read API not permitted -> REST fallback
        log.warning("BQ Storage API unavailable (%s). Falling back to REST download.", type(e).__name__)
        return job.result().to_dataframe(create_bqstorage_client=False)
