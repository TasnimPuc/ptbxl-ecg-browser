"""Builds notebooks/kaggle_run.ipynb from the repo (self-contained). Run: python tools/build_notebook.py
Texts specific to this project live in tools/nb_config.py; the step cells come from src/pipeline.py."""
from pathlib import Path

import sys

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbsteps import cells as step_cells  # noqa: E402
import nb_config as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INSTALL = """!pip install -q google-cloud-bigquery google-cloud-bigquery-storage google-cloud-storage db-dtypes pyarrow 2>&1 | tail -1
import torch, platform
print("python", platform.python_version(), "| torch", torch.__version__, "| GPU:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")"""

CHECK = """from src.gcp import check_access
check_access(cfg.project_id, probes={probes})"""

SHOW = """import re, shutil, glob
from IPython.display import Markdown, Image, display
rep = f"{{cfg.work_dir}}/report"
text = open(f"{{rep}}/REPORT.md", encoding="utf-8").read()
display(Markdown(re.sub(r"!\\[.*?\\]\\(figures/.*?\\)", "", text)))
for f in sorted(glob.glob(f"{{rep}}/figures/*.png")):
    display(Image(f))
shutil.make_archive("/kaggle/working/results_for_github", "zip", rep)
print("Results zip : /kaggle/working/results_for_github.zip  -> unzip into the repo as results/")
print("Models (.pth, private — keep out of GitHub):")
for f in sorted(glob.glob(f"{{cfg.work_dir}}/model/*.pth")):
    print("  ", f)
print("Bucket copy :", "gs://" + cfg.bucket + "/" + cfg.gcs_prefix + "/runs/" + cfg.run_name + "/model/")"""


def main():
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(C.INTRO), md("## 1 · Setup\nBigQuery/GCS clients (already on Kaggle; this only fills gaps) and GPU check."),
             code(INSTALL.replace("db-dtypes pyarrow", "db-dtypes pyarrow " + getattr(C, "EXTRA_PIP", "")).replace("  ", " ")),
             md("## 2 · Project code\nThese cells write the package to `/kaggle/working/repo`. "
                "No need to edit them; the steps further down call this code."),
             code("!mkdir -p /kaggle/working/repo/src")]
    for f in sorted((ROOT / "src").glob("*.py")):
        rel = f.relative_to(ROOT).as_posix()
        cells.append(code(f"%%writefile /kaggle/working/repo/{rel}\n" + f.read_text(encoding="utf-8")))
    cells += [md("## 3 · Config\nChange only this cell."), code(C.CONFIG),
              md(getattr(C, "CHECK_MD", "## 4 · Pre-flight check\nProject ID, Kaggle → Google Cloud add-on (BigQuery + "
                                       "Cloud Storage), billing and PhysioNet access. Costs nothing (dry runs).")),
              code(getattr(C, "CHECK_CODE", None) or CHECK.format(probes=repr(getattr(C, "PROBES", []))))]
    steps, n = step_cells(ROOT / "src" / "pipeline.py", 5)
    cells += steps + [md(f"## {n} · Results report\n"
                         + getattr(C, "SHOW_NOTE", "Aggregate results only (safe for GitHub). Model weights stay in `/kaggle/working` and your private bucket.")),
                      code(getattr(C, "SHOW_CODE", None) or SHOW.format())]
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "kaggle": {"accelerator": C.ACCELERATOR, "isInternetEnabled": True,
                                 "isGpuEnabled": C.ACCELERATOR != "none"}}
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, ROOT / "notebooks" / "kaggle_run.ipynb")
    print("wrote notebooks/kaggle_run.ipynb with", len(cells), "cells")


if __name__ == "__main__":
    main()
