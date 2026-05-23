
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

try:
	import preprocessing.config as cfg
except ModuleNotFoundError:
	# Allow running as a script from the repository root.
	pmsm_root = Path(__file__).resolve().parents[1]  # .../pmsm
	sys.path.insert(0, str(pmsm_root))
	import preprocessing.config as cfg


PROFILE_ID_COL = "profile_id"


def _resolve_dataset_path() -> Path:
	"""Resolve the dataset path independent of current working directory.

	The repo's training scripts expect `cfg.data_cfg['file_path']` to be
	relative to the `pmsm/` package directory.
	"""
	pmsm_root = Path(__file__).resolve().parents[1]  # .../pmsm
	configured = Path(cfg.data_cfg["file_path"])
	if configured.is_absolute():
		return configured
	return (pmsm_root / configured).resolve()


def main() -> None:
	dataset_path = _resolve_dataset_path()
	if not dataset_path.exists():
		raise FileNotFoundError(
			f"Dataset not found at {dataset_path}. "
			"Set preprocessing.config:data_cfg['file_path'] to your CSV."
		)

	df = pd.read_csv(dataset_path)

	n_rows = int(df.shape[0])
	if PROFILE_ID_COL not in df.columns:
		raise KeyError(
			f"Expected column '{PROFILE_ID_COL}' in dataset, got columns: {list(df.columns)}"
		)

	n_profiles = int(df[PROFILE_ID_COL].nunique(dropna=False))

	print(f"Dataset: {dataset_path}")
	print(f"Rows: {n_rows}")
	print(f"Profiles ({PROFILE_ID_COL} unique): {n_profiles}")


if __name__ == "__main__":
	main()

