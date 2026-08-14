"""Canonical directory layout for the CHR quotation pipeline."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PRIMARY_DIR = ROOT / "sources" / "primary"
SOURCE_METADATA = ROOT / "sources" / "metadata.csv"

SEGMENTATION_DIR = ROOT / "derived" / "segmentation"
CHAPTER_INDEX = SEGMENTATION_DIR / "chapter_index.json"

AUXILIARY_DIR = ROOT / "auxiliary"
VINELAND_AUX = AUXILIARY_DIR / "vineland"
ATD_AUX = AUXILIARY_DIR / "against_the_day"

EVALUATION_DIR = ROOT / "evaluation"
CANDIDATES_DIR = EVALUATION_DIR / "candidates"
ANNOTATIONS_DIR = EVALUATION_DIR / "annotations"
GOLD_DIR = EVALUATION_DIR / "gold"
SAMPLE_MANIFEST = EVALUATION_DIR / "sample_manifest.csv"
LEGACY_EVALUATION_DIR = EVALUATION_DIR / "legacy"
LEGACY_GOLD_DIR = LEGACY_EVALUATION_DIR / "gold"
LEGACY_CENSUS_DIR = LEGACY_EVALUATION_DIR / "gold_census"
LEGACY_CANDIDATES_DIR = LEGACY_EVALUATION_DIR / "candidates"

SOURCE_BACKUPS_DIR = ROOT / "legacy" / "source_backups"

PRIMARY_FILES = {
    **{f"P0{i}": PRIMARY_DIR / f"P0{i}.txt" for i in range(1, 10)},
    "P10": PRIMARY_DIR / "P10_ST.txt",
}


def ensure_generated_dirs():
    """Create directories written by normal pipeline runs."""
    for path in (SEGMENTATION_DIR, CANDIDATES_DIR, ANNOTATIONS_DIR, GOLD_DIR):
        path.mkdir(parents=True, exist_ok=True)
