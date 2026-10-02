from typing import List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from config.hyperparams import (
    FOLD_DATASET_ID,
    LANGUAGES,
    MAX_TRAIN_SAMPLES,
    MAX_VAL_SAMPLES,
    N_FOLDS,
    TRAIN_DATASET_ID,
)
from data.normalizer import normalize_record


def load_train_val(
    dataset_id: str = TRAIN_DATASET_ID,
    languages: List[str] = LANGUAGES,
    max_train: int = MAX_TRAIN_SAMPLES,
    max_val: int = MAX_VAL_SAMPLES,
    seed: int = 42,
):
    """Load and split the training dataset from HuggingFace.

    Returns:
        (train_records, val_records) — normalized record dicts.
    """
    ds = load_dataset(dataset_id)

    # Use existing split or create one
    if "train" in ds:
        split = ds["train"].train_test_split(test_size=0.1, seed=seed)
        raw_train = split["train"]
        raw_val = split["test"]
    else:
        key = list(ds.keys())[0]
        split = ds[key].train_test_split(test_size=0.1, seed=seed)
        raw_train = split["train"]
        raw_val = split["test"]

    # Language filter (only if the dataset actually carries a `language` column,
    # otherwise the filter would drop every row for datasets without it).
    if languages:
        if "language" in raw_train.column_names:
            raw_train = raw_train.filter(lambda x: x.get("language") in languages)
        if "language" in raw_val.column_names:
            raw_val = raw_val.filter(lambda x: x.get("language") in languages)

    # Cap sizes
    if max_train and len(raw_train) > max_train:
        raw_train = raw_train.select(range(max_train))
    if max_val and len(raw_val) > max_val:
        raw_val = raw_val.select(range(max_val))

    train_records = [normalize_record(x) for x in tqdm(raw_train, desc="normalize train")]
    val_records = [normalize_record(x) for x in tqdm(raw_val, desc="normalize val")]
    return train_records, val_records


def load_fold_split(
    fold_id: int,
    dataset_id: str = FOLD_DATASET_ID,
    languages: List[str] = LANGUAGES,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Load fold_k_train / fold_k_test splits for k-fold cross-validation.

    val_fraction of fold_k_train is carved out as an in-fold validation set
    (no data leakage from the test split).

    Args:
        fold_id: Fold index (0 to N_FOLDS-1).
        dataset_id: HuggingFace dataset repo with fold_k_train/fold_k_test splits.
        languages: Language codes to keep (None = keep all).
        val_fraction: Fraction of fold_k_train reserved for validation.
        seed: Random seed for the train/val split.

    Returns:
        (train_records, val_records, test_records) — normalized record dicts.
    """
    if not (0 <= fold_id < N_FOLDS):
        raise ValueError(f"fold_id must be in [0, {N_FOLDS - 1}], got {fold_id}")

    train_split = f"fold_{fold_id}_train"
    test_split  = f"fold_{fold_id}_test"

    raw_train_full = load_dataset(dataset_id, split=train_split)
    raw_test       = load_dataset(dataset_id, split=test_split)

    if languages:
        raw_train_full = raw_train_full.filter(lambda x: x.get("language") in languages)
        raw_test       = raw_test.filter(lambda x: x.get("language") in languages)

    # Carve out validation set from training split
    tv = raw_train_full.train_test_split(test_size=val_fraction, seed=seed)
    raw_train = tv["train"]
    raw_val   = tv["test"]

    train_records = [normalize_record(x) for x in tqdm(raw_train, desc=f"normalize fold_{fold_id}_train")]
    val_records   = [normalize_record(x) for x in tqdm(raw_val,   desc=f"normalize fold_{fold_id}_val")]
    test_records  = [normalize_record(x) for x in tqdm(raw_test,  desc=f"normalize fold_{fold_id}_test")]

    return train_records, val_records, test_records
