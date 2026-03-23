from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "deposit"


def load_dataset() -> pd.DataFrame:
    """Load bank marketing dataset using KaggleHub with local fallbacks."""
    df = None

    try:
        import kagglehub

        path = kagglehub.dataset_download("janiobachmann/bank-marketing-dataset")
        candidates = sorted(Path(path).rglob("*.csv"))
        if candidates:
            df = pd.read_csv(candidates[0])
    except Exception:
        pass

    if df is None:
        fallback_paths = [
            Path("bank.csv"),
            Path.cwd() / "bank.csv",
            Path("data/raw/bank.csv"),
            Path("C:/Users/kk/Downloads/bank.csv"),
        ]
        for p in fallback_paths:
            if p.exists():
                df = pd.read_csv(p)
                break

    if df is None:
        raise FileNotFoundError(
            "Could not load dataset. Install kagglehub with internet access or place bank.csv locally."
        )

    return df


def split_xy(df: pd.DataFrame, drop_cols=None) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = drop_cols or []
    y = (df[TARGET_COL] == "yes").astype(int)
    X = df.drop(columns=[TARGET_COL] + drop_cols, errors="ignore").copy()
    return X, y


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    num_transform = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", num_transform, num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
