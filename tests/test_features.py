from __future__ import annotations

import pandas as pd

from src.features import FeatureEngineer, FeatureSpec, build_preprocessor, infer_feature_spec


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lead_time": [10, 20, 30, 40],
            "country": ["TR", "US", "TR", "DE"],
            "arrival_date_month": ["January", "feb", " 3 ", "unknown"],
            "hotel": ["City", "Resort", "City", "City"],
            "target": [0, 1, 0, 1],
        }
    )


def test_feature_engineer_fit_transform_and_feature_names():
    df = _df().drop(columns=["target"])
    fe = FeatureEngineer()

    fitted = fe.fit(df)
    assert fitted is fe
    assert "country" in fe._freq_maps
    assert "arrival_date_month_sin" in fe.get_feature_names_out()
    assert "arrival_date_month_cos" in fe.get_feature_names_out()

    out = fe.transform(df)
    assert "arrival_date_month" not in out.columns
    assert "arrival_date_month_sin" in out.columns
    assert "arrival_date_month_cos" in out.columns
    assert pd.api.types.is_float_dtype(out["country"])

    # unseen country should map to 0.0
    out2 = fe.transform(
        pd.DataFrame(
            {
                "lead_time": [1],
                "country": ["ZZ"],
                "arrival_date_month": ["12"],
                "hotel": ["City"],
            }
        )
    )
    assert float(out2.loc[0, "country"]) == 0.0


def test_feature_engineer_feature_names_before_fit_returns_none():
    fe = FeatureEngineer()
    assert fe.get_feature_names_out() is None


def test_feature_spec_roundtrip_and_infer():
    df = _df()
    spec = infer_feature_spec(df, target_col="target")
    assert "lead_time" in spec.numeric
    assert "hotel" in spec.categorical

    payload = spec.to_dict()
    restored = FeatureSpec.from_dict(payload)
    assert restored.numeric == spec.numeric
    assert restored.categorical == spec.categorical
    assert "lead_time" in restored.all_features


def test_build_preprocessor_handles_fe_columns():
    spec = FeatureSpec(
        numeric=["lead_time"],
        categorical=["arrival_date_month", "country", "hotel"],
    )
    pipe = build_preprocessor(spec)

    ct = pipe.named_steps["column_transformer"]
    num_cols = ct.transformers[0][2]
    cat_cols = ct.transformers[1][2]

    assert "arrival_date_month_sin" in num_cols
    assert "arrival_date_month_cos" in num_cols
    assert "country" in num_cols
    assert "hotel" in cat_cols
    assert "country" not in cat_cols

    X = pd.DataFrame(
        {
            "lead_time": [10, 20, 30],
            "arrival_date_month": ["January", "July", "March"],
            "country": ["TR", "US", "FR"],
            "hotel": ["City", "Resort", "City"],
        }
    )
    arr = pipe.fit_transform(X)
    assert arr.shape[0] == 3
