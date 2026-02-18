from pathlib import Path

import pandas as pd

from src.io import read_csv, read_parquet, write_parquet


def _sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


class TestWriteParquet:
    def test_creates_file(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        write_parquet(_sample_df(), out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "nested" / "deep" / "out.parquet"
        write_parquet(_sample_df(), out)
        assert out.exists()

    def test_preserves_columns_and_rows(self, tmp_path: Path):
        df = _sample_df()
        out = tmp_path / "out.parquet"
        write_parquet(df, out)
        loaded = pd.read_parquet(out)
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)


class TestReadParquet:
    def test_roundtrip_equals_original(self, tmp_path: Path):
        df = _sample_df()
        out = tmp_path / "rt.parquet"
        write_parquet(df, out)
        loaded = read_parquet(out)
        pd.testing.assert_frame_equal(loaded, df)

    def test_preserves_dtypes(self, tmp_path: Path):
        df = pd.DataFrame({"i": [1, 2], "f": [1.5, 2.5], "s": ["a", "b"]})
        out = tmp_path / "types.parquet"
        write_parquet(df, out)
        loaded = read_parquet(out)
        assert loaded["i"].dtype == df["i"].dtype
        assert loaded["f"].dtype == df["f"].dtype


class TestReadCsv:
    def test_reads_csv_correctly(self, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        df = _sample_df()
        df.to_csv(csv_path, index=False)
        loaded = read_csv(csv_path)
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)

    def test_preserves_values(self, tmp_path: Path):
        csv_path = tmp_path / "vals.csv"
        df = pd.DataFrame({"x": [10, 20, 30]})
        df.to_csv(csv_path, index=False)
        loaded = read_csv(csv_path)
        assert loaded["x"].tolist() == [10, 20, 30]
