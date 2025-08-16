import pandas as pd
import utils.poisson_utils.data as data


def test_load_data_writes_and_reads_parquet(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HY,AY,HR,AR,HF,AF\n"
        "01/01/2024,A,B,1,0,H,5,3,2,1,4,2,1,0,0,0,3,5\n"
    )

    df1 = data.load_data(str(csv_path))
    parquet_path = csv_path.with_suffix(".parquet")
    assert parquet_path.exists()

    csv_path.unlink()
    df2 = data.load_data(str(csv_path))

    pd.testing.assert_frame_equal(df1, df2)
