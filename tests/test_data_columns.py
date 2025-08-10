import pandas as pd
import utils.poisson_utils.data as data

def test_load_data_casts_numeric_columns(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HY,AY,HR,AR,HF,AF\n"
        "01/01/2024,A,B,1,0,H,5,3,2,1,4,2,1,0,0,0,3,5\n"
    )
    df = data.load_data(str(csv_path))
    numeric_columns = [
        "FTHG","FTAG","HS","AS","HST","AST","HC","AC","HY","AY","HR","AR","HF","AF"
    ]
    for col in numeric_columns:
        assert str(df[col].dtype) == "Int64"
