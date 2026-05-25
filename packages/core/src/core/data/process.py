import pandas as pd
import os

def preprocess(df: pd.DataFrame, out_dir: str, fit_encoders: bool = True) -> pd.DataFrame:
    df['station_id'] = df['station_id'].str.strip()
    
    df["distance_km"] = (
        df.groupby('station_id')['distance_km']
        .transform('first')
    )

    df["distance_km"] = df["distance_km"].fillna(
        df["distance_km"].median()
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    
    import numpy as np

    # Month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Day of year
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    
    from sklearn.preprocessing import LabelEncoder
    import joblib
    
    categorical_cols = ["river", "province", "station_id"]

    os.makedirs(out_dir, exist_ok=True)
    for col in categorical_cols:
        if fit_encoders:
            le = LabelEncoder()

            df[col + "_encoded"] = le.fit_transform(df[col])
            joblib.dump(le, f"{out_dir}/le_{col}.pkl")
        else:
            le = joblib.load(f"{out_dir}/le_{col}.pkl")
            df[col + "_encoded"] = le.transform(df[col])
    
    drop_cols = [
        "river",
        "province",
        "month",
        "day_of_year",
        "station_id",
    ]

    df = df.drop(columns=drop_cols, errors='ignore')
    return df