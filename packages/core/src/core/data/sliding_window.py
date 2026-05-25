import pandas as pd
import numpy as np

WINDOW_SIZE = 6
HORIZON     = 3

FEATURE_ORDER = [
    "salinity",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "river_encoded",
    "province_encoded",
    "station_id_encoded",
]

def sliding(
        df: pd.DataFrame, train_cutoff: int, model_type: str = "xgboost",
        window_size: int = WINDOW_SIZE, horizon: int = HORIZON
    ):
    
    X_train, y_train = [], []
    X_valid, y_valid = [], []
    
    grouped = df.groupby("station_id_encoded")

    for _, station_df in grouped:
        
        station_df = (
            station_df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        
        station_train_size = (
            station_df["timestamp"] <= train_cutoff
        ).sum()
        
        feature_values  = station_df[FEATURE_ORDER].values
        dist_values     = station_df["distance_km"].values
        salinity_values = station_df["salinity"].values
        
        total_samples = (
            len(station_df) 
            - window_size 
            - horizon 
            + 1
        )
        
        if total_samples <= 0:
            continue
        
        
        X_station, y_station = [], []
        for i in range(total_samples):
            start_idx = i
            end_idx = i + window_size
            target_idx = end_idx
            
            # ─────────────────────────────
            # Dynamic sequence window
            # Shape: (window_size, num_features)
            # ─────────────────────────────
            dynamic_window = feature_values[
                start_idx : end_idx
            ]

            distance = dist_values[target_idx]
            
            # ─────────────────────────────
            # Sequence features
            # Shape: Based on the chosen model
            # ─────────────────────────────
            x = _build_input(
                dynamic_window  = dynamic_window,
                distance        = distance,
                model_type      = model_type,
                window_size     = window_size
            )
            
            # ─────────────────────────────
            # Target
            # ─────────────────────────────
            y = salinity_values[
                target_idx:target_idx + horizon
            ]

            X_station.append(x)
            y_station.append(y)
        
        
        # ─────────────────────────────
        # Split train / valid
        # ─────────────────────────────
        split_idx = station_train_size - window_size
        
        X_train.extend(X_station[:split_idx])
        y_train.extend(y_station[:split_idx])

        X_valid.extend(X_station[split_idx:])
        y_valid.extend(y_station[split_idx:])

        
    # Convert to numpy
    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_valid),
        np.array(y_valid),
    )

    
def _build_input(dynamic_window: np.ndarray, distance: float, model_type: str, window_size: int):
    
    # ─────────────────────────────
    # XGBoost
    # Flatten sequence
    # ─────────────────────────────
            
    if model_type == "xgboost":
        return np.concatenate([
            dynamic_window.flatten(),
            np.array([distance]),
        ])
    
    # ─────────────────────────────
    # LSTM or iTransformer
    # Keep 3D sequence
    # ─────────────────────────────
            
    elif model_type == "lstm" or model_type == "iTransformer":
        dist_feature = np.full(
            (window_size, 1),
            distance
        )

        return np.concatenate([dynamic_window, dist_feature], axis=1)
    
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}"
        )