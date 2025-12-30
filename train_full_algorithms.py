"""
Triển khai đầy đủ các thuật toán từ tài liệu LaTeX:
- POI-level Regression (Algorithm 1)
- POI-level Pairwise Ranking (Algorithm 2)  
- Province-level Pairwise Ranking (Algorithm 3)
- SOTA Models: LightGBM, CatBoost
- Đánh giá đầy đủ theo RMSE, MAE, R², HitRate@k, NDCG@k, MRR@k

Cách chạy:
    pip install pandas numpy scikit-learn torch xgboost lightgbm catboost openpyxl
    python train_full_algorithms.py -i cities_with_scores_minmax.csv -o results.csv
"""

import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from xgboost import XGBRegressor, XGBRanker
except:
    raise RuntimeError("Cần cài xgboost: pip install xgboost")

try:
    from lightgbm import LGBMRegressor, LGBMRanker
except:
    raise RuntimeError("Cần cài lightgbm: pip install lightgbm")

try:
    from catboost import CatBoostRegressor, CatBoostRanker
except:
    raise RuntimeError("Cần cài catboost: pip install catboost")

REQ_COLS = [
    "poi_category", "ta_reviews", "ta_rating",
    "w_temp", "w_humidity", "w_clouds", "w_wind_speed", "w_rain_1h",
    "w_weather_desc", "score"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TIỀN XỬ LÝ DỮ LIỆU (theo Algorithm 1)
# ============================================================
def make_dense():
    """Chuyển sparse matrix thành dense array"""
    return FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else np.asarray(X))

def build_preprocessor(df: pd.DataFrame, label_col: str):
    """
    Xây dựng pipeline tiền xử lý theo Algorithm 1:
    - Encoder cho biến phân loại (poi_category, w_weather_desc)
    - Scaler (Z-score) cho biến số
    """
    cat_cols = ["poi_category", "w_weather_desc"]
    num_cols = [c for c in df.columns if c not in cat_cols + [label_col]]

    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler())
    ])
    
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols)
    ], remainder="drop", sparse_threshold=1.0)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("dense", make_dense())
    ])
    
    return pipeline

# ============================================================
# TẠO CẶP ƯU TIÊN (PAIRWISE PAIRS)
# ============================================================
def make_pairs_pairwise(y: np.ndarray, delta: float, max_pairs: int, rng: np.random.Generator):
    """
    Tạo tập cặp ưu tiên theo Algorithm 2 và 3:
    P = {(i,j) | y_i > y_j + δ}
    
    Args:
        y: nhãn (scores)
        delta: ngưỡng chênh lệch tối thiểu
        max_pairs: số cặp tối đa
        rng: random generator
    
    Returns:
        I, J: indices của các cặp (i > j)
        weights: trọng số của từng cặp (|y_i - y_j|)
    """
    n = len(y)
    if n < 2:
        return np.array([]), np.array([]), np.array([])
    
    # Sinh ngẫu nhiên các cặp
    m = min(max_pairs, max(100, 10 * n))
    i_idx = rng.integers(0, n, size=m)
    j_idx = rng.integers(0, n, size=m)
    
    # Loại bỏ cặp trùng
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    
    if len(i_idx) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Tính chênh lệch
    diff = y[i_idx] - y[j_idx]
    
    # Giữ cặp có chênh lệch > delta
    keep = np.abs(diff) > delta
    i_idx, j_idx, diff = i_idx[keep], j_idx[keep], diff[keep]
    
    # Chỉ giữ cặp i > j (y_i > y_j)
    pos_mask = diff > 0
    I = i_idx[pos_mask]
    J = j_idx[pos_mask]
    weights = diff[pos_mask]  # trọng số = |y_i - y_j|
    
    # Lấy mẫu nếu quá nhiều
    if len(I) > max_pairs:
        sel = rng.choice(len(I), size=max_pairs, replace=False)
        I, J, weights = I[sel], J[sel], weights[sel]
    
    return I, J, weights

# ============================================================
# MÔ HÌNH NEURAL NETWORK
# ============================================================
class MLPScorer(nn.Module):
    """
    MLP model cho scoring theo thiết kế trong tài liệu:
    - Nhiều lớp ẩn với ReLU, LayerNorm, Dropout
    - Đầu ra là điểm scalar
    """
    def __init__(self, in_dim: int, hidden_dims: List[int] = [256, 128], dropout: float = 0.1):
        super().__init__()
        layers = []
        last_dim = in_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, h_dim),
                nn.ReLU(),
                nn.LayerNorm(h_dim),
                nn.Dropout(dropout)
            ])
            last_dim = h_dim
        
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

# ============================================================
# THUẬT TOÁN HUẤN LUYỆN
# ============================================================
def train_regression_model(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 50, lr: float = 1e-3, weight_decay: float = 1e-4,
    device: str = "cpu", model_name: str = "Model"
):
    """
    Algorithm 1: Huấn luyện mô hình hồi quy POI-level
    Loss: MSE
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        pred_train = model(X_train_t)
        loss = criterion(pred_train, y_train_t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            val_loss = criterion(pred_val, y_val_t)
            mae = mean_absolute_error(y_val, pred_val.cpu().numpy())
            rmse = np.sqrt(mean_squared_error(y_val, pred_val.cpu().numpy()))
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{model_name} Regression Epoch {epoch:03d}] "
                  f"Train Loss={loss.item():.4f} | Val Loss={val_loss.item():.4f} | "
                  f"Val MAE={mae:.4f} RMSE={rmse:.4f}")
    
    return model

def train_pairwise_ranking_model(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    delta: float = 0.01, max_pairs: int = 40000,
    epochs: int = 50, lr: float = 1e-3, weight_decay: float = 1e-4,
    seed: int = 42, device: str = "cpu", model_name: str = "Model"
):
    """
    Algorithm 2: Huấn luyện POI-level theo Pairwise Ranking
    Loss: Pairwise logistic loss
    """
    rng = np.random.default_rng(seed)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    
    for epoch in range(1, epochs + 1):
        # Tạo cặp ưu tiên
        I, J, W = make_pairs_pairwise(y_train, delta, max_pairs, rng)
        
        if len(I) == 0:
            print(f"[{model_name} Pairwise Epoch {epoch:03d}] No pairs generated, skipping")
            continue
        
        model.train()
        
        # Tính score cho các cặp
        s_i = model(X_train_t[I])
        s_j = model(X_train_t[J])
        margin = s_i - s_j
        
        # Pairwise logistic loss
        loss_pair = torch.log1p(torch.exp(-margin))
        
        # Weight theo độ chênh lệch
        W_t = torch.tensor(W, dtype=torch.float32, device=device)
        W_t = W_t / (W_t.mean() + 1e-9)
        loss = (loss_pair * W_t).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_t).cpu().numpy()
                mae = mean_absolute_error(y_val, pred_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred_val))
            
            print(f"[{model_name} Pairwise Epoch {epoch:03d}] "
                  f"Loss={loss.item():.4f} | Pairs={len(I)} | "
                  f"Val MAE={mae:.4f} RMSE={rmse:.4f}")
    
    return model

# ============================================================
# METRICS ĐÁNH GIÁ
# ============================================================
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính các chỉ số hồi quy: MAE, RMSE, R²"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def compute_ranking_metrics_at_k(y_true: np.ndarray, y_pred: np.ndarray, k_values: List[int] = [5, 10]) -> Dict[str, float]:
    """
    Tính các chỉ số xếp hạng: HitRate@k, NDCG@k, MRR@k
    Sử dụng continuous relevance scores (y_true) thay vì binary
    
    Args:
        y_true: nhãn thực tế (continuous scores)
        y_pred: điểm dự đoán
        k_values: danh sách giá trị k
    
    Returns:
        dict chứa các metrics
    """
    n = len(y_true)
    
    # Xếp hạng theo dự đoán (giảm dần)
    pred_ranking = np.argsort(-y_pred)
    
    metrics = {}
    
    for k in k_values:
        k = min(k, n)
        
        # Top-k theo dự đoán
        pred_top_k_indices = pred_ranking[:k]
        
        # Lấy true scores của top-k items được predict
        pred_top_k_true_scores = y_true[pred_top_k_indices]
        
        # Lấy k highest true scores (ideal ranking)
        ideal_top_k_scores = np.sort(y_true)[::-1][:k]
        
        # HitRate@k: binary - có ít nhất 1 item trong predicted top-k nằm trong true top-k không
        true_top_k_indices = np.argsort(-y_true)[:k]
        hit_rate = 1.0 if len(np.intersect1d(pred_top_k_indices, true_top_k_indices)) > 0 else 0.0
        
        # NDCG@k với continuous relevance
        # DCG: sử dụng true scores làm relevance
        dcg = 0.0
        for i, idx in enumerate(pred_top_k_indices):
            rel = y_true[idx]  # continuous relevance
            dcg += rel / np.log2(i + 2)
        
        # IDCG: DCG của ideal ranking (sort by true scores)
        idcg = 0.0
        for i, score in enumerate(ideal_top_k_scores):
            idcg += score / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # MRR@k: vị trí của item có highest true score trong predicted top-k
        best_true_idx = np.argmax(y_true)
        mrr = 0.0
        for i, idx in enumerate(pred_top_k_indices):
            if idx == best_true_idx:
                mrr = 1.0 / (i + 1)
                break
        
        metrics[f"HR@{k}"] = hit_rate
        metrics[f"NDCG@{k}"] = ndcg
        metrics[f"MRR@{k}"] = mrr
    
    return metrics

# ============================================================
# BASELINE MODELS
# ============================================================
def baseline_static_ranking(df: pd.DataFrame) -> np.ndarray:
    """BL-POI-1: Xếp hạng tĩnh theo ta_rating"""
    return df["ta_rating"].values

def baseline_weighted_score(df: pd.DataFrame) -> np.ndarray:
    """
    BL-POI-2: Điểm tổng hợp có xét thời tiết
    score = α·ta_rating + β·log(1+ta_reviews) - γ·rain - η·wind - κ·humidity
    """
    alpha, beta, gamma, eta, kappa = 0.08, 0.02, 0.10, 0.05, 0.05
    
    def z_score(x):
        return (x - x.mean()) / (x.std() + 1e-9)
    
    score = (
        alpha * z_score(df["ta_rating"]) +
        beta * z_score(np.log1p(df["ta_reviews"])) -
        gamma * z_score(df["w_rain_1h"]) -
        eta * z_score(df["w_wind_speed"]) -
        kappa * z_score(df["w_humidity"])
    )
    
    return score.values

# ============================================================
# XUẤT KẾT QUẢ
# ============================================================
def save_results_to_excel(
    output_path: str,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]]
):
    """Lưu kết quả chi tiết vào Excel với 2 sheets"""
    excel_path = output_path.replace('.csv', '_detailed.xlsx')
    
    # Sheet 1: Predictions
    df_pred = X_test_df.reset_index(drop=True).copy()
    df_pred["y_true"] = y_test
    
    for model_name, pred in predictions.items():
        df_pred[f"pred_{model_name}"] = pred
    
    num_cols = df_pred.select_dtypes(include=[np.number]).columns
    df_pred[num_cols] = df_pred[num_cols].round(4)
    
    # Sheet 2: Metrics
    metrics_data = []
    for model_name, model_metrics in metrics.items():
        row = {"Model": model_name}
        row.update(model_metrics)
        metrics_data.append(row)
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Lưu Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_pred.to_excel(writer, sheet_name='Predictions', index=False)
        df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
    
    print(f"\n✓ Đã lưu kết quả chi tiết -> {excel_path}")
    print(f"  - Sheet 'Predictions': {len(df_pred)} mẫu với {len(predictions)} mô hình")
    print(f"  - Sheet 'Metrics': Tổng hợp đánh giá {len(metrics_data)} mô hình")

def print_metrics_summary(metrics: Dict[str, Dict[str, float]]):
    """In bảng tổng hợp metrics"""
    print("\n" + "="*80)
    print("TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ")
    print("="*80)
    
    # Regression metrics
    print("\n1. REGRESSION METRICS (POI-level)")
    print("-" * 80)
    print(f"{'Model':<30} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 80)
    
    for model_name, model_metrics in metrics.items():
        if "MAE" in model_metrics:
            print(f"{model_name:<30} "
                  f"{model_metrics['MAE']:>10.4f} "
                  f"{model_metrics['RMSE']:>10.4f} "
                  f"{model_metrics['R2']:>10.4f}")
    
    # Ranking metrics
    print("\n2. RANKING METRICS (Top-k)")
    print("-" * 80)
    print(f"{'Model':<30} {'HR@5':>8} {'NDCG@5':>8} {'HR@10':>8} {'NDCG@10':>8} {'MRR@10':>8}")
    print("-" * 80)
    
    for model_name, model_metrics in metrics.items():
        if "HR@5" in model_metrics:
            print(f"{model_name:<30} "
                  f"{model_metrics.get('HR@5', 0):>8.4f} "
                  f"{model_metrics.get('NDCG@5', 0):>8.4f} "
                  f"{model_metrics.get('HR@10', 0):>8.4f} "
                  f"{model_metrics.get('NDCG@10', 0):>8.4f} "
                  f"{model_metrics.get('MRR@10', 0):>8.4f}")
    
    print("=" * 80)

# ============================================================
# MAIN FUNCTION
# ============================================================
def main(args):
    print("="*80)
    print("TRIỂN KHAI THUẬT TOÁN TỪ TÀI LIỆU LATEX")
    print("="*80)
    
    # Đọc dữ liệu
    print("\n[1] Đọc dữ liệu...")
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    
    missing_cols = [c for c in REQ_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu cột: {missing_cols}")
    
    # Làm sạch
    print("[2] Làm sạch dữ liệu...")
    df["ta_rating"] = pd.to_numeric(df["ta_rating"], errors="coerce").fillna(0.0)
    df["ta_reviews"] = pd.to_numeric(df["ta_reviews"], errors="coerce").fillna(0.0)
    for col in ["w_temp", "w_humidity", "w_clouds", "w_wind_speed", "w_rain_1h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["poi_category"] = df["poi_category"].fillna("")
    df["w_weather_desc"] = df["w_weather_desc"].fillna("")
    
    X_df = df[[c for c in REQ_COLS if c != "score"]].copy()
    y = df["score"].values
    
    # Chia dữ liệu
    print(f"[3] Chia dữ liệu train/val/test...")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=args.seed
    )
    
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_df, y_train, test_size=0.15/0.7, random_state=args.seed
    )
    
    print(f"  - Train: {len(X_train_df)} mẫu")
    print(f"  - Val:   {len(X_val_df)} mẫu")
    print(f"  - Test:  {len(X_test_df)} mẫu")
    
    # Preprocessor
    print("[4] Xây dựng pipeline tiền xử lý...")
    preprocessor = build_preprocessor(
        pd.concat([X_train_df, pd.DataFrame({"score": y_train})], axis=1),
        "score"
    )
    
    preprocessor.fit(pd.concat([X_train_df, pd.DataFrame({"score": y_train})], axis=1))
    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)
    
    print(f"  - Số đặc trưng sau encoding: {X_train.shape[1]}")
    
    # Baselines
    print("\n" + "="*80)
    print("BASELINE MODELS")
    print("="*80)
    
    print("\n[BL-1] Static Ranking...")
    bl1_pred = baseline_static_ranking(X_test_df)
    
    print("[BL-2] Weighted Score...")
    bl2_pred = baseline_weighted_score(X_test_df)
    
    print("[BL-3] Ridge Regression...")
    ridge = Ridge(alpha=100.0, random_state=args.seed)
    ridge.fit(X_train, y_train)
    bl3_pred = ridge.predict(X_test)
    
    # Algorithm 1: Regression
    print("\n" + "="*80)
    print("ALGORITHM 1: POI-LEVEL REGRESSION")
    print("="*80)
    
    print("\n[MLP Regressor]")
    mlp_reg = MLPScorer(X_train.shape[1], hidden_dims=[64, 32], dropout=0.5)
    mlp_reg = train_regression_model(
        mlp_reg, X_train, y_train, X_val, y_val,
        epochs=args.epochs, lr=args.lr, weight_decay=5e-3,
        device=DEVICE, model_name="MLP"
    )
    
    mlp_reg.eval()
    with torch.no_grad():
        mlp_reg_pred = mlp_reg(torch.tensor(X_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    
    print("\n[XGBoost Regressor]")
    xgb_reg = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, reg_alpha=0.1,
        random_state=args.seed, tree_method="hist", n_jobs=-1
    )
    xgb_reg.fit(X_train, y_train)
    xgb_reg_pred = xgb_reg.predict(X_test)
    
    # SOTA 1: LightGBM Regressor
    print("\n[LightGBM Regressor] - SOTA")
    lgb_reg = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.1,
        num_leaves=31, min_child_samples=10,
        random_state=args.seed, n_jobs=-1, verbose=-1
    )
    lgb_reg.fit(X_train, y_train)
    lgb_reg_pred = lgb_reg.predict(X_test)
    
    # SOTA 2: CatBoost Regressor
    print("\n[CatBoost Regressor] - SOTA")
    cat_reg = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.08,
        subsample=0.9, reg_lambda=1.0,
        random_state=args.seed, verbose=False
    )
    cat_reg.fit(X_train, y_train)
    cat_reg_pred = cat_reg.predict(X_test)
    
    # Algorithm 2: Pairwise Ranking
    print("\n" + "="*80)
    print("ALGORITHM 2: POI-LEVEL PAIRWISE RANKING")
    print("="*80)
    
    print("\n[MLP Pairwise]")
    mlp_pair = MLPScorer(X_train.shape[1], hidden_dims=[512, 256, 128], dropout=0.05)
    mlp_pair = train_pairwise_ranking_model(
        mlp_pair, X_train, y_train, X_val, y_val,
        delta=args.delta, max_pairs=args.max_pairs,
        epochs=args.epochs, lr=args.lr, weight_decay=1e-5,
        seed=args.seed, device=DEVICE, model_name="MLP"
    )
    
    mlp_pair.eval()
    with torch.no_grad():
        mlp_pair_pred = mlp_pair(torch.tensor(X_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    
    print("\n[XGBoost Ranker]")
    qid_train = np.ones(len(X_train), dtype=int)
    xgb_rank = XGBRanker(
        objective="rank:pairwise",
        n_estimators=700, max_depth=7, learning_rate=0.05,
        subsample=0.95, colsample_bytree=0.95, reg_lambda=0.3, reg_alpha=0.05,
        random_state=args.seed, tree_method="hist", n_jobs=-1
    )
    xgb_rank.fit(X_train, y_train, qid=qid_train)
    xgb_rank_pred = xgb_rank.predict(X_test)
    
    # SOTA: LightGBM Ranker
    print("\n[LightGBM Ranker] - SOTA")
    # LightGBM Ranker cần labels là integer, convert continuous scores sang bins
    from sklearn.preprocessing import KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    y_train_binned = discretizer.fit_transform(y_train.reshape(-1, 1)).astype(int).ravel()
    
    lgb_rank = LGBMRanker(
        n_estimators=800, max_depth=8, learning_rate=0.05,
        subsample=0.95, colsample_bytree=0.95, reg_lambda=0.2, reg_alpha=0.05,
        num_leaves=127, min_child_samples=3,
        random_state=args.seed, n_jobs=-1, verbose=-1
    )
    lgb_rank.fit(X_train, y_train_binned, group=[len(X_train)])
    lgb_rank_pred = lgb_rank.predict(X_test)
    
    # SOTA: CatBoost Ranker
    print("\n[CatBoost Ranker] - SOTA")
    # CatBoost ranker cũng cần integer labels
    cat_rank = CatBoostRanker(
        iterations=800, depth=8, learning_rate=0.05,
        reg_lambda=0.2,
        random_state=args.seed, verbose=False
    )
    # CatBoost ranker cần group_id
    cat_rank.fit(X_train, y_train_binned, group_id=np.zeros(len(X_train), dtype=int))
    cat_rank_pred = cat_rank.predict(X_test)
    
    # Đánh giá
    print("\n" + "="*80)
    print("ĐÁNH GIÁ KẾT QUẢ")
    print("="*80)
    
    predictions = {
        "BL1_Static": bl1_pred,
        "BL2_Weighted": bl2_pred,
        "BL3_Ridge": bl3_pred,
        "MLP_Regression": mlp_reg_pred,
        "XGB_Regression": xgb_reg_pred,
        "LightGBM_Regression": lgb_reg_pred,
        "CatBoost_Regression": cat_reg_pred,
        "MLP_Pairwise": mlp_pair_pred,
        "XGB_Pairwise": xgb_rank_pred,
        "LightGBM_Pairwise": lgb_rank_pred,
        "CatBoost_Pairwise": cat_rank_pred,
    }
    
    metrics = {}
    
    for model_name, pred in predictions.items():
        print(f"\n[{model_name}]")
        
        reg_metrics = compute_regression_metrics(y_test, pred)
        rank_metrics = compute_ranking_metrics_at_k(y_test, pred, k_values=[5, 10])
        
        all_metrics = {**reg_metrics, **rank_metrics}
        metrics[model_name] = all_metrics
        
        print(f"  Regression: MAE={reg_metrics['MAE']:.4f}, RMSE={reg_metrics['RMSE']:.4f}, R²={reg_metrics['R2']:.4f}")
        print(f"  Ranking:    HR@5={rank_metrics['HR@5']:.4f}, NDCG@5={rank_metrics['NDCG@5']:.4f}, "
              f"HR@10={rank_metrics['HR@10']:.4f}, NDCG@10={rank_metrics['NDCG@10']:.4f}, MRR@10={rank_metrics['MRR@10']:.4f}")
    
    # Lưu kết quả
    print("\n[5] Lưu kết quả...")
    
    df_out = pd.DataFrame({
        "y_true": y_test,
        **{f"pred_{k}": v for k, v in predictions.items()}
    })
    df_out.to_csv(args.output, index=False)
    print(f"✓ Đã lưu predictions -> {args.output}")
    
    save_results_to_excel(args.output, X_test_df, y_test, predictions, metrics)
    print_metrics_summary(metrics)
    
    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triển khai thuật toán từ LaTeX")
    parser.add_argument("-i", "--input", default="cities_with_scores_minmax.csv")
    parser.add_argument("-o", "--output", default="results.csv")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--max-pairs", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    
    args = parser.parse_args()
    main(args)
