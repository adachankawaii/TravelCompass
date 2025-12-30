"""
Triển khai Algorithm 3: Province-level Pairwise Ranking từ tài liệu LaTeX
- Đánh giá ranking trên các tỉnh/thành
- 19 tỉnh có data: 13 tỉnh train, 6 tỉnh test
- Top-k = 3
- Metrics: HitRate@3, NDCG@3

Cách chạy:
    python train_ranknet_tourist.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple

try:
    from xgboost import XGBRanker
except:
    print("Warning: XGBoost not installed. XGBoost ranker will not be available.")
    XGBRanker = None

try:
    from lightgbm import LGBMRanker
except:
    print("Warning: LightGBM not installed. LightGBM ranker will not be available.")
    LGBMRanker = None

try:
    from catboost import CatBoostRanker
except:
    print("Warning: CatBoost not installed. CatBoost ranker will not be available.")
    CatBoostRanker = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MÔ HÌNH MLP CHO PAIRWISE RANKING
# ============================================================
class MLPRanker(nn.Module):
    """MLP Ranker cho Province-level theo Algorithm 3"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        last_dim = input_dim
        
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
# TẠO CẶP ƯU TIÊN (PAIRWISE PAIRS)
# ============================================================
def make_pairs_province(y: np.ndarray, delta: float = 0.01, max_pairs: int = 1000, 
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tạo tập cặp ưu tiên theo Algorithm 3:
    P = {(i,j) | y_i > y_j + δ}
    
    Args:
        y: nhãn (tourist_count normalized)
        delta: ngưỡng chênh lệch tối thiểu
        max_pairs: số cặp tối đa
        seed: random seed
    
    Returns:
        I, J: indices của các cặp (i > j)
        weights: trọng số của từng cặp (|y_i - y_j|)
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    
    if n < 2:
        return np.array([]), np.array([]), np.array([])
    
    # Tạo tất cả các cặp có thể
    all_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and y[i] > y[j] + delta:
                all_pairs.append((i, j, y[i] - y[j]))
    
    if len(all_pairs) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to arrays
    all_pairs = np.array(all_pairs)
    I = all_pairs[:, 0].astype(int)
    J = all_pairs[:, 1].astype(int)
    weights = all_pairs[:, 2]
    
    # Lấy mẫu nếu quá nhiều
    if len(I) > max_pairs:
        sel = rng.choice(len(I), size=max_pairs, replace=False)
        I, J, weights = I[sel], J[sel], weights[sel]
    
    return I, J, weights

# ============================================================
# HUẤN LUYỆN MLP PAIRWISE
# ============================================================
def train_mlp_pairwise(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    delta: float = 0.01, max_pairs: int = 1000,
    epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4,
    seed: int = 42, device: str = "cpu"
) -> nn.Module:
    """Huấn luyện MLP theo pairwise loss (Algorithm 3)"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    
    for epoch in range(1, epochs + 1):
        # Tạo cặp ưu tiên
        I, J, W = make_pairs_province(y_train, delta, max_pairs, seed + epoch)
        
        if len(I) == 0:
            print(f"[Epoch {epoch:03d}] No pairs generated, skipping")
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
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"[MLP Pairwise Epoch {epoch:03d}] Loss={loss.item():.4f} | Pairs={len(I)}")
    
    return model

# ============================================================
# METRICS ĐÁNH GIÁ
# ============================================================
def compute_ranking_metrics_at_k(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    k_values: List[int] = [3, 5]
) -> Dict[str, float]:
    """
    Tính HitRate@k và NDCG@k cho province ranking với nhiều giá trị k
    
    Args:
        y_true: nhãn thực tế (tourist_count normalized)
        y_pred: điểm dự đoán
        k_values: danh sách các giá trị k
    
    Returns:
        dict chứa HR@k và NDCG@k cho mỗi k
    """
    n = len(y_true)
    
    # Xếp hạng theo dự đoán (giảm dần)
    pred_ranking = np.argsort(-y_pred)
    
    metrics = {}
    
    for k in k_values:
        k = min(k, n)
        
        # Top-k theo dự đoán
        pred_top_k_indices = pred_ranking[:k]
        
        # Top-k theo ground truth
        true_top_k_indices = np.argsort(-y_true)[:k]
        
        # HitRate@k: có ít nhất 1 item trong predicted top-k nằm trong true top-k không
        hit_rate = 1.0 if len(np.intersect1d(pred_top_k_indices, true_top_k_indices)) > 0 else 0.0
        
        # NDCG@k
        # DCG: sử dụng true scores làm relevance
        dcg = 0.0
        for i, idx in enumerate(pred_top_k_indices):
            rel = y_true[idx]  # continuous relevance
            dcg += rel / np.log2(i + 2)
        
        # IDCG: DCG của ideal ranking
        ideal_top_k_scores = np.sort(y_true)[::-1][:k]
        idcg = 0.0
        for i, score in enumerate(ideal_top_k_scores):
            idcg += score / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        metrics[f"HR@{k}"] = hit_rate
        metrics[f"NDCG@{k}"] = ndcg
    
    return metrics

# ============================================================
# BASELINE MODELS
# ============================================================
def baseline_static_ranking(df: pd.DataFrame) -> np.ndarray:
    """BL-PROV-1: Xếp hạng tĩnh theo tourist_count"""
    return df["tourist_count"].values

def baseline_weighted_score(df: pd.DataFrame) -> np.ndarray:
    """
    BL-PROV-1: Điểm tổng hợp
    score = α·hotel_avg + β·restaurant_avg + γ·attraction_avg + η·log(tourist_count)
    """
    alpha, beta, gamma, eta = 0.2, 0.2, 0.3, 0.3
    
    def z_score(x):
        return (x - x.mean()) / (x.std() + 1e-9)
    
    score = (
        alpha * z_score(df["hotel_avg_score"]) +
        beta * z_score(df["restaurant_avg_score"]) +
        gamma * z_score(df["attraction_avg_score"]) +
        eta * z_score(np.log1p(df["tourist_count"]))
    )
    
    return score.values

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    print("="*80)
    print("PROVINCE-LEVEL PAIRWISE RANKING (Algorithm 3)")
    print("="*80)
    
    # Đọc dữ liệu
    print("\n[1] Đọc dữ liệu city_summary.csv...")
    df = pd.read_csv("city_summary.csv", encoding="utf-8-sig")
    
    print(f"  - Số tỉnh/thành: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")
    
    # Kiểm tra số lượng tỉnh
    if len(df) != 19:
        print(f"WARNING: Expected 19 provinces, got {len(df)}")
    
    # Feature columns
    feature_cols = [
        "hotel_count", "hotel_avg_score",
        "restaurant_count", "restaurant_avg_score",
        "attraction_count", "attraction_avg_score"
    ]
    
    # Target
    target_col = "tourist_count"
    
    # Normalize target (Min-Max)
    y_raw = df[target_col].values.astype(float)
    y_min, y_max = y_raw.min(), y_raw.max()
    y_normalized = (y_raw - y_min) / (y_max - y_min + 1e-8)
    
    print(f"\n[2] Chia dữ liệu: 13 tỉnh train, 6 tỉnh test...")
    
    # Shuffle với seed cố định
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    
    train_indices = indices[:13]
    test_indices = indices[13:19]
    
    # Train/test split
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()
    
    X_train_df = df_train[feature_cols]
    X_test_df = df_test[feature_cols]
    
    y_train = y_normalized[train_indices]
    y_test = y_normalized[test_indices]
    
    print(f"  - Train cities: {list(df_train['city_name'].values)}")
    print(f"  - Test cities: {list(df_test['city_name'].values)}")
    
    # Standardize features (Z-score)
    print("\n[3] Chuẩn hóa features (Z-score)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values.astype(float))
    X_test = scaler.transform(X_test_df.values.astype(float))
    
    print(f"  - Train shape: {X_train.shape}")
    print(f"  - Test shape: {X_test.shape}")
    
    # Baselines
    print("\n" + "="*80)
    print("BASELINE MODELS")
    print("="*80)
    
    print("\n[BL-PROV-1] Weighted Score...")
    bl1_pred_test = baseline_weighted_score(df_test)
    
    # Train models
    print("\n" + "="*80)
    print("PAIRWISE RANKING MODELS")
    print("="*80)
    
    # MLP Pairwise
    print("\n[1] MLP Pairwise Ranker...")
    mlp_ranker = MLPRanker(X_train.shape[1], hidden_dims=[128, 64], dropout=0.2)
    mlp_ranker = train_mlp_pairwise(
        mlp_ranker, X_train, y_train,
        delta=0.01, max_pairs=500,
        epochs=100, lr=1e-3, weight_decay=1e-4,
        seed=42, device=DEVICE
    )
    
    mlp_ranker.eval()
    with torch.no_grad():
        mlp_pred_test = mlp_ranker(torch.tensor(X_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    
    # XGBoost Ranker
    xgb_pred_test = None
    if XGBRanker is not None:
        print("\n[2] XGBoost Ranker...")
        qid_train = np.zeros(len(X_train), dtype=int)  # All same group
        xgb_ranker = XGBRanker(
            objective="rank:pairwise",
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=0.5,
            random_state=42, tree_method="hist", n_jobs=-1
        )
        xgb_ranker.fit(X_train, y_train, qid=qid_train)
        xgb_pred_test = xgb_ranker.predict(X_test)
    
    # LightGBM Ranker
    lgb_pred_test = None
    if LGBMRanker is not None:
        print("\n[3] LightGBM Ranker...")
        from sklearn.preprocessing import KBinsDiscretizer
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        y_train_binned = discretizer.fit_transform(y_train.reshape(-1, 1)).astype(int).ravel()
        
        lgb_ranker = LGBMRanker(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=0.5,
            num_leaves=31, min_child_samples=3,
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_ranker.fit(X_train, y_train_binned, group=[len(X_train)])
        lgb_pred_test = lgb_ranker.predict(X_test)
    
    # CatBoost Ranker
    cat_pred_test = None
    if CatBoostRanker is not None:
        print("\n[4] CatBoost Ranker...")
        from sklearn.preprocessing import KBinsDiscretizer
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        y_train_binned = discretizer.fit_transform(y_train.reshape(-1, 1)).astype(int).ravel()
        
        cat_ranker = CatBoostRanker(
            iterations=300, depth=5, learning_rate=0.05,
            reg_lambda=0.5,
            random_state=42, verbose=False
        )
        cat_ranker.fit(X_train, y_train_binned, group_id=np.zeros(len(X_train), dtype=int))
        cat_pred_test = cat_ranker.predict(X_test)
    
    # Evaluation
    print("\n" + "="*80)
    print("ĐÁNH GIÁ KẾT QUẢ (HR@k và NDCG@k)")
    print("="*80)
    
    predictions = {
        "BL-PROV-1 (Weighted)": bl1_pred_test,
        "MLP Pairwise": mlp_pred_test,
    }
    
    if xgb_pred_test is not None:
        predictions["XGBoost Ranker"] = xgb_pred_test
    if lgb_pred_test is not None:
        predictions["LightGBM Ranker"] = lgb_pred_test
    if cat_pred_test is not None:
        predictions["CatBoost Ranker"] = cat_pred_test
    
    print(f"\n{'Model':<25} {'HR@3':>8} {'NDCG@3':>8} {'HR@5':>8} {'NDCG@5':>8}")
    print("-" * 70)
    
    results = {}
    for model_name, pred in predictions.items():
        metrics = compute_ranking_metrics_at_k(y_test, pred, k_values=[3, 5])
        results[model_name] = metrics
        
        print(f"{model_name:<25} {metrics['HR@3']:>8.4f} {metrics['NDCG@3']:>8.4f} "
              f"{metrics['HR@5']:>8.4f} {metrics['NDCG@5']:>8.4f}")
    
    # Print detailed ranking
    print("\n" + "="*80)
    print("DETAILED RANKING (Top-3 for each model)")
    print("="*80)
    
    test_cities = df_test['city_name'].values
    print(f"\nGround Truth (by tourist_count):")
    true_ranking = np.argsort(-y_test)[:3]
    for i, idx in enumerate(true_ranking, 1):
        print(f"  {i}. {test_cities[idx]} (score={y_test[idx]:.4f})")
    
    for model_name, pred in predictions.items():
        print(f"\n{model_name}:")
        pred_ranking = np.argsort(-pred)[:3]
        for i, idx in enumerate(pred_ranking, 1):
            print(f"  {i}. {test_cities[idx]} (pred={pred[idx]:.4f}, true={y_test[idx]:.4f})")
    
    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)

if __name__ == "__main__":
    main()
