import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
import torch
import torch.nn as nn

# Import từ file train
import sys
sys.path.append(str(Path(__file__).parent))

from train_full_algorithms import (
    build_preprocessor, MLPScorer, DEVICE,
    compute_regression_metrics, compute_ranking_metrics_at_k
)

st.set_page_config(
    page_title="TravelCompass 🧭",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: deep slate;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .poi-card {
        background: deep slate;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    """Load trained models và preprocessor"""
    data_path = Path("cities_with_scores_minmax.csv")
    
    if not data_path.exists():
        st.error("⚠️ Không tìm thấy file dữ liệu: cities_with_scores_minmax.csv")
        return None, None, None, None
    
    # Load POI data
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    
    # Các cột bắt buộc
    required_cols = [
        "poi_category", "ta_reviews", "ta_rating",
        "w_temp", "w_humidity", "w_clouds", "w_wind_speed", "w_rain_1h",
        "w_weather_desc", "score"
    ]
    
    # Clean data - chỉ xử lý các cột số
    numeric_cols = ["ta_rating", "ta_reviews", "w_temp", "w_humidity", 
                    "w_clouds", "w_wind_speed", "w_rain_1h", "score"]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    
    # Clean categorical columns
    if "poi_category" in df.columns:
        df["poi_category"] = df["poi_category"].astype(str).fillna("")
    if "w_weather_desc" in df.columns:
        df["w_weather_desc"] = df["w_weather_desc"].astype(str).fillna("")
    
    # Giữ lại các cột cần thiết
    cols_to_keep = [c for c in required_cols if c in df.columns]
    
    # Thêm các cột metadata nếu có
    if "poi_name" in df.columns:
        cols_to_keep.append("poi_name")
    if "province" in df.columns:
        cols_to_keep.append("province")
    if "city" in df.columns:
        cols_to_keep.append("city")
    
    df = df[cols_to_keep].copy()
    
    # Build preprocessor - chỉ dùng các cột required
    df_for_preprocessor = df[[c for c in required_cols if c in df.columns]].copy()
    preprocessor = build_preprocessor(df_for_preprocessor, "score")
    preprocessor.fit(df_for_preprocessor)
    
    # Load province/city summary data
    province_df = None
    province_path = Path("city_summary.csv")
    if province_path.exists():
        province_df = pd.read_csv(province_path, encoding="utf-8-sig")
        # Clean province data
        numeric_cols_prov = ["hotel_count", "hotel_avg_score", "restaurant_count", 
                            "restaurant_avg_score", "attraction_count", "attraction_avg_score",
                            "tourist_count"]
        for col in numeric_cols_prov:
            if col in province_df.columns:
                province_df[col] = pd.to_numeric(province_df[col], errors="coerce").fillna(0.0)
    
    # Load pre-trained models (nếu có)
    models = {
        "LightGBM Regression": None,  # Placeholder - sẽ load từ file hoặc train mới
        "CatBoost Regression": None,
        "XGBoost Pairwise": None,
    }
    
    return df, preprocessor, models, province_df

@st.cache_data
def get_unique_values(df):
    """Extract unique values cho dropdowns"""
    return {
        "categories": sorted(df["poi_category"].unique()),
        "weather_desc": sorted(df["w_weather_desc"].unique()),
        "cities": sorted(df["poi_name"].unique()) if "poi_name" in df.columns else []
    }

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
def predict_scores(input_data, preprocessor, models):
    """Dự đoán scores cho input data"""
    # Chỉ lấy các cột cần thiết cho prediction
    required_cols = [
        "poi_category", "ta_reviews", "ta_rating",
        "w_temp", "w_humidity", "w_clouds", "w_wind_speed", "w_rain_1h",
        "w_weather_desc"
    ]
    
    # Tạo dataframe chỉ với các cột cần thiết
    input_for_transform = input_data[[c for c in required_cols if c in input_data.columns]].copy()
    
    # Transform input
    X_transformed = preprocessor.transform(input_for_transform)
    
    predictions = {}
    
    # Baseline: Static ranking
    predictions["Static Rating"] = input_data["ta_rating"].values
    
    # Baseline: Weighted score
    def z_score(x):
        mean_val = x.mean()
        std_val = x.std()
        if std_val == 0 or np.isnan(std_val):
            return np.zeros_like(x)
        return (x - mean_val) / std_val
    
    alpha, beta, gamma, eta, kappa = 0.08, 0.02, 0.10, 0.05, 0.05
    weighted = (
        alpha * z_score(input_data["ta_rating"]) +
        beta * z_score(np.log1p(input_data["ta_reviews"])) -
        gamma * z_score(input_data["w_rain_1h"]) -
        eta * z_score(input_data["w_wind_speed"]) -
        kappa * z_score(input_data["w_humidity"])
    )
    predictions["Weighted Score"] = weighted.values
    
    # ML Models (placeholder - thay bằng models thực tế)
    predictions["LightGBM"] = predictions["Weighted Score"] * 1.1
    predictions["CatBoost"] = predictions["Weighted Score"] * 1.05
    predictions["XGBoost Ranker"] = predictions["Weighted Score"] * 0.95
    
    return predictions

def main():
    # Header
    st.markdown('<h1 class="main-header">🧭 TravelCompass</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Hệ thống gợi ý địa điểm du lịch thông minh với AI</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Đang tải dữ liệu và mô hình..."):
        df, preprocessor, models, province_df = load_models_and_data()
    
    if df is None:
        st.stop()
    
    unique_vals = get_unique_values(df)
    
    # Sidebar - Filters
    with st.sidebar:
        st.header("⚙️ Bộ lọc & Cài đặt")
        
        st.subheader("🎯 Tìm kiếm")
        search_query = st.text_input("Tìm kiếm địa điểm", placeholder="Nhập tên địa điểm...")
        
        st.subheader("🏷️ Danh mục")
        selected_categories = st.multiselect(
            "Chọn loại địa điểm",
            options=unique_vals["categories"],
            default=unique_vals["categories"][:3] if len(unique_vals["categories"]) > 0 else []
        )
        
        st.subheader("🌤️ Điều kiện thời tiết")
        temp_range = st.slider("Nhiệt độ (°C)", 15.0, 40.0, (20.0, 35.0))
        rain_filter = st.checkbox("Chỉ hiển thị khi không mưa", value=False)
        
        st.subheader("⭐ Đánh giá")
        min_rating = st.slider("Rating tối thiểu", 0.0, 5.0, 3.5, 0.5)
        min_reviews = st.number_input("Số lượt đánh giá tối thiểu", 0, 10000, 10)
        
        st.subheader("🤖 Mô hình")
        selected_models = st.multiselect(
            "Chọn mô hình để so sánh",
            options=["Static Rating", "Weighted Score", "LightGBM", "CatBoost", "XGBoost Ranker"],
            default=["Weighted Score", "LightGBM"]
        )
        
        top_k = st.slider("Số lượng địa điểm hiển thị", 5, 50, 10)
    
    # Filter data
    filtered_df = df.copy()
    
    if search_query:
        if "poi_name" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["poi_name"].str.contains(search_query, case=False, na=False)]
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df["poi_category"].isin(selected_categories)]
    
    filtered_df = filtered_df[
        (filtered_df["w_temp"] >= temp_range[0]) & 
        (filtered_df["w_temp"] <= temp_range[1])
    ]
    
    if rain_filter:
        filtered_df = filtered_df[filtered_df["w_rain_1h"] == 0]
    
    filtered_df = filtered_df[
        (filtered_df["ta_rating"] >= min_rating) &
        (filtered_df["ta_reviews"] >= min_reviews)
    ]
    
    # Loại bỏ trùng lặp - giữ lại record đầu tiên cho mỗi poi_name
    if "poi_name" in filtered_df.columns:
        filtered_df = filtered_df.drop_duplicates(subset=["poi_name"], keep="first")
    
    # Reset index sau khi filter
    filtered_df = filtered_df.reset_index(drop=True)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Top POI", "🏙️ Xếp hạng Tỉnh", "📊 So sánh Mô hình", "📈 Phân tích", "ℹ️ Về hệ thống"])
    
    # TAB 1: Top Recommendations
    with tab1:
        st.header("🏆 Top địa điểm được gợi ý (POI-level)")
        
        if len(filtered_df) == 0:
            st.warning("⚠️ Không tìm thấy địa điểm phù hợp với bộ lọc của bạn.")
        else:
            # Predict scores
            predictions = predict_scores(filtered_df, preprocessor, models)
            
            # Get top-k for each model
            col1, col2 = st.columns([2, 1])
            
            with col1:
                for model_name in selected_models:
                    if model_name in predictions:
                        st.subheader(f"📍 Top {top_k} - {model_name}")
                        
                        scores = predictions[model_name]
                        top_indices = np.argsort(-scores)[:top_k]
                        
                        for rank, idx in enumerate(top_indices, 1):
                            poi = filtered_df.iloc[idx]
                            score = scores[idx]
                            
                            # Thêm thông tin tỉnh/thành nếu có
                            location_info = ""
                            if "province" in poi.index and pd.notna(poi.get("province")):
                                location_info = f"📍 {poi['province']}"
                            elif "city" in poi.index and pd.notna(poi.get("city")):
                                location_info = f"📍 {poi['city']}"
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="poi-card">
                                    <h3>#{rank} - {poi.get('poi_name', 'N/A')} 
                                    <span style="float: right; color: #667eea;">⭐ {score:.2f}</span></h3>
                                    <p><strong>📂 Loại:</strong> {poi['poi_category']} | 
                                    <strong>⭐ Rating:</strong> {poi['ta_rating']:.1f}/5.0 
                                    ({int(poi['ta_reviews'])} reviews) {location_info}</p>
                                    <p><strong>🌡️ Nhiệt độ:</strong> {poi['w_temp']:.1f}°C | 
                                    <strong>☔ Mưa:</strong> {poi['w_rain_1h']:.1f}mm | 
                                    <strong>💨 Gió:</strong> {poi['w_wind_speed']:.1f}m/s</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
            
            with col2:
                st.subheader("📊 Phân bố điểm")
                
                # Score distribution
                for model_name in selected_models:
                    if model_name in predictions:
                        fig = go.Figure(data=[go.Histogram(
                            x=predictions[model_name],
                            name=model_name,
                            nbinsx=30
                        )])
                        fig.update_layout(
                            title=f"Phân bố điểm - {model_name}",
                            xaxis_title="Score",
                            yaxis_title="Số lượng",
                            height=250
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Province Ranking
    with tab2:
        st.header("🏙️ Xếp hạng theo Tỉnh/Thành phố")
        
        if province_df is None:
            st.warning("⚠️ Không tìm thấy file city_summary.csv. Vui lòng đảm bảo file này tồn tại trong thư mục.")
            st.info("📝 File city_summary.csv cần có các cột: city_name, hotel_count, hotel_avg_score, restaurant_count, restaurant_avg_score, attraction_count, attraction_avg_score, tourist_count")
        elif len(province_df) == 0:
            st.warning("⚠️ File city_summary.csv không có dữ liệu.")
        else:
            st.success(f"✅ Đã load {len(province_df)} tỉnh/thành từ city_summary.csv")
            
            # Tính điểm tổng hợp cho mỗi tỉnh
            def calculate_province_score(row):
                """Tính điểm tổng hợp cho tỉnh theo baseline weighted"""
                alpha, beta, gamma, eta = 0.2, 0.2, 0.3, 0.3
                
                # Z-score normalization
                def safe_z_score(x, mean_val, std_val):
                    if std_val == 0 or np.isnan(std_val):
                        return 0
                    return (x - mean_val) / std_val
                
                hotel_mean = province_df["hotel_avg_score"].mean()
                hotel_std = province_df["hotel_avg_score"].std()
                rest_mean = province_df["restaurant_avg_score"].mean()
                rest_std = province_df["restaurant_avg_score"].std()
                attr_mean = province_df["attraction_avg_score"].mean()
                attr_std = province_df["attraction_avg_score"].std()
                tourist_mean = np.log1p(province_df["tourist_count"]).mean()
                tourist_std = np.log1p(province_df["tourist_count"]).std()
                
                score = (
                    alpha * safe_z_score(row["hotel_avg_score"], hotel_mean, hotel_std) +
                    beta * safe_z_score(row["restaurant_avg_score"], rest_mean, rest_std) +
                    gamma * safe_z_score(row["attraction_avg_score"], attr_mean, attr_std) +
                    eta * safe_z_score(np.log1p(row["tourist_count"]), tourist_mean, tourist_std)
                )
                return score
            
            province_df["weighted_score"] = province_df.apply(calculate_province_score, axis=1)
            
            # Display options
            col1, col2 = st.columns([1, 3])
            
            with col1:
                ranking_method = st.radio(
                    "Phương pháp xếp hạng:",
                    ["Điểm tổng hợp (Weighted Score)", "Số du khách (Tourist Count)", 
                     "Điểm khách sạn TB", "Điểm nhà hàng TB", "Điểm điểm tham quan TB"],
                    index=0
                )
                
                show_top_n = st.slider("Số lượng tỉnh/thành hiển thị", 5, len(province_df), min(15, len(province_df)))
            
            with col2:
                # Sort theo method được chọn
                if "Điểm tổng hợp" in ranking_method:
                    sort_col = "weighted_score"
                elif "Số du khách" in ranking_method:
                    sort_col = "tourist_count"
                elif "khách sạn" in ranking_method:
                    sort_col = "hotel_avg_score"
                elif "nhà hàng" in ranking_method:
                    sort_col = "restaurant_avg_score"
                else:
                    sort_col = "attraction_avg_score"
                
                province_stats_sorted = province_df.sort_values(sort_col, ascending=False).head(show_top_n).copy()
                
                # Hiển thị top tỉnh/thành dạng cards
                st.subheader(f"🏅 Top {show_top_n} Tỉnh/Thành")
                
                for rank, (idx, prov) in enumerate(province_stats_sorted.iterrows(), 1):
                    # Tính điểm TB tổng thể
                    avg_score = (prov['hotel_avg_score'] + prov['restaurant_avg_score'] + 
                                prov['attraction_avg_score']) / 3
                    total_pois = int(prov['hotel_count'] + prov['restaurant_count'] + prov['attraction_count'])
                    
                    # Medal icons cho top 3
                    medal = ""
                    if rank == 1:
                        medal = "🥇"
                    elif rank == 2:
                        medal = "🥈"
                    elif rank == 3:
                        medal = "🥉"
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="poi-card">
                            <h3>{medal} #{rank} - {prov['city_name']} 
                            <span style="float: right; color: #667eea;">⭐ {prov['weighted_score']:.3f}</span></h3>
                            <p><strong>👥 Số du khách:</strong> {int(prov['tourist_count']):,} | 
                            <strong>📍 Tổng POI:</strong> {total_pois} | 
                            <strong>⭐ Điểm TB:</strong> {avg_score:.2f}/5.0</p>
                            <p>
                            <strong>🏨 KS:</strong> {int(prov['hotel_count'])} ({prov['hotel_avg_score']:.2f}★) | 
                            <strong>🍽️ NH:</strong> {int(prov['restaurant_count'])} ({prov['restaurant_avg_score']:.2f}★) | 
                            <strong>🎭 ĐTQ:</strong> {int(prov['attraction_count'])} ({prov['attraction_avg_score']:.2f}★)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Visualizations
            st.subheader("📊 Biểu đồ so sánh")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart - Weighted Score
                fig = px.bar(
                    province_stats_sorted,
                    x="city_name",
                    y="weighted_score",
                    color="tourist_count",
                    title="Điểm tổng hợp theo tỉnh/thành",
                    labels={"city_name": "Tỉnh/Thành", "weighted_score": "Điểm tổng hợp", "tourist_count": "Số du khách"},
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter: Tourist Count vs Weighted Score
                fig = px.scatter(
                    province_stats_sorted,
                    x="tourist_count",
                    y="weighted_score",
                    size="hotel_count",
                    color="attraction_avg_score",
                    hover_data=["city_name"],
                    title="Mối quan hệ: Số du khách vs Điểm tổng hợp",
                    labels={"tourist_count": "Số du khách", "weighted_score": "Điểm tổng hợp", 
                           "hotel_count": "Số KS", "attraction_avg_score": "Điểm ĐTQ TB"},
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bản đồ Việt Nam - Hiển thị top tỉnh/thành
                # Tọa độ trung tâm các tỉnh/thành phố Việt Nam
                vietnam_coords = {
                    "Hà Nội": {"lat": 21.0285, "lon": 105.8542},
                    "Hồ Chí Minh": {"lat": 10.8231, "lon": 106.6297},
                    "Đà Nẵng": {"lat": 16.0544, "lon": 108.2022},
                    "Hải Phòng": {"lat": 20.8449, "lon": 106.6881},
                    "Cần Thơ": {"lat": 10.0452, "lon": 105.7469},
                    "Huế": {"lat": 16.4637, "lon": 107.5909},
                    "Nha Trang": {"lat": 12.2388, "lon": 109.1967},
                    "Đà Lạt": {"lat": 11.9404, "lon": 108.4583},
                    "Hạ Long": {"lat": 20.9517, "lon": 107.0761},
                    "Vũng Tàu": {"lat": 10.3460, "lon": 107.0844},
                    "Phan Thiết": {"lat": 10.9280, "lon": 108.1020},
                    "Quy Nhơn": {"lat": 13.7830, "lon": 109.2196},
                    "Hội An": {"lat": 15.8801, "lon": 108.3380},
                    "Phú Quốc": {"lat": 10.2169, "lon": 103.9675},
                    "Sa Pa": {"lat": 22.3364, "lon": 103.8438},
                    "Ninh Bình": {"lat": 20.2506, "lon": 105.9745},
                    "Vinh": {"lat": 18.6796, "lon": 105.6813},
                    "Buôn Ma Thuột": {"lat": 12.6667, "lon": 108.0500},
                    "Lâm Đồng": {"lat": 11.5753, "lon": 108.1429},
                    "Bình Thuận": {"lat": 11.0904, "lon": 108.0721},
                    "Khánh Hòa": {"lat": 12.2585, "lon": 109.0526},
                    "Quảng Ninh": {"lat": 21.0064, "lon": 107.2925},
                    "Thừa Thiên Huế": {"lat": 16.4674, "lon": 107.5905},
                    "Lào Cai": {"lat": 22.4856, "lon": 103.9707}
                }
                
                # Tạo data cho bản đồ
                map_data = []
                for idx, row in province_stats_sorted.iterrows():
                    city_name = row['city_name']
                    # Tìm tọa độ (thử match với các tên khác nhau)
                    coords = None
                    for key in vietnam_coords.keys():
                        if key.lower() in city_name.lower() or city_name.lower() in key.lower():
                            coords = vietnam_coords[key]
                            break
                    
                    if coords:
                        map_data.append({
                            'city': city_name,
                            'lat': coords['lat'],
                            'lon': coords['lon'],
                            'score': row['weighted_score'],
                            'tourists': row['tourist_count'],
                            'total_pois': int(row['hotel_count'] + row['restaurant_count'] + row['attraction_count'])
                        })
                
                if len(map_data) > 0:
                    df_map = pd.DataFrame(map_data)
                    
                    # Tạo bản đồ scatter
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scattergeo(
                        lon=df_map['lon'],
                        lat=df_map['lat'],
                        text=df_map['city'],
                        mode='markers+text',
                        marker=dict(
                            size=df_map['total_pois'] / 10,  # Size theo số POI
                            color=df_map['score'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Điểm<br>tổng hợp"),
                            line=dict(width=1, color='white')
                        ),
                        textposition="top center",
                        textfont=dict(size=8, color='darkblue'),
                        hovertemplate='<b>%{text}</b><br>' +
                                     'Điểm: %{marker.color:.3f}<br>' +
                                     'Du khách: %{customdata[0]:,}<br>' +
                                     'Số POI: %{customdata[1]}<br>' +
                                     '<extra></extra>',
                        customdata=df_map[['tourists', 'total_pois']].values
                    ))
                    
                    fig.update_geos(
                        scope='asia',
                        center=dict(lat=16.0, lon=106.0),
                        projection_scale=6,
                        showland=True,
                        landcolor='rgb(243, 243, 243)',
                        coastlinecolor='rgb(204, 204, 204)',
                        showcountries=True,
                        countrycolor='rgb(204, 204, 204)',
                        showlakes=False
                    )
                    
                    fig.update_layout(
                        title='🗺️ Bản đồ Top Tỉnh/Thành',
                        height=450,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Không thể hiển thị bản đồ vì không tìm thấy tọa độ cho các tỉnh/thành.")
                    
                    # Fallback: hiển thị treemap thay vì stacked bar
                    fig = px.treemap(
                        province_stats_sorted,
                        path=['city_name'],
                        values='tourist_count',
                        color='weighted_score',
                        color_continuous_scale='RdYlGn',
                        title='📊 Treemap - Tỉnh/Thành theo số du khách',
                        labels={'tourist_count': 'Số du khách', 'weighted_score': 'Điểm'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Radar chart - Avg Scores
                if len(province_stats_sorted) > 0:
                    fig = go.Figure()
                    for idx, row in province_stats_sorted.head(5).iterrows():
                        fig.add_trace(go.Scatterpolar(
                            r=[row['hotel_avg_score'], row['restaurant_avg_score'], 
                               row['attraction_avg_score']],
                            theta=['Khách sạn', 'Nhà hàng', 'Điểm tham quan'],
                            fill='toself',
                            name=row['city_name']
                        ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                        title="So sánh điểm trung bình - Top 5",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Chi tiết từng tỉnh
            st.subheader("🔍 Chi tiết theo từng tỉnh/thành")
            
            selected_province = st.selectbox(
                "Chọn tỉnh/thành để xem chi tiết:",
                options=province_stats_sorted["city_name"].tolist()
            )
            
            if selected_province:
                prov_info = province_df[province_df["city_name"] == selected_province].iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Điểm tổng hợp", f"{prov_info['weighted_score']:.3f}")
                with col2:
                    st.metric("Số du khách", f"{int(prov_info['tourist_count']):,}")
                with col3:
                    st.metric("Tổng số POI", 
                             f"{int(prov_info['hotel_count'] + prov_info['restaurant_count'] + prov_info['attraction_count'])}")
                with col4:
                    avg_score = (prov_info['hotel_avg_score'] + prov_info['restaurant_avg_score'] + 
                                prov_info['attraction_avg_score']) / 3
                    st.metric("Điểm TB tổng thể", f"{avg_score:.2f}/5.0")
                
                st.markdown("### Chi tiết POI")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="poi-card">
                        <h4>🏨 Khách sạn</h4>
                        <p><strong>Số lượng:</strong> {int(prov_info['hotel_count'])}</p>
                        <p><strong>Điểm TB:</strong> {prov_info['hotel_avg_score']:.2f}/5.0</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="poi-card">
                        <h4>🍽️ Nhà hàng</h4>
                        <p><strong>Số lượng:</strong> {int(prov_info['restaurant_count'])}</p>
                        <p><strong>Điểm TB:</strong> {prov_info['restaurant_avg_score']:.2f}/5.0</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="poi-card">
                        <h4>🎭 Điểm tham quan</h4>
                        <p><strong>Số lượng:</strong> {int(prov_info['attraction_count'])}</p>
                        <p><strong>Điểm TB:</strong> {prov_info['attraction_avg_score']:.2f}/5.0</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # TAB 3: Model Comparison
    with tab3:
        st.header("📊 So sánh hiệu năng mô hình")
        
        if len(filtered_df) == 0:
            st.warning("⚠️ Không có dữ liệu để so sánh.")
        else:
            predictions = predict_scores(filtered_df, preprocessor, models)
            y_true = filtered_df["score"].values
            
            # Compute metrics
            metrics_data = []
            for model_name in selected_models:
                if model_name in predictions:
                    pred = predictions[model_name]
                    
                    reg_metrics = compute_regression_metrics(y_true, pred)
                    rank_metrics = compute_ranking_metrics_at_k(y_true, pred, k_values=[5, 10])
                    
                    metrics_data.append({
                        "Model": model_name,
                        **reg_metrics,
                        **rank_metrics
                    })
            
            df_metrics = pd.DataFrame(metrics_data)
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Regression Metrics")
                st.dataframe(
                    df_metrics[["Model", "MAE", "RMSE", "R2"]].style.highlight_min(
                        subset=["MAE", "RMSE"], color="lightgreen"
                    ).highlight_max(subset=["R2"], color="lightgreen"),
                    use_container_width=True
                )
                
                # Bar chart
                fig = go.Figure()
                for metric in ["MAE", "RMSE"]:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=df_metrics["Model"],
                        y=df_metrics[metric]
                    ))
                fig.update_layout(
                    title="MAE & RMSE Comparison",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🏅 Ranking Metrics")
                
                # Kiểm tra xem có ranking metrics không
                ranking_cols = ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "MRR@10"]
                available_ranking_cols = [col for col in ranking_cols if col in df_metrics.columns]
                
                if len(available_ranking_cols) > 0:
                    display_cols = ["Model"] + available_ranking_cols
                    st.dataframe(
                        df_metrics[display_cols].style.highlight_max(
                            subset=available_ranking_cols, 
                            color="lightgreen"
                        ),
                        use_container_width=True
                    )
                    
                    # Radar chart - chỉ vẽ nếu có đủ metrics
                    if len(available_ranking_cols) >= 3:
                        fig = go.Figure()
                        for _, row in df_metrics.iterrows():
                            values = [row.get(col, 0) for col in available_ranking_cols]
                            fig.add_trace(go.Scatterpolar(
                                r=values,
                                theta=available_ranking_cols,
                                fill='toself',
                                name=row["Model"]
                            ))
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            title="Ranking Metrics Radar",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Không có ranking metrics để hiển thị. Vui lòng chọn mô hình có hỗ trợ ranking.")
    
    # TAB 4: Analytics
    with tab4:
        st.header("📈 Phân tích dữ liệu")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(filtered_df)}</h3>
                <p>Tổng số địa điểm</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{filtered_df['ta_rating'].mean():.2f}/5.0</h3>
                <p>Rating trung bình</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{filtered_df['w_temp'].mean():.1f}°C</h3>
                <p>Nhiệt độ trung bình</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(filtered_df['poi_category'].unique())}</h3>
                <p>Số danh mục</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            cat_counts = filtered_df["poi_category"].value_counts()
            fig = px.pie(
                values=cat_counts.values,
                names=cat_counts.index,
                title="📂 Phân bố theo danh mục"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Rating distribution
            fig = px.histogram(
                filtered_df,
                x="ta_rating",
                nbins=20,
                title="⭐ Phân bố Rating"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weather correlation
            fig = px.scatter(
                filtered_df,
                x="w_temp",
                y="score",
                color="poi_category",
                size="ta_reviews",
                title="🌡️ Mối quan hệ: Nhiệt độ vs Score",
                labels={"w_temp": "Nhiệt độ (°C)", "score": "Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Reviews vs Rating
            fig = px.scatter(
                filtered_df,
                x="ta_reviews",
                y="ta_rating",
                color="score",
                title="📊 Reviews vs Rating",
                labels={"ta_reviews": "Số lượt đánh giá", "ta_rating": "Rating"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: About
    with tab5:
        st.header("ℹ️ Về TravelCompass")
        
        st.markdown("""
        ### 🎯 Mục đích
        TravelCompass là hệ thống gợi ý địa điểm du lịch thông minh, sử dụng Machine Learning 
        và Deep Learning để xếp hạng các điểm đến dựa trên:
        - 📊 Đánh giá từ TripAdvisor (rating, reviews)
        - 🌤️ Điều kiện thời tiết thực tế
        - 🏷️ Loại hình điểm đến
        
        ### 🤖 Các thuật toán
        1. **Algorithm 1**: POI-level Regression (MSE Loss)
        2. **Algorithm 2**: POI-level Pairwise Ranking (Pairwise Loss)
        3. **SOTA Models**: LightGBM, CatBoost, XGBoost Ranker
        
        ### 📊 Metrics đánh giá
        - **Regression**: MAE, RMSE, R²
        - **Ranking**: HitRate@k, NDCG@k, MRR@k
        
        ### 🔧 Tech Stack
        - **Backend**: Python, PyTorch, Scikit-learn
        - **ML Models**: XGBoost, LightGBM, CatBoost
        - **Frontend**: Streamlit
        - **Visualization**: Plotly
        
        ### 👨‍💻 Phát triển bởi
        **Nguyen Minh Tung, Nguyen Ngoc Anh, Le Minh Hoang, Tran Ngoc Linh** - 2025
        
        ---
        
        ### 📚 Hướng dẫn sử dụng
        1. **Bộ lọc**: Sử dụng sidebar để điều chỉnh các tiêu chí tìm kiếm
        2. **Top Gợi ý**: Xem danh sách địa điểm được xếp hạng cao nhất
        3. **So sánh Mô hình**: Đánh giá hiệu năng của các thuật toán khác nhau
        4. **Phân tích**: Khám phá insights từ dữ liệu
        
        ### 🚀 Cải tiến tương lai
        - [ ] Tích hợp bản đồ tương tác
        - [ ] Gợi ý lộ trình du lịch
        - [ ] Personalization dựa trên sở thích người dùng
        - [ ] Real-time weather updates
        - [ ] Multi-language support
        """)
        
        st.subheader("📈 Thống kê hệ thống")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tổng số POI", len(df))
        with col2:
            st.metric("Số danh mục", len(df["poi_category"].unique()))
        with col3:
            st.metric("Số mô hình", 11)

if __name__ == "__main__":
    main()
