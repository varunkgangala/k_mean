import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
}
.cluster-info {
    background-color: rgba(255,255,255,0.95);
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers (4).csv")

df = load_data()

# =========================
# TRAIN MODEL (NO PKL)
# =========================
FEATURES = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[FEATURES])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# =========================
# CLUSTER INFO
# =========================
CLUSTER_INFO = {
    0: "High Value Customers",
    1: "Potential Targets",
    2: "Average Customers",
    3: "Loyal Customers",
    4: "Budget Conscious"
}

COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]

# =========================
# TITLE
# =========================
st.markdown("""
<h1 style='color:white;text-align:center;'>🛍️ Mall Customer Clustering</h1>
<p style='color:white;text-align:center;'>Live K-Means clustering without model files</p>
""", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("👤 Age", int(df.Age.min()), int(df.Age.max()), 30)
    income = st.slider("💰 Annual Income (k$)",
                       int(df['Annual Income (k$)'].min()),
                       int(df['Annual Income (k$)'].max()), 50)
    score = st.slider("🎯 Spending Score", 1, 100, 50)

with col2:
    st.metric("Total Customers", len(df))
    st.metric("Avg Age", f"{df.Age.mean():.1f}")
    st.metric("Avg Income", f"${df['Annual Income (k$)'].mean():.1f}k")

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Cluster", use_container_width=True):

    input_df = pd.DataFrame({
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [score]
    })

    scaled_input = scaler.transform(input_df)
    cluster = int(kmeans.predict(scaled_input)[0])

    st.markdown(f"""
    <div class="prediction-box">
        <h2>Cluster {cluster}</h2>
        <h1>{CLUSTER_INFO[cluster]}</h1>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # CLUSTER STATS
    # =========================
    cluster_data = df[df.Cluster == cluster]

    st.markdown("### 📊 Cluster Statistics")
    c1, c2, c3 = st.columns(3)

    c1.metric("Customers", len(cluster_data))
    c2.metric("Avg Age", f"{cluster_data.Age.mean():.1f}")
    c3.metric("Avg Spending", f"{cluster_data['Spending Score (1-100)'].mean():.1f}")

    # =========================
    # VISUALIZATION
    # =========================
    st.markdown("### 📈 Visualizations")
    v1, v2 = st.columns(2)

    with v1:
        fig3d = px.scatter_3d(
            df,
            x="Age",
            y="Annual Income (k$)",
            z="Spending Score (1-100)",
            color=df.Cluster.astype(str),
            title="3D Customer Clusters",
            color_discrete_sequence=COLORS
        )

        fig3d.add_scatter3d(
            x=[age], y=[income], z=[score],
            mode="markers",
            marker=dict(size=12, color="red", symbol="diamond"),
            name="Your Input"
        )

        fig3d.update_layout(height=500)
        st.plotly_chart(fig3d, use_container_width=True)

    with v2:
        counts = df.Cluster.value_counts().sort_index()

        pie = go.Figure(data=[
            go.Pie(
                labels=[f"Cluster {i}" for i in counts.index],
                values=counts.values,
                marker=dict(colors=COLORS)
            )
        ])

        pie.update_layout(title="Cluster Distribution", height=500)
        st.plotly_chart(pie, use_container_width=True)

    # =========================
    # RECOMMENDATION
    # =========================
    RECOMMENDATIONS = {
        0: "🎯 Focus on premium & loyal customers",
        1: "📈 Target with marketing campaigns",
        2: "🎁 Provide discounts & bundles",
        3: "🤝 Build long-term relationships",
        4: "💎 Upsell selectively"
    }

    st.info(f"**Business Recommendation:** {RECOMMENDATIONS[cluster]}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:white;'>Live K-Means Clustering | No Model Files Used</p>",
    unsafe_allow_html=True
)
