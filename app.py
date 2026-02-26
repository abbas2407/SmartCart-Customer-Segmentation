import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import matplotlib.pyplot as plt
import seaborn as sns

st.title("SmartCart Customer Segmentation")

# Load your data
df = pd.read_csv("smartcart_customers.csv")
st.write("Sample Data", df.head())

# --- Sidebar Controls ---
st.sidebar.header("Clustering Controls")
clustering_method = st.sidebar.selectbox("Clustering Method", ["KMeans", "Agglomerative"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)

# --- Feature Engineering ---
df["Age"] = 2026 - df["Year_Birth"]
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
reference_date = df["Dt_Customer"].max()
df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days
df["Total_Spending"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

# Education and Marital Status grouping
df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate", "PhD": "Postgraduate"
})
df["Living_With"] = df["Marital_Status"].replace({
    "Married": "Partner", "Together": "Partner",
    "Single": "Alone", "Divorced": "Alone",
    "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
})

# Drop unnecessary columns
cols_to_drop = [
    "ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer",
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]
df_cleaned = df.drop(columns=cols_to_drop)

# Remove outliers
df_cleaned = df_cleaned[(df_cleaned["Age"] < 90)]
df_cleaned = df_cleaned[(df_cleaned["Income"] < 600_000)]

# One-hot encoding
cat_cols = ["Education", "Living_With"]
ohe = OneHotEncoder(sparse_output=False)
enc_cols = ohe.fit_transform(df_cleaned[cat_cols])
enc_df = pd.DataFrame(enc_cols, columns=ohe.get_feature_names_out(cat_cols), index=df_cleaned.index)
df_encoded = pd.concat([df_cleaned.drop(columns=cat_cols), enc_df], axis=1)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# --- Elbow Method for Optimal Clusters ---
wcss = []
scores = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(X_pca)
    wcss.append(kmeans.inertia_)
    if k >= 2:
        labels = kmeans.labels_
        scores.append(silhouette_score(X_pca, labels))

from kneed import KneeLocator
knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_k = knee.elbow

# Show optimal value in sidebar
st.sidebar.markdown(f"**Optimal clusters (Elbow method):** {optimal_k}")

# --- Clustering based on sidebar controls ---
if clustering_method == "KMeans":
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    labels = clusterer.fit_predict(X_pca)
else:
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = clusterer.fit_predict(X_pca)

df_encoded["cluster"] = labels

# --- Color palette for clusters ---
pal = sns.color_palette("tab10", n_colors=n_clusters)

# --- Prepare 2D scatter plot (Total Spending vs Income by Cluster) ---
fig_2d, ax = plt.subplots()
sns.scatterplot(
    x=df_encoded["Total_Spending"],
    y=df_encoded["Income"],
    hue=df_encoded["cluster"],
    palette=pal,
    ax=ax
)
ax.set_title("Total Spending vs Income by Cluster")

# --- Prepare 3D PCA plot for tabs ---
fig_3d = plt.figure(figsize=(8, 6))
ax3d = fig_3d.add_subplot(111, projection="3d")
ax3d.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=df_encoded["cluster"], cmap='tab10')
ax3d.set_xlabel("PCA1")
ax3d.set_ylabel("PCA2")
ax3d.set_zlabel("PCA3")
ax3d.set_title("3D PCA Projection")

# --- Elbow Method Plot ---
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(range(1, 11), wcss, marker='o')
ax_elbow.set_xlabel("Number of Clusters (K)")
ax_elbow.set_ylabel("WCSS")
ax_elbow.set_title("Elbow Method For Optimal K")

# --- Silhouette Score Plot ---
fig_silhouette, ax_silhouette = plt.subplots()
ax_silhouette.plot(range(2, 11), scores, marker='o', color='orange')
ax_silhouette.set_xlabel("Number of Clusters (K)")
ax_silhouette.set_ylabel("Silhouette Score")
ax_silhouette.set_title("Silhouette Scores for KMeans")

# --- Combined Plot (WCSS + Silhouette Score) ---
k_range = range(2, 11)
fig_combined, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(k_range, wcss[1:len(k_range)+1], marker="o", color="blue", label="WCSS")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("WCSS", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(k_range, scores[:len(k_range)], marker="x", color="red", linestyle="--", label="Silhouette Score")
ax2.set_ylabel("Silhouette Score", color="red")
ax2.tick_params(axis='y', labelcolor='red')

fig_combined.suptitle("WCSS and Silhouette Score vs K")
fig_combined.tight_layout()

# --- Cluster summary ---
cluster_summary = df_encoded.groupby("cluster").mean(numeric_only=True)

# --- Tabs for Multiple Visualizations ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Summary", 
    "2D Scatter Plot", 
    "3D PCA Plot", 
    "Elbow Method", 
    "Silhouette Score",
    "Combined Metrics"
])

with tab1:
    st.subheader("Cluster Summary (Mean Values)")
    st.dataframe(cluster_summary)

with tab2:
    st.subheader("Total Spending vs Income by Cluster")
    st.pyplot(fig_2d)

with tab3:
    st.subheader("3D PCA Projection")
    st.pyplot(fig_3d)

with tab4:
    st.subheader("Elbow Method for Optimal K")
    st.pyplot(fig_elbow)

with tab5:
    st.subheader("Silhouette Scores for KMeans")
    st.pyplot(fig_silhouette)

with tab6:
    st.subheader("WCSS and Silhouette Score vs K")
    st.pyplot(fig_combined)