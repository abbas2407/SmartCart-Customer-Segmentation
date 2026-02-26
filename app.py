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

st.sidebar.header("Clustering Controls")
clustering_method = st.sidebar.selectbox("Clustering Method", ["KMeans", "Agglomerative"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)

# Feature Engineering 
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

# 3D PCA Scatter Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
ax.set_title("3D PCA Projection")
st.pyplot(fig)

# Elbow Method for KMeans
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(X_pca)
    wcss.append(kmeans.inertia_)

knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_k = knee.elbow
st.write(f"Optimal number of clusters (elbow method): {optimal_k}")

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_xlabel("K")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Method For Optimal K")
st.pyplot(fig)

# Silhouette Score Plot
scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    scores.append(score)

fig, ax = plt.subplots()
ax.plot(range(2, 11), scores, marker='o')
ax.set_xlabel("K")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Scores for KMeans")
st.pyplot(fig)

# Combined WCSS and Silhouette Score Plot
k_range = range(2, 11)
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(k_range, wcss[1:len(k_range)+1], marker="o", color="blue", label="WCSS")
ax1.set_xlabel("K")
ax1.set_ylabel("WCSS", color="blue")
ax2 = ax1.twinx()
ax2.plot(k_range, scores, marker="x", color="red", linestyle="--", label="Silhouette Score")
ax2.set_ylabel("Silhouette Score", color="red")
fig.suptitle("WCSS and Silhouette Score vs K")
st.pyplot(fig)

# KMeans Clustering (using 4 clusters as example)
kmeans = KMeans(n_clusters=4, random_state=42)
labels_kmeans = kmeans.fit_predict(X_pca)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_kmeans)
ax.set_title("KMeans Clusters (3D PCA)")
st.pyplot(fig)

# Agglomerative Clustering
agg_clf = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels_agg = agg_clf.fit_predict(X_pca)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_agg)
ax.set_title("Agglomerative Clusters (3D PCA)")
st.pyplot(fig)

# Add cluster labels to data
df_encoded["cluster"] = labels_agg

# Countplot for clusters
pal = ["red", "blue", "yellow", "green"]
fig, ax = plt.subplots()
sns.countplot(x=df_encoded["cluster"], palette=pal, ax=ax)
ax.set_title("Cluster Counts")
st.pyplot(fig)

# Scatterplot for Total Spending vs Income by cluster
fig, ax = plt.subplots()
sns.scatterplot(x=df_encoded["Total_Spending"], y=df_encoded["Income"], hue=df_encoded["cluster"], palette=pal, ax=ax)
ax.set_title("Total Spending vs Income by Cluster")
st.pyplot(fig)

# Cluster summary
cluster_summary = df_encoded.groupby("cluster").mean(numeric_only=True)


# Prepare 2D scatter plot (Total Spending vs Income by Cluster)
fig_2d, ax = plt.subplots()
sns.scatterplot(
    x=df_encoded["Total_Spending"],
    y=df_encoded["Income"],
    hue=df_encoded["cluster"],
    palette=pal,
    ax=ax
)
ax.set_title("Total Spending vs Income by Cluster")

# Prepare 3D PCA plot
fig_3d = plt.figure(figsize=(8, 6))
ax3d = fig_3d.add_subplot(111, projection="3d")
ax3d.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=df_encoded["cluster"])
ax3d.set_xlabel("PCA1")
ax3d.set_ylabel("PCA2")
ax3d.set_zlabel("PCA3")
ax3d.set_title("3D PCA Projection")

# --- Tabs for Multiple Visualizations ---
tab1, tab2, tab3 = st.tabs(["Summary", "2D Scatter Plot", "3D PCA Plot"])

with tab1:
    st.subheader("Cluster Summary (Mean Values)")
    st.dataframe(cluster_summary)

with tab2:
    st.subheader("Total Spending vs Income by Cluster")
    st.pyplot(fig_2d)

with tab3:
    st.subheader("3D PCA Projection")
    st.pyplot(fig_3d)


