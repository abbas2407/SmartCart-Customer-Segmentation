# SmartCart Customer Segmentation System

## 📌 Project Overview

**SmartCart** is a growing e-commerce platform aiming to enhance its marketing and customer retention strategies. This project uses unsupervised machine learning (clustering) to segment customers based on their demographics, purchase behavior, website activity, and engagement levels.

By identifying distinct customer groups, SmartCart can:
- Personalize marketing campaigns
- Improve customer retention
- Identify high-value and churn-prone users

---

## 🗂️ Dataset Description

Each row in the dataset represents a customer and contains multiple attributes describing their spending and activity on the platform.

**Key Features:**
- Demographics: Age, Education, Marital Status, Income, Family Size
- Purchase Behavior: Amount spent on various product categories, purchase frequency
- Website Activity: Number of web visits, response to campaigns
- Engagement: Customer tenure, loyalty indicators

---

## ⚙️ How It Works

1. **Data Preprocessing:** Cleaning, handling missing values, feature engineering, and scaling
2. **Clustering:** Applying unsupervised learning algorithms (KMeans, Agglomerative) to group customers
3. **Cluster Analysis:** Interpreting clusters to understand customer segments
4. **Visualization:** Presenting clusters using interactive charts and dashboards

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit (for interactive dashboard)
- streamlit-lottie (for Lottie animations)

---

## 🚀 Features

- Interactive dashboard with multiple tabs:
  - **Summary:** Cluster mean values
  - **2D Scatter Plot:** Visualize clusters by spending and income
  - **3D PCA Plot:** Explore clusters in reduced dimensions
  - **Cluster Distribution:** Pie chart of cluster sizes
  - **Silhouette Score:** Evaluate clustering quality
  - **Raw Data & Download:** View and download clustered data
- Lottie animations for a modern look
- User controls for cluster number and algorithm selection

---

## 🚦 How to Run Locally

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/SmartCart-Customer-Segmentation.git
    cd SmartCart-Customer-Segmentation
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

4. **Open your browser:**  
   Go to [http://localhost:8501](http://localhost:8501) to interact with the app.

---

## 🎯 Learning Outcomes

- Applied unsupervised learning for real-world business problems
- Gained experience in customer segmentation and cluster analysis
- Built an interactive dashboard for business decision-making

---

## 🌱 Future Improvements

- Integrate more advanced clustering algorithms (DBSCAN, Gaussian Mixture)
- Add customer profiling and recommendation features
- Deploy the app online for broader access

---

## 🤝 Feedback

Feedback and suggestions are welcome!  
Feel free to open an issue or submit a pull request.