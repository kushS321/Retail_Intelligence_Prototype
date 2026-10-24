# 🛍️ Retail Intelligence System
### Customer Segmentation & Purchasing Behavior Prediction System

An end-to-end Machine Learning decision support platform built for retail intelligence. This system combines **RFM behavioral customer segmentation** with **predictive classification (Logistic Regression pipeline)** to forecast 30-day customer repurchasing propensity, benchmark classifiers on temporal holdout data, and deliver data-driven marketing strategies.

---

## 🚀 Key Features

1. **🏠 Executive Command Hub:** Project KPI summary and quick navigation cards to all core modules.
2. **👥 Customer Segmentation Explorer:**
   - Discovers meaningful customer cohorts across 6 behavioral tiers (*Best Customers*, *Loyal Customers*, *At-Risk Valuable Customers*, *Regular Customers*, *Inactive Customers*, *Recent Customers*).
   - Interactive Plotly donut and bar distributions for customer counts and monetary spend.
   - Comprehensive segment profiles table with plain-English business definitions.
   - Real-time customer search, filtering by segment/prediction/outcome, and filtered CSV export.
3. **🛒 Purchase Prediction Interface:**
   - Real-time 30-day purchasing prediction powered by a trained Scikit-Learn Pipeline (`StandardScaler` → `LogisticRegression`).
   - Quick persona presets (*Best*, *Loyal*, *At-Risk*, *Inactive*) for demonstration.
   - Dynamic probability scoring, custom visual propensity gauge, and rule-based business action guidance.
   - Historical holdout confusion matrix and records inspector.
4. **📈 Machine Learning Model Evaluation:**
   - Rigorous temporal holdout benchmark comparing **Logistic Regression**, **Decision Tree**, and **Random Forest**.
   - Side-by-side grouped metrics comparison across Accuracy, Precision, Recall, and F1 Score.
   - Rationale explaining why Logistic Regression was selected for production (highest F1 Score of 53.36% and accuracy of 70.27%, minimizing false positives).
5. **💡 Strategic Business Insights:**
   - Key business KPIs, actual vs. predicted customer comparison by cohort, and average purchase propensity by segment.
   - Actionable data-driven marketing playbooks for each verified customer segment.

---

## 🛠️ Technology Stack

- **Python 3.10+**
- **Streamlit** (UI Framework & Dashboard)
- **Pandas & NumPy** (Data processing & feature analysis)
- **Plotly Express & Plotly Graph Objects** (Interactive visualizations)
- **Scikit-learn & Joblib** (ML Pipeline & Inference)

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── config.toml                                    # Theme configuration (Dark Navy & Slate)
├── app.py                                             # Main Streamlit application
├── requirements.txt                                   # Python dependencies
├── customer_segmentation_and_purchase_predictions.csv # 3,175 evaluated customer records
├── purchase_prediction_by_customer_segment.csv        # Segment summary benchmarks
├── purchasing_model_comparison_time_based.csv         # Benchmark metrics across 3 classifiers
├── logistic_regression_purchasing_model.pkl          # Trained Scikit-Learn model pipeline
├── test_retail_system.py                              # Automated test suite (11 unit tests)
├── .gitignore                                         # Git ignore rules
└── README.md                                          # Documentation
```

---

## 💻 Local Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

4. **Run automated tests:**
   ```bash
   python3 test_retail_system.py
   ```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to your **GitHub** account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your GitHub repository, branch (`main`), and set **Main file path** to `app.py`.
5. Click **"Deploy!"**.
