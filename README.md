# 🏡 Bengaluru House Price Prediction

### An End-to-End Machine Learning Project

Predict residential property prices in Bengaluru using **XGBoost, Python, Scikit-learn, and Streamlit**. Explore property details, select a location, and get an estimated price through an interactive web application.

<p align="center">
  <a href="https://house-price-prediction-seemz.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Try_the_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://github.com/Seema2908/Bangalore-House-Price-Prediction">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

---

## 📌 Project Overview

This project uses historical Bengaluru housing data to build a regression model that estimates residential property prices from selected property characteristics.

| Project Detail | Description |
|---|---|
| **Project Type** | Machine Learning — Regression |
| **Dataset** | Bengaluru housing dataset |
| **Final Dataset** | 6,155 property records |
| **Selected Model** | XGBoost Regressor |
| **Web Framework** | Streamlit |
| **Target Variable** | `price_in_lakhs` |

## 📊 Model Performance

The XGBoost model was evaluated on a held-out test set containing **1,231 records**.

<p align="center">
  <img src="https://img.shields.io/badge/R²_Score-91.96%25-2E8B57?style=for-the-badge" alt="R2 Score">
  <img src="https://img.shields.io/badge/MAE-₹11.13_Lakhs-2563EB?style=for-the-badge" alt="MAE">
  <img src="https://img.shields.io/badge/RMSE-₹17.54_Lakhs-7C3AED?style=for-the-badge" alt="RMSE">
</p>

| Model | MAE (lakhs) | RMSE (lakhs) | R² Score |
|---|---:|---:|---:|
| 🏆 **XGBoost** | **11.1315** | **17.5409** | **0.9196** |
| Linear Regression | 12.5766 | 19.6946 | 0.8987 |
| Ridge Regression | 12.6084 | 19.7433 | 0.8981 |
| Random Forest | 11.6195 | 20.2393 | 0.8930 |

**Evaluation setup:** 80:20 train-test split with `random_state=42` — 4,924 training records and 1,231 test records.

> **Note:** These results represent performance on one test split. They do not guarantee the same accuracy for every property or real-world market price.

## ✨ Key Features

- 🏠 **Price Prediction:** Estimate property prices in lakhs and crores.
- 📍 **Location Selection:** Choose from locations available in the cleaned dataset.
- 📐 **Property Details:** Enter area type, BHK, square footage, bathrooms, and balconies.
- 🤖 **XGBoost Model:** Use the selected regression model for predictions.
- 🔄 **Consistent Preprocessing:** Apply the saved preprocessing and model pipeline to incoming inputs.
- 🌐 **Live Web App:** Access the application through Streamlit Community Cloud.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | XGBoost, Scikit-learn |
| Preprocessing | OneHotEncoder, SimpleImputer, StandardScaler |
| Model Saving | Joblib |
| Web Application | Streamlit |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## 🔍 Project Workflow

### 1. Data Cleaning and Preprocessing

- Started with approximately **13,320 records and 9 columns**.
- Removed unnecessary columns and handled missing or invalid property records.
- Converted square-footage ranges into numeric values where required.
- Cleaned location values and renamed relevant columns.
- Applied the selected data-filtering rules, including the ₹430-lakh upper price cutoff.
- Prepared a final dataset containing **6,155 rows and 7 columns**.

### 2. Exploratory Data Analysis (EDA)

Investigated the dataset to understand property characteristics and price patterns.

- Price and square-footage distributions
- BHK and property characteristics
- Location-wise price variation
- Missing values, invalid entries, and outliers

### 3. Price Filtering

Applied the project's selected filtering rules to restrict the modeling dataset to properties priced at or below **₹430 lakhs**. The resulting 6,155 records were used for the final model training and evaluation.

### 4. Model Development

Evaluated four regression algorithms using the same train-test split and preprocessing approach:

- Linear Regression
- Ridge Regression
- Random Forest
- **XGBoost Regressor — selected model**

The preprocessing pipeline includes:

- `OneHotEncoder(handle_unknown="ignore")` for `area_type` and `location`
- Median imputation and `StandardScaler` for numerical features
- A combined Scikit-learn pipeline to apply consistent transformations during training and prediction

The trained pipeline was saved using Joblib as `models/house_price_model.pkl`.

### 5. Model Evaluation

Used an **80:20 train-test split** with `random_state=42`.

| Dataset | Records |
|---|---:|
| Training set | 4,924 |
| Test set | 1,231 |
| **Total** | **6,155** |

The final XGBoost model achieved an R² score of **0.9196**, with an MAE of **₹11.1315 lakhs** and an RMSE of **₹17.5409 lakhs** on the held-out test set.

### 6. Streamlit Deployment

The trained model is integrated into an interactive web application.

**[🚀 Open the Live Bengaluru House Price Prediction App](https://house-price-prediction-seemz.streamlit.app/)**

Users can select a location and area type, enter property details, and view the predicted price. The app loads the saved XGBoost pipeline and cleaned dataset.

---

## 🧾 Input Features

| Feature | Description |
|---|---|
| `area_type` | Type of property area |
| `location` | Bengaluru property location |
| `BHK` | Number of bedrooms, hall, and kitchen |
| `total_sqft` | Total property area in square feet |
| `bath` | Number of bathrooms |
| `balcony` | Number of balconies |
| `price_in_lakhs` | **Target variable** — property price in lakhs |

## 📁 Project Structure

```text
Bangalore-House-Price-Prediction/
│
├── data/
│   └── cleaned_bengaluru_house_data.csv
│
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
│
├── models/
│   └── house_price_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Run the Project Locally

### Prerequisites

- Python installed
- Git installed

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Seema2908/Bangalore-House-Price-Prediction.git
```

**2. Navigate to the project directory**

```bash
cd Bangalore-House-Price-Prediction
```

**3. Install the required dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal. Ensure the cleaned dataset and saved model exist at the paths specified in the project.

## ⚠️ Limitations

- The model is trained on a filtered dataset with prices at or below ₹430 lakhs; predictions outside this range may be less reliable.
- Predictions depend on the quality and coverage of the historical dataset.
- The evaluation metrics are based on one held-out test split.
- The model does not account for every factor that affects property prices, such as building condition, property age, amenities, legal status, or real-time market changes.

## 📌 Disclaimer

This project is intended for educational purposes. Predicted prices are estimates and should not be treated as guaranteed market values or official property valuations.

---

## 👩‍💻 Author

### Seema Timmanagoudar

🔗 **GitHub:** [Seema2908](https://github.com/Seema2908)

💻 **Project Repository:** [Bengaluru House Price Prediction](https://github.com/Seema2908/Bangalore-House-Price-Prediction)

🌐 **Live Application:** [Try the Prediction App](https://house-price-prediction-seemz.streamlit.app/)

---

<p align="center">
  <b>If you find this project interesting, consider giving the repository a ⭐ on GitHub!</b>
</p>
