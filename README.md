# Bengaluru House Price Prediction

A machine learning project that predicts residential property prices in Bengaluru using **Python, XGBoost, Scikit-learn, and Streamlit**.

## Project Highlights

- Started with the Bengaluru housing dataset containing approximately **13,320 records and 9 columns**.
- Cleaned and filtered the data to **6,155 property records** with **6 input features** and the target `price_in_lakhs`.
- Applied data-quality checks and removed invalid/outlier records, including properties priced above **₹430 lakhs**.
- Evaluated Linear Regression, Ridge Regression, Random Forest, and XGBoost using the same preprocessing pipeline and train-test split.
- Selected **XGBoost Regressor** as the final model based on its test-set performance.
- Achieved an **R² score of 0.9196** on the held-out test set.
- Built an interactive Streamlit application for estimating property prices in lakhs and crores.

## Tech Stack

Python | Pandas | NumPy | Matplotlib | Scikit-learn | XGBoost | Streamlit | Jupyter Notebook | Git | GitHub

## Project Workflow

### 1. Data Cleaning and Preprocessing

- Removed unnecessary columns and handled missing or invalid property records.
- Converted square-footage ranges into numeric values where needed.
- Renamed relevant columns to `BHK` and `price_in_lakhs`.
- Cleaned location values and removed invalid/outlier records.
- Applied the project's selected filtering rules, including the ₹430-lakh upper price cutoff.
- Prepared a final dataset with **6,155 rows and 7 columns**: six input features and one target variable.

### 2. Exploratory Data Analysis (EDA)

- Examined price and square-footage distributions.
- Analyzed BHK and other property characteristics.
- Investigated location-wise price variation.
- Checked missing values, invalid records, and outliers.

### 3. Price Filtering

- Applied a cutoff of **₹430 lakhs** to restrict the target price range used by the final model.
- Excluded records above the cutoff from the final modeling dataset.
- Used the resulting **6,155-record dataset** for the final training notebook and Streamlit application.

### 4. Model Development

- Compared **Linear Regression**, **Ridge Regression**, **Random Forest**, and **XGBoost**.
- Selected **XGBoost Regressor** as the final model.
- Used `OneHotEncoder(handle_unknown="ignore")` for the categorical features `area_type` and `location`.
- Applied median imputation and `StandardScaler` to numerical features in a Scikit-learn preprocessing pipeline.
- Combined preprocessing and the estimator in a pipeline so the same transformations are applied during training and prediction.
- Saved the final XGBoost pipeline to `models/house_price_model.pkl` using Joblib.

**Input features:** `area_type`, `location`, `BHK`, `total_sqft`, `bath`, `balcony`  
**Target variable:** `price_in_lakhs`

### 5. Model Evaluation

The notebook uses an **80:20 train-test split** with `random_state=42` on the final dataset of 6,155 records: **4,924 training records** and **1,231 held-out test records**.

The reported metrics for the selected XGBoost model are:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | ₹11.1315 lakhs |
| Root Mean Squared Error (RMSE) | ₹17.5409 lakhs |
| R² Score | 0.9196 |

The notebook's model comparison reported the following results on the same split:

| Model | MAE (lakhs) | RMSE (lakhs) | R² Score |
|---|---:|---:|---:|
| **XGBoost** | **11.1315** | **17.5409** | **0.9196** |
| Linear Regression | 12.5766 | 19.6946 | 0.8987 |
| Ridge Regression | 12.6084 | 19.7433 | 0.8981 |
| Random Forest | 11.6195 | 20.2393 | 0.8930 |

MAE and RMSE are expressed in lakhs. Lower error values are better, while a higher R² score indicates that the model explains more of the variation in the target values. These metrics describe the evaluated split, not a guarantee of accuracy for every property.

### 6. Streamlit Application

- Provides an interactive interface for selecting a property area type and Bengaluru location.
- Accepts BHK, total square footage, bathroom count, and balcony count.
- Loads the saved XGBoost pipeline and cleaned dataset.
- Displays the predicted price in lakhs and an approximate value in crores.
- Includes a property summary and information about the dataset.

## Project Structure

```text
Bangalore-House-Price-Prediction/
├── data/
│   └── cleaned_bengaluru_house_data.csv
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
├── models/
│   └── house_price_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Usage

```bash
git clone https://github.com/YOUR_USERNAME/Bangalore-House-Price-Prediction.git
cd Bangalore-House-Price-Prediction
pip install -r requirements.txt
streamlit run app.py
```

Replace `YOUR_USERNAME` with your GitHub username. Make sure the cleaned dataset and saved model are present at the paths shown in the project structure.

## Limitations

- The model is trained on a filtered dataset with prices at or below ₹430 lakhs; estimates outside this range may be less reliable.
- Predictions depend on the coverage and quality of the historical dataset and the selected input features.
- The reported metrics come from one held-out test split and may differ on other data.
- The model does not account for every factor that can affect market price, such as exact building condition, age, amenities, legal status, or real-time market changes.

## Disclaimer

This project is for educational purposes only. Predicted prices are estimates and should not be treated as guaranteed market values or official property valuations.
