# Bangalore House Price Prediction

A machine learning project that predicts residential property prices in Bengaluru using Python, Scikit-learn, and Streamlit.

## Project Highlights

- Processed an original dataset containing approximately **13,320 records and 9 columns**.
- Cleaned and filtered the data to approximately **7,275 property records**.
- Removed properties priced above **₹430 lakhs**.
- Evaluated regression models and selected **Linear Regression** for the final application.
- Achieved an **R² score of 0.8697** on held-out test data within the selected price range.
- Built an interactive Streamlit application for house price estimation.

## Tech Stack

Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn | Streamlit | Jupyter Notebook | Git | GitHub

## Project Workflow

### 1. Data Cleaning and Preprocessing
- Removed 2 unnecessary columns: `availability` and `society`.
- Handled missing values in BHK, bathrooms, and balconies.
- Converted `total_sqft` ranges into numeric values using their midpoint.
- Renamed `size` to `BHK` and `price` to `price_in_lakhs`.
- Grouped locations with **10 or fewer listings** into `others`.
- Removed properties with less than **300 sq. ft. per BHK**.
- Applied location-wise price-per-square-foot outlier filtering and BHK-based outlier removal.
- Removed the intermediate `price_per_sqft` feature after outlier handling.

### 2. Exploratory Data Analysis (EDA)
- Examined price and square-footage distributions.
- Analyzed BHK and property characteristics.
- Investigated location-wise price variation.
- Identified missing values, invalid records, and outliers.

### 3. Price Filtering
- Set a price cutoff of **₹430 lakhs**, based on the 99th percentile of the cleaned dataset.
- Excluded properties above this cutoff from the final modeling dataset.
- Retained approximately **7,275 records** for the final model and application.

### 4. Model Development
- Evaluated Linear Regression and Random Forest models.
- Selected **Linear Regression** for the final model based on the evaluation results.
- Used `OneHotEncoder(handle_unknown="ignore")` for `area_type` and `location`.
- Applied median imputation and `StandardScaler` to numerical features.
- Built a Scikit-learn pipeline and saved the trained model as `house_price_model.pkl`.

**Input features:** `area_type`, `location`, `BHK`, `total_sqft`, `bath`, `balcony`

**Target variable:** `price_in_lakhs`

### 5. Model Evaluation

The model was evaluated on **1,452 held-out test records** priced at or below ₹430 lakhs.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | ₹16.39 lakhs |
| Root Mean Squared Error (RMSE) | ₹25.88 lakhs |
| R² Score | 0.8697 |

### 6. Streamlit Application
- Accepts property details through an interactive interface.
- Loads the saved model and cleaned dataset.
- Predicts estimated house prices based on the selected features.
- Displays predictions in lakhs.

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

Replace `YOUR_USERNAME` with your GitHub username.

## Limitations
- The model focuses on properties priced at or below ₹430 lakhs.
- Predictions depend on historical data and the features available.
- Estimates are not official property valuations.

## Disclaimer
This project is intended for educational purposes. Predicted prices should not be treated as guaranteed market values.
