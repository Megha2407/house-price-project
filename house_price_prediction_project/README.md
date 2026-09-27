# 🏡 House Price Prediction

A Machine Learning web application that predicts the estimated sale price of residential properties in India based on property characteristics such as location, area, BHK, bathrooms, property age, and amenities.

## 📌 Project Overview

House prices depend on several factors, including location, property size, building characteristics, and nearby facilities.

This project uses machine learning regression algorithms to learn patterns from historical housing data and estimate property prices. The application provides a simple interface where users can enter property details and receive a predicted price in Indian rupees lakhs.

## 🎯 Objectives

* Perform data cleaning and exploratory data analysis (EDA).
* Handle missing values and analyze outliers.
* Perform feature engineering.
* Preprocess numerical and categorical features.
* Train and compare machine learning regression models.
* Evaluate models using MAE, RMSE, and R².
* Save the trained model pipeline.
* Deploy an interactive prediction application.

## ✨ Features

* Interactive property input form.
* Property location and type selection.
* Support for BHK, bathrooms, and area details.
* Consideration of floor, age, and furnishing.
* Property amenities and accessibility features.
* Machine learning-based price estimation.
* Web interface built using Streamlit.

## 🧰 Technologies Used

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Programming language               |
| Pandas       | Data manipulation                  |
| NumPy        | Numerical operations               |
| Scikit-learn | Machine learning and preprocessing |
| Matplotlib   | Data visualization                 |
| Seaborn      | Exploratory data analysis          |
| Joblib       | Model serialization                |
| Streamlit    | Web application                    |
| Git & GitHub | Version control                    |

## 📊 Dataset

The dataset contains residential property information, including:

* City and locality type
* Property type
* BHK and bathrooms
* Super area and carpet area
* Floor number and total floors
* Age of property
* Furnishing status
* Parking and lift availability
* Gated community
* Distance to metro and city centre

**Target variable:** `Price_INR_Lakhs`

The target represents the property price in lakhs of Indian rupees.

## 🔍 Project Workflow

1. **Data collection:** Load the housing dataset.
2. **Data cleaning:** Inspect data types, missing values, and inconsistencies.
3. **Exploratory data analysis:** Analyze distributions, relationships, and outliers.
4. **Feature engineering:** Create additional features from existing property details.
5. **Preprocessing:** Impute missing values, scale numerical features, and encode categorical features.
6. **Model training:** Train regression algorithms.
7. **Model comparison:** Compare validation results and tune hyperparameters.
8. **Evaluation:** Assess performance using regression metrics.
9. **Model saving:** Save the trained pipeline using Joblib.
10. **Deployment:** Use Streamlit to provide predictions through a web interface.

## 🤖 Machine Learning Models

The project compares the following regression models:

* Random Forest Regressor
* Extra Trees Regressor
* HistGradientBoosting Regressor

The deployed application currently loads the **Extra Trees** model.

## 📈 Evaluation Metrics

* **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
* **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more heavily.
* **R² Score:** Measures how much variation in the target prices is explained by the model relative to a mean-prediction baseline.

## 📁 Project Structure

```text
house-price-project/
└── house_price_prediction_project/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    └── models/
        └── house_price_prediction_pipeline.joblib
```

## ⚙️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Megha2407/house-price-project.git
cd house-price-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r house_price_prediction_project/requirements.txt
```

### 5. Run the application

```bash
streamlit run house_price_prediction_project/app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## 📚 Reference

* [HTML Anchor Element — MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a)

## 🔮 Future Enhancements

* Expand the dataset with more recent housing records.
* Improve model performance through additional feature engineering.
* Add interactive price and location visualizations.
* Incorporate more detailed locality information.
* Monitor prediction errors and retrain the model when needed.

## 👩‍💻 Author

**Megha Mithra B**

B.E. Computer Science and Engineering
Artificial Intelligence and Machine Learning

GitHub: [Megha2407](https://github.com/Megha2407)

## ⚠️ Disclaimer

This application provides estimated property prices based on patterns learned from the training dataset. Actual market prices may vary depending on location, property condition, market trends, and other factors. Predictions should not be considered official property valuations.
