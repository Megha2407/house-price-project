# 🏡 Indian House Price Prediction

A Machine Learning web application that predicts the estimated price of residential properties in India based on property features such as location, property type, area, BHK, bathrooms, floor details, furnishing, and nearby facilities.

The project includes data preprocessing, feature engineering, model evaluation, and an interactive web application built with Streamlit.

## 🚀 Live Demo

**Try the application:**
[House Price Prediction – Live App](https://house-price-project-69eevgslroucvss3xbc.streamlit.app/)

## 📌 Project Overview

House prices vary depending on location, property size, property type, age, and other factors. This project uses historical housing data to train a regression model that estimates a property's price from the details provided by the user.

### Objectives

* Analyze and preprocess housing data.
* Create useful features for prediction.
* Train and compare regression models.
* Evaluate model performance using suitable regression metrics.
* Build an interactive prediction interface.
* Deploy the application online using Streamlit Community Cloud.

## ✨ Features

* Enter property details through a user-friendly interface.
* Select the city, locality type, and property type.
* Specify BHK, bathrooms, and property area.
* Provide additional property characteristics.
* Get an estimated property price.
* Access the application through a public web link.

## 🧰 Tech Stack

| Technology                | Purpose                                 |
| ------------------------- | --------------------------------------- |
| Python                    | Programming language                    |
| Pandas & NumPy            | Data processing                         |
| Scikit-learn              | Machine Learning and preprocessing      |
| Matplotlib & Seaborn      | Exploratory data analysis               |
| Streamlit                 | Web application                         |
| Joblib                    | Saving and loading the trained model    |
| Git & GitHub              | Version control and source code hosting |
| Streamlit Community Cloud | Deployment                              |

## 🤖 Machine Learning

This project uses a regression approach to estimate house prices.

The model-comparison process included:

* HistGradientBoosting Regressor
* Extra Trees Regressor
* Random Forest Regressor

The deployed application currently loads the **Extra Trees** model.

Model performance is evaluated using:

* **MAE:** Mean Absolute Error
* **RMSE:** Root Mean Squared Error
* **R²:** Coefficient of Determination

These metrics measure prediction error and how well the model explains variation in the target values. R² is not classification accuracy.

## 📊 Dataset

The project uses housing data containing property characteristics and the target variable `Price_INR_Lakhs`.

Features include:

* City and locality type
* Property type
* Number of bedrooms (BHK) and bathrooms
* Super area and carpet area
* Floor number and total floors
* Property age and furnishing status
* Parking, lift, and gated community availability
* Distance to metro and city centre

The target variable is **`Price_INR_Lakhs`**, representing the property price in lakhs of Indian rupees.

> The dataset is not included in this repository by default. Ensure you have the right to share any dataset before publishing it.

## 📁 Project Structure

```text
house-price-project/
└── house_price_prediction_project/
    ├── app.py
    ├── train_models.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    └── models/
        └── house_price_prediction_pipeline.joblib
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Megha2407/house-price-project.git
cd house-price-project/house_price_prediction_project
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## ☁️ Deployment

The application is deployed using Streamlit Community Cloud and connected to this GitHub repository.

The deployed app loads the saved model pipeline and uses it to generate predictions from user-provided property details.

## 🔮 Future Improvements

* Add more housing data from different Indian cities.
* Improve model performance through further feature engineering and tuning.
* Display comparable properties and price ranges.
* Add interactive visualizations of housing trends.
* Improve model monitoring and retraining workflows.

## 👩‍💻 Author

**Megha Mithra B**

B.E. Computer Science and Engineering
Artificial Intelligence & Machine Learning

GitHub: [@Megha2407](https://github.com/Megha2407)

---

*This project was developed for learning and educational purposes. Predictions are estimates and should not be treated as official property valuations or financial advice.*
