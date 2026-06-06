# Prediction of Agriculture Crop Production in India

## Project Overview

Agriculture plays a vital role in the Indian economy. Accurate prediction of crop production helps farmers, agricultural planners, and policymakers make informed decisions. This project uses Machine Learning techniques to predict crop yield based on crop type, state, and cultivation-related costs.

---

## Objective

To develop a Machine Learning model that predicts agricultural crop production/yield in India using historical crop and cultivation data.

---

## Dataset Description

The dataset contains agricultural information from different states in India.

### Features

* Crop
* State
* Cost of Cultivation (A2+FL)
* Cost of Cultivation (C2)
* Cost of Production (C2)

### Target Variable

* Yield (Quintal/Hectare)

### Dataset Statistics

* Total Records: 49
* Total Features: 5
* Missing Values: 0

---

## Data Preprocessing

* Checked and verified missing values.
* Encoded categorical features (Crop and State) using Label Encoding.
* Prepared data for machine learning model training.

---

## Machine Learning Model

### Algorithm Used

Random Forest Regressor

### Why Random Forest?

* Handles nonlinear relationships effectively.
* Works well on small and medium-sized datasets.
* Reduces overfitting through ensemble learning.

---

## Model Performance

| Metric   | Value  |
| -------- | ------ |
| MAE      | 28.44  |
| RMSE     | 69.66  |
| R² Score | 0.9463 |

The model successfully explained approximately 94.63% of the variance in crop yield.

---

## Visualizations

Generated visualizations include:

* Average Yield by Crop
* Average Yield by State
* Actual vs Predicted Yield

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib

---

## Conclusion

The developed machine learning model effectively predicts agricultural crop yield using cultivation costs and crop-related information. The project demonstrates the practical application of machine learning in agriculture and production forecasting.

---

## Future Scope

* Use larger datasets covering more years and regions.
* Integrate weather and rainfall data.
* Develop a real-time crop yield prediction dashboard.
* Compare advanced models such as XGBoost and Gradient Boosting.
