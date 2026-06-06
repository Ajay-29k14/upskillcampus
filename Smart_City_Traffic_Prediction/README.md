# Smart City Traffic Prediction

## Project Overview

The Smart City Traffic Prediction project aims to forecast vehicle traffic at different city junctions using Machine Learning techniques. Accurate traffic prediction helps city planners optimize traffic management, reduce congestion, and improve transportation efficiency.

---

## Objective

To predict the number of vehicles passing through city junctions based on historical traffic data and time-related features.

---

## Dataset Description

### Training Dataset

* File: `train_aWnotuB.csv`
* Records: 48,120
* Features:

  * DateTime
  * Junction
  * Vehicles (Target Variable)
  * ID

### Test Dataset

* File: `test_BdBKkAj.csv`
* Records: 11,808
* Features:

  * DateTime
  * Junction
  * ID

---

## Data Preprocessing

* Checked for missing values.
* Converted DateTime into datetime format.
* Extracted:

  * Year
  * Month
  * Day
  * Hour
  * DayOfWeek
  * Weekend

---

## Machine Learning Model

Model Used:

* Random Forest Regressor

Reason:

* Handles non-linear relationships effectively.
* Provides high prediction accuracy.
* Robust against overfitting.

---

## Model Performance

| Metric   | Value |
| -------- | ----- |
| MAE      | 2.39  |
| RMSE     | 3.55  |
| R² Score | 0.969 |

The model successfully explained approximately 96.9% of the variance in traffic volume.

---

## Visualizations

Generated:

* Traffic by Hour
* Traffic by Junction

These visualizations help understand traffic trends across time and locations.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

---

## Conclusion

The developed model successfully predicts traffic volume with high accuracy. The project demonstrates the practical application of Machine Learning in smart city traffic management and urban planning.

---

## Future Scope

* Incorporate weather data.
* Use advanced forecasting models such as XGBoost and LSTM.
* Develop a real-time traffic monitoring dashboard.
