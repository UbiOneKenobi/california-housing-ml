# California Housing ML

End-to-end machine learning project for predicting median house values in California districts using **Python** and **Scikit-Learn**.

The project follows the workflow presented in *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron, while implementing and commenting the complete pipeline in a standalone Python script.

## Project Overview

The goal is to predict the **median house value** of a California district using census information such as:

- median income
- housing median age
- total rooms and bedrooms
- population
- households
- latitude and longitude
- proximity to the ocean

This is a **supervised regression problem**.

The project covers the full machine learning workflow, from loading and exploring the dataset to preprocessing, model selection, hyperparameter tuning, final evaluation, and model persistence.

## Workflow

The script implements the following steps:

1. **Data loading**
   - Automatically downloads the California Housing dataset if it is not available locally.
   - Extracts the dataset and loads it into a Pandas DataFrame.

2. **Exploratory Data Analysis**
   - Dataset inspection with Pandas.
   - Histograms of numerical features.
   - Geographic visualization.
   - Correlation analysis.
   - Analysis of the relationship between median income and house values.

3. **Train/Test Split**
   - Exploration of different splitting approaches.
   - Random splitting.
   - Hash-based stable splitting.
   - Stratified sampling based on median income categories.

4. **Feature Engineering**
   - Rooms per household.
   - Population per household.
   - Bedroom-to-room ratio.
   - Log transformations for skewed features.
   - Geographic similarity features based on clustering.

5. **Data Preprocessing**
   - Missing-value imputation with `SimpleImputer`.
   - Numerical feature scaling with `StandardScaler`.
   - One-hot encoding for categorical variables.
   - Custom transformations with `FunctionTransformer`.
   - Separate preprocessing pipelines combined with `ColumnTransformer`.

6. **Geographical Feature Engineering**
   - `KMeans` is used to identify geographic clusters.
   - An RBF kernel converts distance from cluster centers into similarity features.

7. **Model Training**
   - Linear Regression.
   - Decision Tree Regression.
   - Random Forest Regression.

8. **Model Evaluation**
   - Root Mean Squared Error (RMSE).
   - Cross-validation to estimate generalization performance.
   - Comparison between training and validation performance to identify underfitting and overfitting.

9. **Hyperparameter Tuning**
   - `GridSearchCV`.
   - `RandomizedSearchCV`.
   - Optimization of:
     - number of geographic clusters
     - Random Forest `max_features`

10. **Final Evaluation**
    - The best pipeline is evaluated on the untouched test set.

11. **Model Persistence**
    - The final trained model is saved locally using `joblib`.

## Machine Learning Pipeline

The final model combines preprocessing and prediction into a single Scikit-Learn pipeline:

```text
Raw California Housing Data
            │
            ▼
    Train/Test Split
            │
            ▼
     Stratified Sampling
            │
            ▼
      Preprocessing
            │
            ├── Missing-value imputation
            ├── Feature ratios
            ├── Log transformations
            ├── Standardization
            ├── Geographic clustering
            └── One-hot encoding
            │
            ▼
    Random Forest Regressor
            │
            ▼
   Hyperparameter Search
            │
            ▼
       Final Model
            │
            ▼
    Test Set Evaluation
