# Music-Popularity-Predictor
This project predicts the popularity of songs based on their musical features using a Random Forest Regression model. By analyzing various attributes like energy, danceability, loudness, and acousticness, we aim to determine how these characteristics influence the popularity of tracks.

## Features
- **Prediction Model**: Random Forest Regression is used to predict a song's popularity.
- **Data Visualization**: The relationship between different musical features and popularity is analyzed using scatter plots and correlation matrices.
- **Feature Normalization**: Feature scaling is performed to ensure fair model training.

## Technologies Used
- **Python**: Main programming language.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib & Seaborn**: For data visualization and feature analysis.
- **scikit-learn**: For building the Random Forest model and evaluating performance.

## Dataset
The dataset used in this project is a collection of songs with various musical attributes, such as:
- Energy
- Valence
- Danceability
- Loudness
- Acousticness
- Tempo
- Speechiness
- Liveness
- Popularity (Target variable)

The dataset is loaded from a local file named `Spotify_data.csv`.

## Project Overview

1. **Data Preprocessing**:  
   - The dataset is cleaned by removing unnecessary columns (like unnamed indices).
   - Exploratory Data Analysis (EDA) is performed to visualize the relationship between features and popularity using scatter plots and histograms.
   - A heatmap of the correlation matrix helps us understand the relationships between different numerical features.

2. **Model Training**:  
   - The features are normalized using `StandardScaler`.
   - A `RandomForestRegressor` is trained on the dataset using various musical features as inputs, with popularity as the target variable.
   - Hyperparameter tuning is performed using `GridSearchCV` to find the best combination of parameters for the model.

3. **Model Evaluation**:  
   - After training, the model is evaluated using metrics like Mean Squared Error (MSE) and R-squared (R²).
   - The model's performance is visualized using an "Actual vs Predicted" scatter plot.
