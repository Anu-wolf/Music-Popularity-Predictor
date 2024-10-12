#music popularity predictor
import pandas as pd

spotify_data = pd.read_csv("Spotify_data.csv")
print(spotify_data.head())

#drop the un-named coloumn
spotify_data.drop(columns=['Unnamed: 0'], inplace=True)

#printing all the columns (to check)
print(spotify_data.info())

#relationship between music and its popularity
import matplotlib.pyplot as plt
import seaborn as sns
features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness']
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=spotify_data, x=feature, y='Popularity')
    orint(plt.title(f'Popularity vs {feature}'))
    print(plt.show())
    numeric_columns = spotify_data.select_dtypes(include=['float64', 'int64']).columns
    numeric_data = spotify_data[numeric_columns]

    corr_matrix = numeric_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    print(plt.title('Correlation Matrix'))
    print(plt.show())
    # louder and more danceable tracks tend to be more popular

#distribution of data
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.histplot(spotify_data[feature], kde=True)
    print(plt.title(f'Distribution of {feature}'))
    print(plt.show())


#to train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# select the features and target variable
features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness', 'Tempo', 'Speechiness', 'Liveness']
X = spotify_data[features]
y = spotify_data['Popularity']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, refit=True, verbose=2, cv=5)
grid_search_rf.fit(X_train_scaled, y_train)
best_params_rf = grid_search_rf.best_params_
best_rf_model = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_scaled)

# make predictions
y_pred_best_rf = best_rf_model.predict(X_test_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_rf, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
print(plt.xlabel('Actual Popularity'))
print(plt.ylabel('Predicted Popularity'))
print(plt.title('Actual vs Predicted Popularity (Best Random Forest Model)'))
print(plt.show())
