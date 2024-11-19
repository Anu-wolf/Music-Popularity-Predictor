# Music-Popularity-Predictor #
This project predicts the popularity of songs based on their musical features using a Random Forest Regression model, allowing music platforms and developers to understand what makes a song popular. By analysing various attributes like energy, danceability, loudness, and acoustics, we aim to determine how these characteristics influence the popularity of tracks.
This tool can be integrated into web apps to help music platforms make data-driven recommendations, evaluate upcoming tracks, or understand trends in music preference.

Features
- Prediction Model: Random Forest Regression predicts a song's popularity.
- Data Visualization: The relationship between different musical features and popularity is analysed using scatter plots and correlation matrices.
- Feature Normalization: Feature scaling is performed to ensure fair model training.

 Technologies Used
- Python: Main programming language.
- Pandas: For data manipulation and preprocessing.
- Matplotlib & Seaborn: This is for data visualization and feature analysis.
- scikit-learn: This is used to build the Random Forest model and evaluate performance.
Dataset
The dataset used in this project is a collection of songs with various musical attributes, such as:
- Energy
- Valence
- Danceability
- Loudness
- Acoustics
- Tempo
- Speechiness
- Liveness
- Popularity (Target variable)
The dataset is loaded from a local file named `Spotify_data.csv`.

Applications
1.	Music Streaming Platforms
Predict which new releases are likely to become popular.
2.	Music Labels
Evaluate the potential success of songs before release.
3.	Playlist Curators
Create dynamic playlists with high-potential songs.
4.	Data-Driven Recommendation Engines
Enhance song recommendations by analyzing feature-popularity relationships.
How the project is made:
1.	Data Cleaning & Exploration
Dropping Unnecessary Columns: Remove irrelevant columns (like unnamed columns).
Scatter Plots: Visualize the relationship between features and popularity.
Correlation Matrix: Analyze correlations to understand which features impact popularity the most.
Distribution Plots: Check feature distributions to detect patterns.
2.	Data Preparation
Train-Test Split: Divide the data into training and testing sets.
Normalization: Use StandardScaler to normalize features for better model performance.
3.	Model Selection & Hyperparameter Tuning
Use Random Forest Regressor to predict the popularity score of songs.
Apply GridSearchCV for hyperparameter tuning to find the optimal model configuration.
4.	Model Evaluation
Actual vs Predicted Plot: Compare the predicted and actual popularity scores to evaluate model performance using metrics like R² Score and Mean Squared Error (MSE).

How It Solves the Problem
Music streaming platforms and record labels often need to predict which tracks will perform well. This project provides insights by analysing key audio features and how they impact song popularity. The Random Forest Regressor model enables developers to make accurate predictions, helping businesses make data-driven decisions about song recommendations and marketing.
Integration into a Web Project
To integrate this model into a web application, follow these steps:
1.	Export the Trained Model
After training the model, save it as a serialized file (pickle format) to be loaded in the web backend (save it as a .pkl file).
2.	Create a Web Backend with Flask
Install Flask (pip install flask) 
Create a server.py file to load the model and serve predictions through a web API.
3.	Frontend Integration
In the frontend(HTML/JavaScript), make an AJAX call to the predict API and receive predictions.
4.	Run the Web Application
Run the Flask Server with:
“python server.py”
Open your browser and go to http://127.0.0.1:5000 to interact with the web interface.
Future Work
	Explore more advanced models like XGBoost or Neural Networks for better predictions.
	Add more features to the dataset (e.g., genre, release date).
	Integrate a visualization dashboard to explore feature-popularity trends.

