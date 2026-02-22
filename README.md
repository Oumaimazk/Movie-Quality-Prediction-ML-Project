# Movie Quality Prediction Project 🎬

## Overview
This project, focuses on predicting whether a movie will be classified as "Good" or "Bad" based on various features such as popularity, vote count, genres, and release information. The project compares standard machine learning implementations using **Scikit-Learn** with a **custom Logistic Regression model built from scratch**.

## 📊 Dataset
The project uses a dataset containing approximately **10,000 movies** with the following information:
- **ID & Title**: Identifiers for each movie.
- **Genre**: Categorical data (Action, Adventure, Drama, etc.).
- **Original Language**: The language in which the movie was produced.
- **Popularity**: A metric representing the movie's current popularity.
- **Vote Average**: The target metric used to define "Good" (≥ 6.5) or "Bad" (< 6.5).
- **Vote Count**: Number of votes received.
- **Release Date**: Used for feature engineering (Year, Month, Day).

## 🛠️ Tech Stack
- **Languages**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Logistic Regression, Decision Tree, GridSearchCV)
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebooks

## 🚀 Key Features
### 1. Data Preprocessing & Cleaning
- Handling missing values in genres and overviews.
- **Multi-Label Encoding**: Converting multiple genres per movie into binary features.
- **Label Encoding**: Categorizing languages.
- **Feature Engineering**: Extracting temporal features from release dates.

### 2. Model Implementations
- **Scikit-Learn Approach**: Utilizing standard library models for baseline performance (Accuracy ~72%).
- **Scratch Implementation**: A robust custom Logistic Regression class featuring:
  - L1 (Lasso) and L2 (Ridge) Regularization.
  - Mini-batch Stochastic Gradient Descent (SGD).
  - Early Stopping to prevent overfitting.
  - Numerical stability via clipped sigmoid functions.

### 3. Evaluation & Optimization
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- **Hyperparameter Tuning**: Using `GridSearchCV` and `RandomizedSearchCV` to find optimal learning rates, regularization strengths, and batch sizes.

## 📂 Project Structure
- `dataset.csv`: The raw movie data.
- `scikit-learn.ipynb`: Exploratory data analysis and Scikit-Learn model implementation.
- `scratch-model.ipynb`: Custom Logistic Regression implementation and evaluation.
- `notes_des_films.pptx`: Project presentation and summary of findings.

## 🏁 How to Run
1. Ensure you have the following packages installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn tqdm
   ```
2. Open the Jupyter Notebooks:
   ```bash
   jupyter notebook scikit-learn.ipynb
   ```
3. Run all cells to see the data processing and model results.

## 📈 Results
The models achieved an accuracy of approximately **70-73%** in predicting movie quality, with Logistic Regression slightly outperforming the Decision Tree. The custom "from-scratch" implementation showed comparable results to the library-based version after hyperparameter tuning.

---
*Developed as part of a Data Science / Machine Learning project.*
