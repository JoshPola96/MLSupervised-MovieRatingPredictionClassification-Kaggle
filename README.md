# Machine Learning Pipeline for ML Prediction and Classification

This repository showcases a comprehensive machine learning pipeline designed to handle both regression and classification tasks. It includes model training, hyperparameter tuning, evaluation, and visualization for a variety of algorithms.
link: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
## Project Overview

This project implements a complete machine learning workflow that applies both regression and classification techniques to a dataset, involving:

1. **Data Preprocessing**: Data imputation, scaling, encoding, and transformation.
2. **Model Selection**: Includes a diverse range of models, both for regression and classification tasks.
3. **Hyperparameter Tuning**: Automated through grid search for optimal parameter selection.
4. **Evaluation**: Models are evaluated using metrics like Mean Squared Error (MSE) for regression and accuracy for classification.
5. **Visualization**: Model performance is visualized through plots showing predicted vs actual values.

## Pipeline Components

### 1. **Data Preprocessing**
The pipeline uses `ColumnTransformer` and `Pipeline` to preprocess both numerical and categorical data. Techniques include:
- **Scaling**: `StandardScaler`
- **Encoding**: `OneHotEncoder`

The preprocessing step is crucial to prepare the dataset for the model training stage.

### 2. **Regression Models**
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Gradient Boosting Regressor**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **K-Nearest Neighbors Regressor**
- **Support Vector Regressor (SVR)**
- **Polynomial Regression**

Each model is tuned using `GridSearchCV` to find the best hyperparameters, and performance is measured using the Mean Squared Error (MSE).

#### Best Performing Regression Models:
- **Gradient Boosting Regressor**, **Decision Tree Regressor**, and **Random Forest Regressor** achieved the lowest MSE of 0.0001, indicating excellent performance in predicting the target variable.

#### Poor Performers:
- **Polynomial Regression** exhibited the worst performance with a high MSE, suggesting it is not suitable for this dataset.

### 3. **Classification Models**
- **Logistic Regression**
- **K-Nearest Neighbors Classifier**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

Accuracy is used as the evaluation metric, and each model is grid-searched for hyperparameter optimization.

#### Best Performing Classification Models:
- **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier** achieved perfect accuracy (1.0000), highlighting their strong performance.
- **K-Nearest Neighbors Classifier** also performed well with high accuracy (0.9979).

#### Slightly Lower Performer:
- **Support Vector Machine (SVM)** showed slightly lower accuracy compared to the top models but still achieved a strong result.

### 4. **Model Training and Hyperparameter Tuning**
The models were trained using `GridSearchCV` for hyperparameter optimization. Each model's best parameters were recorded and used to make predictions on the test set.

#### Example Tuning:
```python
classification_models = {
    'LogisticRegression': Pipeline(steps=[
        ('preprocessor', preprocessor_class),
        ('model', LogisticRegression(max_iter=1000))
    ]),
    # Other models...
}
classification_param_grids = {
    'LogisticRegression': {
        'model__C': [0.1, 1],
        'model__penalty': ['l2']
    },
    # Other parameter grids...
}
```

### 5. **Results**
#### Regression Models:
| Model                       | Mean Squared Error (MSE) | Best Parameters                                |
|------------------------------|--------------------------|------------------------------------------------|
| Linear Regression            | 0.0389                   | None (default settings used)                   |
| Ridge Regression             | 0.0389                   | {'model__alpha': 10}                           |
| Lasso Regression             | 0.0616                   | {'model__alpha': 0.1}                          |
| Gradient Boosting Regressor  | 0.0001                   | {'model__learning_rate': 0.1, 'model__n_estimators': 200, 'model__max_depth': 5} |
| Decision Tree Regressor      | 0.0001                   | {'model__max_depth': 20, 'model__min_samples_split': 5} |
| Random Forest Regressor      | 0.0001                   | {'model__max_depth': 20, 'model__n_estimators': 100} |
| K-Nearest Neighbors Regressor| 0.0087                   | {'model__n_neighbors': 3, 'model__weights': 'distance', 'model__metric': 'euclidean'} |
| Support Vector Regressor     | 0.0027                   | {'model__C': 10, 'model__kernel': 'rbf', 'model__gamma': 'scale'} |
| Polynomial Regression        | 225060754.9347           | {'poly__degree': 3}                            |

#### Classification Models:
| Model                       | Accuracy | Best Parameters                                  |
|------------------------------|----------|--------------------------------------------------|
| Logistic Regression          | 1.0000   | {'model__C': 1, 'model__penalty': 'l2'}          |
| K-Nearest Neighbors Classifier| 0.9979   | {'model__n_neighbors': 5, 'model__weights': 'uniform'} |
| Decision Tree Classifier     | 1.0000   | {'model__max_depth': None, 'model__min_samples_split': 5} |
| Random Forest Classifier     | 1.0000   | {'model__max_depth': None, 'model__n_estimators': 100} |
| Support Vector Machine (SVM) | 0.9992   | {'model__C': 1, 'model__kernel': 'rbf'}          |

### 6. **Visualizations**
Model performance for regression models is visualized by plotting actual vs. predicted values, helping assess the accuracy of predictions.

Example visualization for regression results:
```python
def plot_regression_results(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: Actual vs. Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    plt.show()
```

### 7. **Conclusion**
- **Best Regression Models**: Gradient Boosting Regressor, Decision Tree Regressor, and Random Forest Regressor all delivered outstanding performance, achieving the lowest MSE of 0.0001.
- **Best Classification Models**: Logistic Regression, Decision Tree Classifier, and Random Forest Classifier achieved perfect accuracy, showing exceptional performance in classification tasks.
- **Overall Observations**: Ensemble methods like Gradient Boosting and Random Forest outperformed other models in regression tasks, while traditional models like Logistic Regression and Decision Tree excelled in classification.
