import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.metrics import r2_score
import time


MODEL_VAULT_DIR = "./model_vault"

def run_automatic_training():
    print('RUNN AUTOMATICS')
    time.sleep(5)

def run_retraining(data, target_var, selected_vars, algorithm):
    # Assuming the target variable is the last column and the rest are features
    X = data[selected_vars].values
    y = data[target_var].values

    # Splitting the data (you might want to adjust the test size or use other splitting methods)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model based on the selected algorithm.

    if algorithm == "Auto (Optimal model based on Adj-R2)":
        models = [LinearRegression, Lasso, Ridge, ElasticNet]

        mm = []
        r2_scores = []
        adj_r2_scores = []

        for m in models:
            m = m()
            m.fit(X_train, y_train)

            # Compute R2
            r2 = r2_score(y_test, y_pred)
            
            # Compute Adjusted R2
            n = X_test.shape[0]  # Number of observations
            p = X_test.shape[1]  # Number of predictors
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
            mm.append(m)
            r2_scores.append(r2)
            adj_r2_scores.append(adj_r2)
        
        max_idx = adj_r2_scores.index(max(adj_r2_scores))

        weights = mm[max_idx].coef_
        y_intercept = mm[max_idx].intercept_

        return weights, y_intercept, mm[max_idx], r2_scores[max_idx], adj_r2_scores[max_idx]
    
    else:
        if algorithm == "Linear Regression":
            model = LinearRegression()
        elif algorithm == "Lasso Regression":
            model = Lasso()
        elif algorithm == "Ridge Regression":
            model = Ridge()
        elif algorithm == "Elastic Regression":
            model = ElasticNet()
        # Additional algorithms can be added as needed.
        else:
            raise ValueError("Unknown algorithm selected!")
        
        model.fit(X_train, y_train)
        
        # Get the model's predictions on the test set
        y_pred = model.predict(X_test)

        # Compute R2
        r2 = r2_score(y_test, y_pred)
        
        # Compute Adjusted R2
        n = X_test.shape[0]  # Number of observations
        p = X_test.shape[1]  # Number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Save the model's weights
        weights = model.coef_
        y_intercept = model.intercept_

        return weights, y_intercept, model, r2, adj_r2


def get_next_id(directory):
    max_id = 0
    for file in os.listdir(directory):
        if file.startswith("SIO-"):
            try:
                file_id = int(file.split("-")[1])
                max_id = max(max_id, file_id)
            except ValueError:
                pass
    return max_id + 1

def save_model_and_config(model, config, algorithm):
    # 5. Determine the filename based on the specified format.
    if not os.path.exists(MODEL_VAULT_DIR):
        os.mkdir(MODEL_VAULT_DIR)

    next_id = get_next_id(MODEL_VAULT_DIR)
    model_algo_name = algorithm.replace(" ", "_")
    model_id = str(next_id).zfill(9)
    filename = f"SIO-{model_id}-{model_algo_name}"
    filepath = os.path.join(MODEL_VAULT_DIR, f'{filename}.json')
    
    # Save the trained model (this will save the model structure along with weights)
    joblib.dump(model, os.path.join(MODEL_VAULT_DIR, f'{filename}.pkl'))
    
    # 6. Save the JSON file.
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

    return filename, model_id