/*************************************************************************
*********************** MT-EDD Module  ***********************************
*************************************************************************/


// Python implementation for MT-EDD Algorithm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, specificity_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

# Data Preprocessing (3.1.1)
def preprocess_data(df, p4_features=None):
    # Handle missing values (mean imputation)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Outlier removal (example using IQR)
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # One-hot encode categorical features (if any)
    df = pd.get_dummies(df, drop_first=True)

    if p4_features is not None:
        df = pd.concat([df, p4_features], axis=1)

    # Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def select_features(X, y, k=20):  # k is the number of features to select
    mutual_info = mutual_info_classif(X, y)
    selector = SelectKBest(score_func=lambda X, y: mutual_info, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector

# Model Training (3.1.2)
def train_models(X_train, y_train, X_val, y_val):
    # Resampling
    smote = SMOTE(random_state=42)
    rus = RandomUnderSampler(random_state=42)

    # Identify classes to oversample and undersample based on counts
    class_counts = y_train.value_counts()
    minority_classes = class_counts[class_counts < class_counts.median()].index
    majority_classes = class_counts[class_counts > class_counts.median()].index
    
    for class_label in minority_classes:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train[y_train == class_label], y_train[y_train == class_label])
        X_train = pd.concat([X_train, X_train_resampled])
        y_train = pd.concat([y_train, y_train_resampled])

    for class_label in majority_classes:
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train[y_train == class_label], y_train[y_train == class_label])
        X_train = X_train[~y_train.isin([class_label])]
        y_train = y_train[~y_train.isin([class_label])]
        X_train = pd.concat([X_train, X_train_resampled])
        y_train = pd.concat([y_train, y_train_resampled])

    # Hyperparameter tuning (RandomizedSearchCV)
    rf_param_grid = {'n_estimators': [200, 500, 800], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'criterion': ['gini', 'entropy']}
    dt_param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'criterion': ['gini', 'entropy']}
    xgb_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 6, 8], 'learning_rate': [0.01, 0.1, 0.3], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8]}

    rf_random = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    dt_random = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    xgb_random = RandomizedSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), xgb_param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1)

    rf_random.fit(X_train, y_train)
    dt_random.fit(X_train, y_train)
    xgb_random.fit(X_train, y_train)

    rf = rf_random.best_estimator_
    dt = dt_random.best_estimator_
    xgb_model = xgb_random.best_estimator_

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42) # Example MLP
    mlp.fit(X_train, y_train)
    return rf, dt, xgb_model, mlp

def predict_ensemble(X, ensemble):
    predictions = []
    confidences = []
    for classifier in ensemble:
        pred = classifier.predict(X)
        try:
            conf = classifier.predict_proba(X)
        except AttributeError:  # DecisionTreeClassifier doesn't have predict_proba
            conf = np.zeros((len(X), 2)) # Dummy confidence as it is not used for DT in the paper
        predictions.append(pred)
        confidences.append(conf)
    return np.array(predictions).T, np.array(confidences).transpose(1,0,2)

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = specificity_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fdr = fp / (fp + tp) if (fp + tp) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, precision, recall, f1, specificity, npv, fpr, fdr, fnr, mcc

# Example usage (replace with your actual data loading and paths)
try:
    df = pd.read_csv("your_cic_iot_dataset.csv")
except FileNotFoundError:
    print("Error: CICIoT2023 dataset file not found.")
    exit()

# Separate features and labels
X = df.drop("label_column_name", axis=1) # Replace "label_column_name"
y = df["label_column_name"]

X_selected, selector = select_features(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# Train models
rf, dt, xgb_model, mlp = train_models(X_train, y_train, X_val, y_val)
ensemble = [rf, dt, xgb_model]
