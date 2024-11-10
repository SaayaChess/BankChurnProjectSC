import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from schemas import InferenceInput, HyperInput, InferenceOutput

SEED = 1234
CAT_STR = ['Geography', 'Gender']
BIN_INT = ['HasCrCard', 'IsActiveMember']
NUMBERS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']


def make_train_lr_pipeline():
    preproc = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown='ignore',drop='first',
                                                                    sparse_output=False), CAT_STR),
                                              ("num", StandardScaler(), NUMBERS)],
                                remainder= 'passthrough')
    lr = LogisticRegression(random_state=SEED, max_iter=5000, class_weight='balanced')
    return Pipeline(steps=[("prep", preproc), ("model", lr)])


def make_train_rf_pipeline():
    preproc = ColumnTransformer(transformers=[("cat", OrdinalEncoder(handle_unknown='use_encoded_value',
                                                                     unknown_value=-1), CAT_STR)],
                                remainder= 'passthrough')
    rf = RandomForestClassifier(random_state=SEED, class_weight='balanced')
    return Pipeline(steps=[("prep", preproc), ("model", rf)])


def train_models(data: pd.DataFrame, model_class, params_dict: dict):
    if model_class == 'lr':
        pipe = make_train_lr_pipeline()
    elif model_class == 'rf':
        pipe = make_train_rf_pipeline()
    else:
        raise ValueError("Model must be 'lr' or 'rf'")
    params_dict = {f'model__{k}':v for k, v in params_dict.items() }
    model_cv = GridSearchCV(pipe, params_dict, cv=5, n_jobs=-1, scoring="f1_weighted")
    y = data["Exited"]
    X = data.drop("Exited", axis=1)
    model_cv.fit(X, y)
    result = model_cv.best_score_
    best_params = model_cv.best_params_
    best_estimator = model_cv.best_estimator_
    return result, best_params, best_estimator


def create_train(data: list[list]):
    cols_full = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                  'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
    data = pd.DataFrame(data, columns=cols_full)
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    return data


def create_inferdf(input: InferenceInput):
    data = pd.DataFrame([[input.CreditScore,
                          input.Geography,
                          input.Gender,
                          input.Age,
                          input.Tenure,
                          input.Balance,
                          input.NumOfProducts,
                          input.HasCrCard,
                          input.IsActiveMember,
                          input.EstimatedSalary]], columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                  'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
    return data


def hyprer2dict(hyper: list[HyperInput]) -> dict:
    out_dict = {}
    for hyp in hyper:
        if hyp.log_scale:
            value = np.logspace(hyp.min_value, hyp.max_value, hyp.steps)
        else:
            value = np.linspace(hyp.min_value, hyp.max_value, hyp.steps)
        if hyp.is_int:
            value = list(map(int, value))
        out_dict[hyp.name] = value
    return out_dict


def inference(model, data):
    result = {"Exited_rate": model.predict_proba(data)[0][1]}
    return InferenceOutput(**result)
