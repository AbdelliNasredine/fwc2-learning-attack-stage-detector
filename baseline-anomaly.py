import os
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

def train_eval(dataset_name: str = 'dapt20'):
    log_dir = f'./results/{datetime.now().strftime("%Y-%m-%d")}/baselines/{dataset_name}'
    os.makedirs(log_dir, exist_ok=True)

    train_set = pd.read_csv(f'./data/{dataset_name}/anomaly/train.csv')
    test_set = pd.read_csv(f'./data/{dataset_name}/anomaly/test.csv')

    X_train, y_train = train_set.drop(columns=['label']), train_set['label']
    X_test, y_test = test_set.drop(columns=['label']), test_set['label']
    y_test = y_test.apply(lambda x: 1 if x != 'benign' else 0)

    models = {
        'LOF': LOF(n_neighbors=20, algorithm='auto', metric='minkowski', p=2, metric_params=None, n_jobs=-1),
        'IForest': IForest(n_estimators=100, random_state=42, n_jobs=-1),
        # 'OCSVM': OCSVM(kernel='linear', nu=0.05, max_iter=-1),
        'AutoEncoder': AutoEncoder(hidden_neuron_list=[64, 32],preprocessing=True, dropout_rate=0.0, random_state=42),
    }

    log = {
        'model': [],
        'acc': [],
        'recall': [],
        'precision': [],
        'f1': [],
        'roc_auc': [],
        'average_precision': []
    }
    for index , (name, model) in enumerate(models.items()):
        model.fit(X_train)

        y_pred = model.predict(X_test)
        y_scores = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)
        average_precision = average_precision_score(y_test, y_scores)

        print(f'result of {name} on ')
        print(f'acc = {acc}, recall = {recall}, precision = {precision}, f1 = {f1}, roc_auc = {roc_auc}, average_precision = {average_precision}')

        log['model'].append(name)
        log['acc'].append(acc)
        log['recall'].append(recall)
        log['precision'].append(precision)
        log['f1'].append(f1)
        log['roc_auc'].append(roc_auc)
        log['average_precision'].append(average_precision)

    pd.DataFrame(log).to_csv(f'{log_dir}/log.csv', index=False)

if __name__ == "__main__":
    # datasets = ['scvic21', 'unraveled']
    datasets = ['scvic21', 'dapt20', 'mscad']
    for dataset in datasets:
        train_eval(dataset)
