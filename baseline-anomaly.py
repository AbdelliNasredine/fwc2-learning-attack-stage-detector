import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score


def main(dataset_name: str = 'dapt20'):
    print(f'** EXPERIMENT WITH {dataset_name}')

    train_set = pd.read_csv(f'./data/{dataset_name}/anomaly/train.csv')
    test_set = pd.read_csv(f'./data/{dataset_name}/anomaly/test.csv')

    scaler = StandardScaler()
    X_train, y_train = train_set.drop(columns=['label']), train_set['label']
    X_test, y_test = test_set.drop(columns=['label']), test_set['label']

    y_test = y_test.apply(lambda x: 1 if x != 'benign' else 0)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    h_contamination = [0.001, 0.01, 0.04, 0.05, 0.1, 0.2]
    h_n_estimators = [20, 50, 100, 150, 250]

    grid_search_params = [(c, ne) for c in h_contamination for ne in h_n_estimators]

    h_metric = []
    for index, (c, ne) in enumerate(grid_search_params):
        model = IsolationForest(contamination=c, n_estimators=ne, random_state=42)

        model.fit(X_train)

        y_pred = pd.Series(model.predict(X_test)).replace([-1, 1], [1, 0])

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')

        print(f'*** c = {c} , ne = {ne}')
        print(f'acc= {acc}, f1 = {f1}')

        h_metric.append(f1)

    best_h_idx = np.argmax(h_metric)

    print('*********')
    print(f'best h param with f1 = {h_metric[best_h_idx]} is {grid_search_params[best_h_idx]}')

if __name__ == "__main__":
    for ds in ['dapt20']:
        main(ds)
