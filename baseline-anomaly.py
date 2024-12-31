import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_curve, average_precision_score


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

    # model = IsolationForest(random_state=42, contamination=0.2, n_estimators=150)
    # model.fit(X_train)
    #
    # scores = model.decision_function(X_test)
    # preds = model.predict(X_test)
    #
    #


    h_contamination = [0.01, 0.05, 0.1, 0.2]
    h_n_estimators = [50, 100, 150, 250]

    grid_search_params = [(c, ne) for c in h_contamination for ne in h_n_estimators]

    h_metrics = []
    for index, (c, ne) in enumerate(grid_search_params):
        model = IsolationForest(contamination=c, n_estimators=ne, random_state=42)

        model.fit(X_train)

        scores = model.decision_function(X_test)
        y_pred = pd.Series(model.predict(X_test)).replace([-1, 1], [1, 0])

        print(f'min score = {np.min(scores)} max score = {np.max(scores)}')

        precision, recall, thresholds = precision_recall_curve(y_test, -scores)
        ap = average_precision_score(y_test, -scores)
        cm = confusion_matrix(y_test, y_pred)

        # 5. Plot the PR curve
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label=f'AP = {ap:.2f}, ne = {ne}, c = {c}', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (IsolationForest)')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(f'./figs/if-AP-cure-{dataset_name}-hp-idx-{index}.svg', bbox_inches='tight', dpi=300, format='svg')

        print("Average Precision (AP):", ap)

        # acc = accuracy_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        #
        # print(f'*** c = {c} , ne = {ne}')
        # print(f'acc= {acc}, f1 = {f1}')

        h_metrics.append(ap)

    best_h_idx = np.argmax(h_metrics)

    print('*********')
    print(f'best h param with f1 = {h_metrics[best_h_idx]} is {grid_search_params[best_h_idx]}')

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--ds", type=str, default='dapt20')

    args = parser.parse_args()

    main(args.ds)
