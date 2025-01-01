import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from argparse import ArgumentParser

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, auc, \
    precision_recall_curve, average_precision_score


class DS(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AE(pl.LightningModule):
    def __init__(self, input_size):
        super(AE, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 35),
            torch.nn.ReLU(),
            torch.nn.Linear(35, 20),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(20, 35),
            torch.nn.ReLU(),
            torch.nn.Linear(35, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, input_size),
            torch.nn.Sigmoid()
        )

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch

        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)

        self.log('test_loss', loss)

        return loss


def train_eval(dataset_name: str = 'dapt20', model: str = 'IF'):
    print(f'** EXPERIMENT WITH {dataset_name}')

    train_set = pd.read_csv(f'./data/{dataset_name}/anomaly/train.csv')
    test_set = pd.read_csv(f'./data/{dataset_name}/anomaly/test.csv')

    in_dim = train_set.shape[1] - 1

    scaler = StandardScaler()
    X_train, y_train = train_set.drop(columns=['label']), train_set['label']
    X_test, y_test = test_set.drop(columns=['label']), test_set['label']

    y_test = y_test.apply(lambda x: 1 if x != 'benign' else 0)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    auc_roc, auc_pr = 0, 0

    if model == 'IF':
        model = IsolationForest(random_state=42)

        model.fit(X_train)

        scores = (-1.0) * model.decision_function(X_test)

        auc_roc = roc_auc_score(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        auc_pr = auc(recall, precision)

    if model == 'AE':
        train_loader = DataLoader(DS(X_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(DS(X_test), batch_size=32)

        model = AE(input_size=in_dim)
        trainer = pl.Trainer(max_epochs=10, enable_progress_bar=True)
        trainer.fit(model, train_loader)
        trainer.test(model, test_loader)

        model.eval()
        scores = []
        with torch.no_grad():
            for x in test_loader:
                x_hat = model(x)
                batch_errors = torch.mean((x_hat - x) ** 2, dim=1).cpu().numpy()
                scores.extend(batch_errors)
        scores = np.array(scores)

        auc_roc = roc_auc_score(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        auc_pr = auc(recall, precision)

    return auc_roc, auc_pr


if __name__ == "__main__":
    # datasets = ['dapt20', 'mscad', 'scvic21']
    datasets = ['dapt20']
    models = ['IF', 'AE']

    log = {
        'dataset': [],
        'model': [],
        'auc_roc': [],
        'auc_pr': [],
    }
    for dataset in datasets:
        for model in models:
            auc_roc, auc_pr = train_eval(dataset, model=model)
            log['model'].append(model)
            log['dataset'].append(dataset)
            log['auc_roc'].append(auc_roc)
            log['auc_pr'].append(auc_pr)

    log_dir = f'./results/{datetime.now().strftime("%Y-%m-%d")}-anomaly-baseline/'

    os.makedirs(log_dir, exist_ok=True)

    pd.DataFrame(log).to_csv(f'{log_dir}/log.csv', index=False)
