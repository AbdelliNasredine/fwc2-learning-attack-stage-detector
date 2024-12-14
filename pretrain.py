import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.manifold import TSNE

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from fwc2.model import FWC2Pretrainer, FWC2V2, MLP
from fwc2.loss import NTXent
from fwc2.data import FWC2Dataset
from fwc2.utils import preprocess

SEED = 42


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def fwc2_pretrain(
        train_ds,
        val_ds,
        encoder_dims=[256, 256, 256, 256],
        projection_dims=[128, 128],
        corruption_rate=0.2,
        tau=1.0,
        max_epochs=200,
        batch_size=256,
        device=torch.device('cpu'),
):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    early_stopping = EarlyStopping(patience=5, delta=0.001)
    loss_fn = NTXent(tau)

    model = FWC2Pretrainer(
        input_dim=train_ds.shape[1],
        features_low=train_ds.features_low,
        features_high=train_ds.features_high,
        dims_hidden_encoder=encoder_dims,
        dims_hidden_head=projection_dims,
        corruption_rate=corruption_rate,
        dropout=0.1,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    model.to(device)

    train_loss_history = []
    val_loss_history = []

    print('Stating training...')

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()

            z_i, z_j = model(x)
            loss = loss_fn(z_i, z_j)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, y in val_loader:
                data = data.to(device)

                emb, emb_c = model(data)
                loss = loss_fn(emb, emb_c)

                val_loss += loss.item() * data.size(0)
            val_loss /= len(val_loader.dataset)
            val_loss_history.append(val_loss)

        print(
            f'epoch [{epoch}/{max_epochs}] - train loss: {train_loss_history[-1]:.5f}, val loss: {val_loss_history[-1]:.5f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch}')
            early_stopping.load_best_model(model)
            break

    history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
    }

    return model, history


def finetune(model, train_ds, val_ds, max_epochs: int, batch_size: int, device: torch.device):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    patience = 5
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    model.to(device)

    best_acc = 0.0
    best_model_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_p, _ = model(x)

            loss = criterion(y_p, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(y_p, 1)
            correct_train += (predicted == y).sum().item()
            total_train += y.size(0)

        train_loss /= len(train_loader)
        train_acc = correct_train / total_train

        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs, _ = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == y).sum().item()
                total_val += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"epoch [{epoch}/{max_epochs}], loss: [{train_loss:.4f},{val_loss:.4f}], acc: [{train_acc:.4f},{val_acc:.4f}]")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"early stopping triggered at epoch{epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def evaluate_model(model, test_ds, device: torch.device):
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)

            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    test_acc = correct / total
    f1 = f1_score(all_targets, all_predictions, average="weighted")
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return test_acc, f1, conf_matrix

def prepare_data(dataset_name='scvic-apt-2021'):
    df = pd.read_csv(f'./datasets/{dataset_name}/all.csv')
    df = preprocess(df)

    X, y = df.drop('label', axis=1), df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.125,
        random_state=SEED,
        stratify=y_train,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.fit_transform(X_val)
    X_test_scaled = scaler.fit_transform(X_test)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.fit_transform(y_val)
    y_test = label_encoder.fit_transform(y_test)

    train_ds = FWC2Dataset(X_train_scaled, y_train, columns=X.columns)
    val_ds = FWC2Dataset(X_val_scaled, y_val, columns=X.columns)
    test_ds = FWC2Dataset(X_test_scaled, y_test, columns=X.columns)

    return train_ds, val_ds, test_ds

def main():
    # data = load('dapt20', base_dir='datasets')
    # data = preprocess(data, pretraining=True)
    # X, y = data.drop('label', axis=1), data['label']
    # pretrain_x, rest_x, pretrain_y, rest_y = train_test_split(X, y, test_size=0.5, random_state=SEED, shuffle=True, stratify=y)
    # train_ds, val_ds, test_ds = prepare_pretrainig_data(pretrain_x, pretrain_y)

    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    encoder_dims = [256, 256, 256, 256]
    projection_dims = [128, 128]
    corruption_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    tau = 1.0
    batch_size = 256
    max_pretrain_epochs = 1000
    max_finetune_epochs = 100

    # datasets = {'dapt20': 5, 'scvic-apt-2021': 6, 'mscad': 6}
    datasets = {'dapt20': 5}
    for dataset, num_stages in datasets.items():
        print(f'Starting with {dataset}')
        os.makedirs(f'./output/{dataset}', exist_ok=True)

        train_ds, val_ds, test_ds = prepare_data(dataset)

        scores = {
            'test_acc': [],
            'f1': []
        }

        for cp in corruption_rates:
            print(f'Staring with c_rate = {cp}...')
            m, pretrain_history = fwc2_pretrain(
                train_ds=train_ds,
                val_ds=val_ds,
                encoder_dims=encoder_dims,
                projection_dims=projection_dims,
                corruption_rate=cp,
                tau=tau,
                batch_size=batch_size,
                max_epochs=max_pretrain_epochs,
                device=device
            )

            torch.save(m.state_dict(), f'./output/{dataset}/pretrained-model--w(cp={cp},tau={tau},ds={dataset}).pt')

            history_df = pd.DataFrame(pretrain_history)
            history_df.to_csv(f'./output/{dataset}/pretraining-history--w(cp={cp},tau={tau},ds={dataset}).csv', encoding='utf-8')

            for param in m.encoder.parameters():
                param.requires_grad = False
            m.head = MLP(encoder_dims[-1], [128, 64, num_stages])

            m, finetune_history = finetune(
                m,
                train_ds=train_ds,
                val_ds=val_ds,
                max_epochs=max_finetune_epochs,
                batch_size=batch_size,
                device=device
            )

            torch.save(m.state_dict(), f'./output/{dataset}/fine-tuned-model--w(cp={cp},tau={tau},ds={dataset}).pt')

            history_df = pd.DataFrame(finetune_history)
            history_df.to_csv(f'./output/{dataset}/fine-tuned-history--w(cp={cp},tau={tau},ds={dataset}).csv', encoding='utf-8')

            print('Evaluations on test data ...')
            acc, f1, cm = evaluate_model(m, test_ds, device)
            (pd.DataFrame(cm)).to_csv(f'./output/{dataset}/eval-cm--w(cp={cp},tau={tau},ds={dataset}).csv',encoding='utf-8')

            scores['test_acc'].append(acc)
            scores['f1'].append(f1)
        (pd.DataFrame(scores)).to_csv(f'./output/{dataset}/eval-scores--w(tau={tau},ds={dataset}).csv',encoding='utf-8')
if __name__ == "__main__":
    main()
