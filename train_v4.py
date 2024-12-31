import os
from argparse import ArgumentParser
from datetime import datetime
import random
import string

from dotenv import load_dotenv

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import pytorch_metric_learning.losses as losses

import wandb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, \
    roc_curve, average_precision_score

from fwc2.model import S2SDEncoder, S2SDClassifier, RandomFeatureCorruption
from fwc2.data import NetFlowDataset
from fwc2.utils import load_train_test_sets, tsne_scatter_anomaly, plot_precision_recall_curve, plot_roc_curve

SEED = 42


def main_fn(
        ds_name: str,
        num_stages: int,
        encoder_hidden_dim: int = 256,
        classifier_hidden_dim: int = 256,
        n_encoder_layers: int = 4,
        n_projection_layers: int = 2,
        n_classifier_layers: int = 2,
        cp: float = 0.4,
        corrupt_both_views: bool = True,
        tau: float = 1.0,
        max_pretrain_epochs: int = 200,
        max_finetune_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        plot_tsne: bool = False,
        wandb_prj_name: str = 's2d2-detector-default',
):
    print(f'****************************')
    print(f'*** EXPERIMENT PARAMS *****')
    print(f'*** dataset = {ds_name} ')
    print(f'*** #stages = {num_stages}')
    print(f'*** encoder_hidden_dim = {encoder_hidden_dim} ')
    print(f'*** classifier_hidden_dim = {classifier_hidden_dim} ')
    print(f'*** #encoder layers = {n_encoder_layers} ')
    print(f'*** #project layers = {n_projection_layers} ')
    print(f'*** #classify layers = {n_classifier_layers} ')
    print(f'*** corrupt_rate = {cp} ')
    print(f'*** corrupt_both_views = {corrupt_both_views} ')
    print(f'*** tau = {tau} ')
    print(f'*** max_pretrain_epochs = {max_pretrain_epochs}')
    print(f'*** max_finetune_epochs = {max_finetune_epochs}')
    print(f'*** learning_rate = {learning_rate}')
    print(f'*** batch_size = {batch_size}')
    print(f'****************************')

    load_dotenv()

    pl.seed_everything(SEED)

    # wandb
    wandb.login(key=os.getenv('WANDB_API_KEY'))

    wandb_logger = WandbLogger(project=wandb_prj_name, log_model=True)
    wandb_logger.experiment.config["dataset"] = ds_name
    wandb_logger.experiment.config["encoder_hidden_dim"] = encoder_hidden_dim
    wandb_logger.experiment.config["classifier_hidden_dim"] = classifier_hidden_dim
    wandb_logger.experiment.config["n_encoder_layers"] = n_encoder_layers
    wandb_logger.experiment.config["n_projection_layers"] = n_projection_layers
    wandb_logger.experiment.config["n_classifier_layers"] = n_classifier_layers
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["corruption_rate"] = cp
    wandb_logger.experiment.config["corruption_both_views"] = corrupt_both_views
    wandb_logger.experiment.config["tau"] = tau

    print(f'*** PRETRAIN STEP ***')
    train_set, val_set, test_set = load_train_test_sets(ds_name, anomaly=True, seed=SEED)

    train_X, train_y = train_set.drop(columns=['label']), train_set['label']
    val_X, val_y = val_set.drop(columns=['label']), val_set['label']
    test_X, test_y = test_set.drop(columns=['label']), test_set['label']

    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    pipline = Pipeline([('scale', scaler)])

    pipline.fit(train_X)
    label_encoder.fit(train_y)

    train_ds = NetFlowDataset(pipline.transform(train_X), labels=label_encoder.transform(train_y))
    val_ds = NetFlowDataset(pipline.transform(val_X), labels=label_encoder.transform(val_y))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    corrupt_fn = RandomFeatureCorruption(
        corruption_rate=cp,
        corrupt_both_views=corrupt_both_views,
        features_low=train_ds.features_low,
        features_high=train_ds.features_high,
    )
    loss_fn = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=tau))
    model = S2SDEncoder(
        hidden_dim=encoder_hidden_dim,
        n_encoder_layers=n_encoder_layers,
        n_projection_layers=n_projection_layers,
        corrupt_fn=corrupt_fn,
        loss_fn=loss_fn,
        lr=learning_rate,
        batch_norm=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.001,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=max_pretrain_epochs,
        callbacks=[early_stopping],
        logger=wandb_logger,
        log_every_n_steps=1,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    print(f'*** ANOMALY DETECTION MODEL STEP ***')

    test_X = pipline.transform(test_X)
    test_y = test_y.apply(lambda x: 1 if x != 'benign' else 0)

    train_Z = model.get_embeddings(torch.tensor(np.array(train_X), dtype=torch.float32))
    test_Z = model.get_embeddings(torch.tensor(np.array(test_X), dtype=torch.float32))

    h_contamination = [0.01, 0.05, 0.1, 0.2]
    h_n_estimators = [50, 100, 150, 250]
    grid_search_params = [(c, ne) for c in h_contamination for ne in h_n_estimators]

    log = {
        'contamination': [],
        'n_estimators': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'avg_precision': [],
        'roc_auc': [],
    }
    for idx, (c, ne) in enumerate(grid_search_params):
        anomaly_detector = IsolationForest(n_estimators=ne, contamination=c, random_state=SEED)
        anomaly_detector.fit(train_Z.numpy())

        scores = anomaly_detector.decision_function(test_Z.numpy())
        y_pred = pd.Series(anomaly_detector.predict(test_Z.numpy())).replace([-1, 1], [1, 0])

        acc = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred)
        recall = recall_score(test_y, y_pred)
        f1 = f1_score(test_y, y_pred)

        precisions, recalls, thresholds = precision_recall_curve(test_y, -scores)
        ap = average_precision_score(test_y, -scores)

        fpr, tpr, threshold = roc_curve(test_y, scores)
        roc_auc = auc(fpr, tpr)

        plot_precision_recall_curve(recalls, precisions, ap, save_as=f'if-hparam-{idx}.svg')
        plot_roc_curve(fpr, tpr, roc_auc, save_as=f'if-hparam-{idx}.svg')

        log['contamination'].append(c)
        log['n_estimators'].append(ne)
        log['accuracy'].append(acc)
        log['precision'].append(precision)
        log['recall'].append(recall)
        log['f1'].append(f1)
        log['avg_precision'].append(ap)
        log['roc_auc'].append(roc_auc)

    log["dataset"] = [ds_name] * 16
    log["encoder_hidden_dim"] = [encoder_hidden_dim] * 16
    log["classifier_hidden_dim"] = [classifier_hidden_dim] * 16
    log["n_encoder_layers"] = [n_encoder_layers] * 16
    log["n_projection_layers"] = [n_projection_layers] * 16
    log["n_classifier_layers"] = [n_classifier_layers] * 16
    log["batch_size"] = [batch_size] * 16
    log["corruption_rate"] = [cp] * 16
    log["corruption_both_views"] = [corrupt_both_views] * 16
    log["tau"] = [tau] * 16

    best_ap_i = np.argmax(log['avg_precision'])
    print(f'best if hparam is {grid_search_params[best_ap_i]}, average p = {log["avg_precision"][best_ap_i]}')

    log_name = ''.join(random.sample((string.ascii_uppercase + string.digits) * 6, 6))
    log_dir = f'./results/{datetime.now().strftime("%Y-%m-%d")}-{log_name}/'

    os.makedirs(log_dir, exist_ok=True)
    pd.DataFrame(log).to_csv(f'{log_dir}/log.csv', index=False, encoding='utf-8')

    # viz embedding
    if plot_tsne:
        fig = tsne_scatter_anomaly(test_Z.numpy(), test_y, save_as=f'{log_dir}/{ds_name}-embeddings-tsne.png')
        wandb.log({f'{ds_name}-tsne-embeddings': wandb.Image(fig)})

    wandb_logger.finalize(status='success')
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--ds", type=str, default='dapt20')
    parser.add_argument("--ns", type=int, default=5)
    parser.add_argument("--edim", type=int, default=256)
    parser.add_argument("--cdim", type=int, default=256)
    parser.add_argument("--nel", type=int, default=4)
    parser.add_argument("--npl", type=int, default=2)
    parser.add_argument("--ncl", type=int, default=2)
    parser.add_argument("--cr", type=float, default=0.1)
    parser.add_argument("--cr_both", action="store_true", default=True)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mpe", type=int, default=200)
    parser.add_argument("--mfe", type=int, default=50)
    parser.add_argument("--plot_tsne", action="store_true", default=False)
    parser.add_argument("--wandb_prj", type=str, default="s2sd-detector")

    args = parser.parse_args()

    main_fn(
        args.ds,
        args.ns,
        args.edim,
        args.cdim,
        args.nel,
        args.npl,
        args.ncl,
        args.cr,
        args.cr_both,
        args.tau,
        args.mpe,
        args.mfe,
        args.lr,
        args.bs,
        args.plot_tsne,
        args.wandb_prj
    )
