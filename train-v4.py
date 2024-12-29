import os
from argparse import ArgumentParser
from dotenv import load_dotenv

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from fwc2.model import S2SDEncoder, S2SDClassifier, RandomFeatureCorruption
from fwc2.data import NetFlowDataset
from fwc2.utils import load_train_test_sets, tsne_scatter

SEED = 42


def main(
        ds_name: str,
        num_stages: int,
        encoder_hidden_dim: int = 256,
        classifier_hidden_dim: int = 256,
        n_encoder_layers: int = 4,
        n_projection_layers: int = 2,
        n_classifier_layers: int = 2,
        cp: float = 0.4,
        tau: float = 1.0,
        max_pretrain_epochs: int = 200,
        max_finetune_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
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

    wandb_logger = WandbLogger(project='s2sd-detector-local', log_model=True)
    wandb_logger.experiment.config["dataset"] = ds_name
    wandb_logger.experiment.config["encoder_hidden_dim"] = encoder_hidden_dim
    wandb_logger.experiment.config["classifier_hidden_dim"] = classifier_hidden_dim
    wandb_logger.experiment.config["n_encoder_layers"] = n_encoder_layers
    wandb_logger.experiment.config["n_projection_layers"] = n_projection_layers
    wandb_logger.experiment.config["n_classifier_layers"] = n_classifier_layers
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["corruption_rate"] = cp
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
        deterministic=True
    )
    trainer.fit(model, train_loader, val_loader)

    # viz embedding vs original data

    # test_Xt = pipline.transform(test_X)
    #
    # test_Z = model.get_embeddings(torch.tensor(test_Xt, dtype=torch.float32))
    #
    # print(f'** embedding shape = {test_Z.shape}')
    #
    # tsne_scatter(test_Z.numpy(), test_y, save_as=f'{ds_name}-z-space-tsne.png')
    print(f'*** ANOMALY DETECTION MODEL STEP ***')

    train_Z = model.get_embeddings(train_X)
    test_Z = model.get_embeddings(test_X)

    anomaly_detector = IsolationForest(n_estimators=150, contamination=0.2, random_state=SEED)
    anomaly_detector.fit(train_Z)

    # testing
    y_pred = pd.Series(anomaly_detector.predict(test_Z)).replace([-1, 1], [1, 0])

    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred)

    print('results on encoded data')
    print(f'accuracy = {accuracy}, precision = {precision}, recall = {recall}, f1 = {f1}')

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
    parser.add_argument("--cr", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mpe", type=int, default=200)
    parser.add_argument("--mfe", type=int, default=50)

    args = parser.parse_args()

    main(
        args.ds,
        args.ns,
        args.edim,
        args.cdim,
        args.nel,
        args.npl,
        args.ncl,
        args.cr,
        args.tau,
        args.mpe,
        args.mfe,
        args.lr,
        args.bs,
    )
