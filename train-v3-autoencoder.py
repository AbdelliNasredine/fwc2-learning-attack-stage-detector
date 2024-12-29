import os
from argparse import ArgumentParser
from dotenv import load_dotenv

import torch

import wandb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from fwc2.model import FWC2AEv3
from fwc2.data import FWC2Dataset
from fwc2.utils import load_train_test_sets, tsne_scatter

SEED = 42


def main(
        ds_name: str,
        in_dim: int,
        num_stages: int,
        cp: float,
        tau: float,
        max_train_epochs: int,
        learning_rate=1e-3,
        batch_size=256,
        dropout_rate=0.1,
        train_early_stopping_patience: int = 5,
):
    print(f'****************************')
    print(f'*** STARING EXPERIMENT *****')
    print(f'*** corrupt_rate = {cp} ')
    print(f'*** dataset = {ds_name} ')
    print(f'*** in_dim = {in_dim}   ')
    print(f'*** tau = {tau}         ')
    print(f'*** #stages = {num_stages}')
    print(f'*** max_train_epochs = {max_train_epochs}')
    print(f'*** learning_rate = {learning_rate}')
    print(f'*** batch_size = {batch_size}')
    print(f'*** dropout_rate = {dropout_rate}')
    print(f'****************************')

    load_dotenv()

    pl.seed_everything(SEED)

    # wandb
    wandb.login(key=os.getenv('WANDB_API_KEY'))

    wandb_logger = WandbLogger(project='fwc2-learning-anomaly-local', log_model=True)
    wandb_logger.experiment.config["dataset"] = ds_name
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["corruption_rate"] = cp
    wandb_logger.experiment.config["temprature"] = tau

    train_set, val_set, test_set = load_train_test_sets(ds_name, anomaly=True, seed=SEED)

    scaler = MinMaxScaler()
    pipline = Pipeline([('normalize', Normalizer()), ('scale', scaler)])

    pipline.fit(train_set)

    train_ds = FWC2Dataset(pipline.transform(train_set))
    val_ds = FWC2Dataset(pipline.transform(val_set))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

    # net archi params
    enc_hidden = 8
    enc_layers = 2

    model = FWC2AEv3(
        in_dim=in_dim,
        hidden_dim=enc_hidden,
        encoder_layers=enc_layers,
        lr=learning_rate,
        dropout=dropout_rate
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=train_early_stopping_patience,
        min_delta=0.0001,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=max_train_epochs,
        callbacks=[early_stopping],
        logger=wandb_logger,
        log_every_n_steps=1,
        deterministic=True
    )
    trainer.fit(
        model,
        train_loader,
        val_loader
    )

    wandb_logger.finalize(status='success')
    wandb.finish()

    # viz embedding vs original data
    test_X, test_y = test_set.drop(columns=['label']), test_set['label']
    test_Xt = pipline.transform(test_X)

    z_test = model.get_embeddings(torch.tensor(test_Xt, dtype=torch.float32))

    print(f'** embedding shape = {z_test.shape}')

    tsne_scatter(z_test.numpy(), test_y, save_as=f'{ds_name}-z-space-tsne.png')

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--cr", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--ds", type=str, default='dapt20')
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dpr", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=30)

    args = parser.parse_args()

    dataset = {'dapt20': (5, 64), 'scvic21': (6, None), 'mscad': (6, None)}[args.ds]

    main(
        args.ds,
        dataset[1],
        dataset[0],
        args.cr,
        args.tau,
        learning_rate=args.lr,
        batch_size=args.bs,
        dropout_rate=args.dpr,
        max_train_epochs=args.max_epochs
    )
