import random

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from recommender.models import Recommender
from recommender.data_processing import get_context, pad_list, map_column, MASK

src_mock = torch.tensor([126049,   2087,   2108,      1,  74362, 138309,      1, 253102, 141627,
        141124, 141124,      1, 141124,   1063,      1,      1, 115923,      1,
           908,      1,   5998,  26121, 105864,      1, 116359,  16082,   1369,
         27315,   4579,  17422,  13596, 206994,   5406,      1,  31258, 107582,
        107582,  30039,   4140, 140373,      1,   5162,      1,      1,  96527,
         34025, 107431,   9681,  38086,   1369,      1,      1, 115923,  16585,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0])

trg_mock = torch.tensor([126049,   2087,   2108,  12224,  74362, 138309, 140373, 253102, 141627,
        141124, 141124,  36107, 141124,   1063,   4828,   4828, 115923,  47617,
           908,   5162,   5998,  26121, 105864,  24684, 116359,  16082,   1369,
         27315,   4579,  17422,  13596, 206994,   5406,  72539,  31258, 107582,
        107582,  30039,   4140, 140373,  14727,   5162,   5080,   4433,  96527,
         34025, 107431,   9681,  38086,   1369, 101000,   4579, 115923,  16585,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0])


def mask_list(l1, p=0.8):

    l1 = [a if random.random() < p else MASK for a in l1]

    return l1


def mask_last_elements_list(l1, val_context_size: int = 5):

    l1 = l1[:-val_context_size] + mask_list(l1[-val_context_size:], p=0.5)

    return l1


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, unique_users, split, history_size=120):
        self.df = df
        self.unique_users = unique_users
        self.split = split
        self.history_size = history_size

    def __len__(self):
        return len(self.unique_users)

    def __getitem__(self, idx):
        # user_id = self.unique_users[idx]
        #
        # df = self.df[self.df["userId"] == user_id]
        #
        # context = get_context(df, split=self.split, context_size=self.history_size)
        #
        # trg_items = context["movieId_mapped"].tolist()
        #
        # if self.split == "train":
        #     src_items = mask_list(trg_items)
        # else:
        #     src_items = mask_last_elements_list(trg_items)
        #
        # pad_mode = "left" if random.random() < 0.5 else "right"
        # trg_items = pad_list(trg_items, history_size=self.history_size, mode=pad_mode)
        # src_items = pad_list(src_items, history_size=self.history_size, mode=pad_mode)
        #
        # src_items = torch.tensor(src_items, dtype=torch.long)
        #
        # trg_items = torch.tensor(trg_items, dtype=torch.long)

        return src_mock, trg_mock


def train(
    data_csv_path: str,
    log_dir: str = "recommender_logs",
    model_dir: str = "recommender_models",
    batch_size: int = 32,
    epochs: int = 2000,
    history_size: int = 120,
):
    data = pd.read_feather(data_csv_path)
    if 'index' in data.columns:
        del data['index']
    data_grp = data.groupby("userId").count()['movieId']
    data_grp = data_grp[data_grp >= 15]
    unique_users = data_grp.index.values
    data = data[data["userId"].isin(unique_users)].reset_index(drop=True)

    data.sort_values(by="timestamp", inplace=True)

    data, mapping, _ = map_column(data, col_name="movieId")

    train_data = Dataset(
        df=data,
        unique_users=unique_users,
        split="train",
        history_size=history_size,
    )
    val_data = Dataset(
        df=data,
        unique_users=unique_users,
        split="val",
        history_size=history_size,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=3,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=3,
        shuffle=False,
    )

    model = Recommender(
        vocab_size=len(mapping) + 2,
        lr=1e-4,
        dropout=0.3,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="recommender",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    result_val = trainer.test(test_dataloaders=val_loader)

    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    print(output_json)

    return output_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train(
        data_csv_path=args.data_csv_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
