import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    train_path: str = "Data/train.csv"
    test_path: str = "Data/test.csv"
    output_dir: str = "output_data_test_point"

    batch_size: int = 32
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.3

    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 10

    val_size: float = 0.25
    random_state: int = 30

    use_class_weight: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()

GROUP_COL = "rally_uid"
ORDER_COL = "strikeNumber"
TARGET = "pointId"

SEQ_CAT_COLS = [
    "gamePlayerId",
    "gamePlayerOtherId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
]

SEQ_NUM_COLS = [
    "is_attack",
    "is_control",
    "is_defensive",
    "is_serve",
]

ATTACK_ACTIONS = [1, 2, 3, 4, 5, 6, 7]
CONTROL_ACTIONS = [8, 9, 10, 11]
DEFENSIVE_ACTIONS = [12, 13, 14]
SERVE_ACTIONS = [15, 16, 17, 18]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values([GROUP_COL, ORDER_COL]).reset_index(drop=True)

    df["score_diff"] = df["scoreSelf"] - df["scoreOther"]
    df["is_attack"] = df["actionId"].isin(ATTACK_ACTIONS).astype(np.float32)
    df["is_control"] = df["actionId"].isin(CONTROL_ACTIONS).astype(np.float32)
    df["is_defensive"] = df["actionId"].isin(DEFENSIVE_ACTIONS).astype(np.float32)
    df["is_serve"] = df["actionId"].isin(SERVE_ACTIONS).astype(np.float32)

    return df


def build_category_maps(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[int, int]]:
    maps = {}
    for col in cat_cols:
        vals = sorted(df[col].dropna().unique().tolist())
        maps[col] = {v: i + 1 for i, v in enumerate(vals)}
    return maps


def apply_category_maps(df: pd.DataFrame, maps: Dict[str, Dict[int, int]], cat_cols: List[str]):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].map(maps[col]).fillna(0).astype(np.int64)
    return df


def get_emb_dim(cardinality: int):
    return min(16, max(4, int(math.sqrt(cardinality) + 1)))


def build_train_samples(df: pd.DataFrame) -> List[dict]:
    samples = []
    for rally_uid, g in df.groupby(GROUP_COL, sort=False):
        g = g.sort_values(ORDER_COL).reset_index(drop=True)
        if len(g) < 2:
            continue

        for i in range(1, len(g)):
            prefix = g.iloc[:i]
            target = g.iloc[i]

            samples.append({
                "rally_uid": int(rally_uid),
                "seq_cat": prefix[SEQ_CAT_COLS].to_numpy(dtype=np.int64),
                "seq_num": prefix[SEQ_NUM_COLS].to_numpy(dtype=np.float32),
                "y_point": int(target[TARGET]),
                "seq_len": len(prefix),
            })

    return samples


def build_test_samples(df: pd.DataFrame) -> List[dict]:
    samples = []
    for rally_uid, g in df.groupby(GROUP_COL, sort=False):
        g = g.sort_values(ORDER_COL).reset_index(drop=True)
        samples.append({
            "rally_uid": int(rally_uid),
            "seq_cat": g[SEQ_CAT_COLS].to_numpy(dtype=np.int64),
            "seq_num": g[SEQ_NUM_COLS].to_numpy(dtype=np.float32),
            "seq_len": len(g),
        })
    return samples


class RallyDataset(Dataset):
    def __init__(self, samples: List[dict], is_train: bool):
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RallyCollator:
    def __init__(self, is_train: bool):
        self.is_train = is_train

    def __call__(self, batch: List[dict]):
        batch = sorted(batch, key=lambda x: x["seq_len"], reverse=True)

        seq_cat = pad_sequence(
            [torch.tensor(x["seq_cat"], dtype=torch.long) for x in batch],
            batch_first=True,
            padding_value=0,
        )
        seq_num = pad_sequence(
            [torch.tensor(x["seq_num"], dtype=torch.float32) for x in batch],
            batch_first=True,
            padding_value=0.0,
        )

        out = {
            "seq_cat": seq_cat,
            "seq_num": seq_num,
            "seq_len": torch.tensor([x["seq_len"] for x in batch], dtype=torch.long),
            "rally_uid": torch.tensor([x["rally_uid"] for x in batch], dtype=torch.long),
        }

        if self.is_train:
            out["y_point"] = torch.tensor([x["y_point"] for x in batch], dtype=torch.long)

        return out


def make_loader(samples: List[dict], batch_size: int, shuffle: bool, is_train: bool):
    return DataLoader(
        RallyDataset(samples, is_train=is_train),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=RallyCollator(is_train=is_train),
        num_workers=CFG.num_workers,
    )


class PointLSTM(nn.Module):
    def __init__(self, category_maps, hidden_dim, num_layers, dropout):
        super().__init__()

        self.cat_cols = SEQ_CAT_COLS
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0

        important_emb_dim = {
            "actionId": 32,
            "pointId": 24,
        }

        for col in self.cat_cols:
            card = len(category_maps[col]) + 1
            dim = important_emb_dim.get(col, get_emb_dim(card))

            self.embeddings[col] = nn.Embedding(card, dim, padding_idx=0)
            total_emb_dim += dim

        input_dim = total_emb_dim + len(SEQ_NUM_COLS)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.point_head = nn.Linear(hidden_dim, len(category_maps["pointId"]) + 1)

    def forward(self, seq_cat, seq_num, seq_len):
        embs = []
        for i, col in enumerate(self.cat_cols):
            embs.append(self.embeddings[col](seq_cat[:, :, i]))

        x = torch.cat(embs + [seq_num], dim=-1)

        packed = pack_padded_sequence(
            x,
            seq_len.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )

        _, (h_n, _) = self.lstm(packed)
        h = self.dropout(h_n[-1])

        return self.point_head(h)


def make_class_weight(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=num_classes)
    counts[0] = 0

    weights = np.zeros_like(counts, dtype=np.float32)
    valid = counts > 0
    weights[valid] = counts[valid].sum() / counts[valid]

    if valid.any():
        weights[valid] = weights[valid] / weights[valid].mean()

    weights[0] = 0.0

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in loader:
        seq_cat = batch["seq_cat"].to(device)
        seq_num = batch["seq_num"].to(device)
        seq_len = batch["seq_len"].to(device)
        y_point = batch["y_point"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(seq_cat, seq_num, seq_len)
            loss = criterion(logits, y_point)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * seq_cat.size(0)
        y_true.extend(y_point.cpu().numpy())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy())

    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return {
        "loss": total_loss / len(loader.dataset),
        "macro_f1": macro_f1,
    }


@torch.no_grad()
def predict_test(model, loader, device):
    model.eval()
    rally_uids = []
    point_idx = []

    for batch in loader:
        seq_cat = batch["seq_cat"].to(device)
        seq_num = batch["seq_num"].to(device)
        seq_len = batch["seq_len"].to(device)

        logits = model(seq_cat, seq_num, seq_len)

        rally_uids.extend(batch["rally_uid"].cpu().numpy())
        point_idx.extend(logits.argmax(dim=1).cpu().numpy())

    return np.asarray(rally_uids), np.asarray(point_idx)


def main():
    set_seed(CFG.random_state)

    out_dir = ensure_dir(CFG.output_dir)
    device = torch.device(CFG.device)

    print(f"Using device: {device}")

    train_df_raw = load_df(CFG.train_path)
    test_df_raw = load_df(CFG.test_path)

    category_maps = build_category_maps(train_df_raw, SEQ_CAT_COLS)
    inverse_point_map = {v: k for k, v in category_maps["pointId"].items()}

    # Rally-based split
    rally_ids = train_df_raw[GROUP_COL].drop_duplicates().to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG.val_size,
        random_state=CFG.random_state,
    )

    tr_idx, va_idx = next(splitter.split(rally_ids, groups=rally_ids))

    tr_rallies = set(rally_ids[tr_idx])
    va_rallies = set(rally_ids[va_idx])

    train_df = train_df_raw[train_df_raw[GROUP_COL].isin(tr_rallies)].copy()
    valid_df = train_df_raw[train_df_raw[GROUP_COL].isin(va_rallies)].copy()

    print("Train rallies:", train_df[GROUP_COL].nunique())
    print("Valid rallies:", valid_df[GROUP_COL].nunique())
    print("Overlap rallies:", set(train_df[GROUP_COL]) & set(valid_df[GROUP_COL]))

    scaler = StandardScaler()

    train_df[SEQ_NUM_COLS] = scaler.fit_transform(train_df[SEQ_NUM_COLS])
    valid_df[SEQ_NUM_COLS] = scaler.transform(valid_df[SEQ_NUM_COLS])

    train_df = apply_category_maps(train_df, category_maps, SEQ_CAT_COLS)
    valid_df = apply_category_maps(valid_df, category_maps, SEQ_CAT_COLS)

    train_samples = build_train_samples(train_df)
    valid_samples = build_train_samples(valid_df)

    print("Train samples:", len(train_samples))
    print("Valid samples:", len(valid_samples))

    train_loader = make_loader(train_samples, CFG.batch_size, shuffle=True, is_train=True)
    valid_loader = make_loader(valid_samples, CFG.batch_size, shuffle=False, is_train=True)

    model = PointLSTM(
        category_maps,
        CFG.hidden_dim,
        CFG.num_layers,
        CFG.dropout,
    ).to(device)

    if CFG.use_class_weight:
        point_w = make_class_weight(
            [s["y_point"] for s in train_samples],
            len(category_maps["pointId"]) + 1,
        ).to(device)
    else:
        point_w = None

    criterion = nn.CrossEntropyLoss(weight=point_w)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    best_score = -np.inf
    best_state = None

    for epoch in range(1, CFG.epochs + 1):
        tr_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_metrics = run_epoch(model, valid_loader, optimizer, criterion, device, train=False)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_metrics['loss']:.4f} | "
            f"valid_loss={va_metrics['loss']:.4f} | "
            f"valid_point_f1={va_metrics['macro_f1']:.4f}"
        )

        if va_metrics["macro_f1"] > best_score:
            best_score = va_metrics["macro_f1"]
            best_state = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No usable model.")

    print(f"Best valid point Macro F1 = {best_score:.4f}")

    torch.save(
        {
            "model_state_dict": best_state,
            "category_maps": category_maps,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "best_valid_point_f1": best_score,
            "config": CFG.__dict__,
        },
        out_dir / "best_point_lstm.pt",
    )

    test_df = test_df_raw.copy()
    test_df[SEQ_NUM_COLS] = scaler.transform(test_df[SEQ_NUM_COLS])
    test_df = apply_category_maps(test_df, category_maps, SEQ_CAT_COLS)

    test_samples = build_test_samples(test_df)
    test_loader = make_loader(test_samples, CFG.batch_size, shuffle=False, is_train=False)

    final_model = PointLSTM(
        category_maps,
        CFG.hidden_dim,
        CFG.num_layers,
        CFG.dropout,
    ).to(device)

    final_model.load_state_dict(best_state)

    rally_uids, point_idx = predict_test(final_model, test_loader, device)

    submission = pd.DataFrame({
        "rally_uid": rally_uids.astype(int),
        "pointId": [
            inverse_point_map.get(int(x), 0)
            for x in point_idx
        ],
    })

    submission = submission.sort_values("rally_uid").reset_index(drop=True)
    output_path = out_dir / "submission_point_lstm.csv"
    submission.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
