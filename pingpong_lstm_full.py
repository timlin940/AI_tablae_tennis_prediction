import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


# =========================
# Config
# =========================

@dataclass
class Config:
    train_path: str = "Data/train.csv"
    test_path: str = "Data/test.csv"
    output_dir: str = "output_data"

    batch_size: int = 64
    hidden_dim: int = 256 # 隱藏層維度，過小可能無法捕捉複雜模式，過大可能導致過擬合和訓練不穩定
    num_layers: int = 5 # LSTM 層數，增加層數可以捕捉更複雜的模式，但也可能導致過擬合和梯度消失問題
    dropout: float = 0.4

    lr: float = 1e-3
    weight_decay: float = 5e-5
    epochs: int = 15

    val_size: float = 0.4
    random_state: int = 40

    use_class_weight: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()

GROUP_COL = "rally_uid"
MATCH_COL = "match"
ORDER_COL = "strikeNumber"

TARGET_ACTION = "actionId"
TARGET_POINT = "pointId"
TARGET_SERVER = "serverGetPoint"

SEQ_CAT_COLS = [ # pointID會有接球方左右手問題，目前無解
    "gamePlayerId",
    "gamePlayerOtherId",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "positionId",
    "actionId",
    "pointId",
]


SEQ_NUM_COLS = [
    "scoreSelf",
    "scoreOther",
    "score_diff",
    # "strikeNumber",
    # "numberGame",

    "is_attack",
    "is_control",
    "is_defensive",
    "is_serve",
]

# =========================
# Basic utils
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


ATTACK_ACTIONS = [1, 2, 3, 4, 5, 6, 7]
CONTROL_ACTIONS = [8, 9, 10, 11]
DEFENSIVE_ACTIONS = [12, 13, 14]
SERVE_ACTIONS = [15, 16, 17, 18]


def load_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.sort_values(
        [GROUP_COL, ORDER_COL]
    ).reset_index(drop=True)

    df["score_diff"] = (
        df["scoreSelf"] - df["scoreOther"]
    )

    df["is_attack"] = (
        df["actionId"]
        .isin(ATTACK_ACTIONS)
        .astype(np.float32)
    )

    df["is_control"] = (
        df["actionId"]
        .isin(CONTROL_ACTIONS)
        .astype(np.float32)
    )

    df["is_defensive"] = (
        df["actionId"]
        .isin(DEFENSIVE_ACTIONS)
        .astype(np.float32)
    )

    df["is_serve"] = (
        df["actionId"]
        .isin(SERVE_ACTIONS)
        .astype(np.float32)
    )

    return df


def build_category_maps(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict]:
    maps = {}

    for col in cat_cols:
        vals = sorted(df[col].dropna().unique().tolist())
        maps[col] = {v: i + 1 for i, v in enumerate(vals)}

    return maps


def apply_category_maps(
    df: pd.DataFrame,
    maps: Dict[str, Dict],
    cat_cols: List[str],
) -> pd.DataFrame:
    df = df.copy()

    for col in cat_cols:
        df[col] = df[col].map(maps[col]).fillna(0).astype(np.int64)

    return df


def get_emb_dim(cardinality: int) -> int:
    return min(32, max(4, int(math.sqrt(cardinality) + 1)))


# =========================
# Sample builders
# =========================

def build_train_samples(df: pd.DataFrame) -> List[dict]:
    """
    建立 prefix 預測下一拍資料：

    1      -> 預測 2
    1~2    -> 預測 3
    1~3    -> 預測 4
    ...
    1~t-1  -> 預測 t

    同時使用同一個 prefix 預測該 rally 最終 serverGetPoint。
    """
    samples = []

    for rally_uid, g in df.groupby(GROUP_COL, sort=False):
        g = g.sort_values(ORDER_COL).reset_index(drop=True)

        if len(g) < 2:
            continue

        final_server = float(g.iloc[-1][TARGET_SERVER])

        for i in range(1, len(g)):
            prefix = g.iloc[:i]
            target = g.iloc[i]

            samples.append({
                "rally_uid": int(rally_uid),
                "seq_cat": prefix[SEQ_CAT_COLS].to_numpy(dtype=np.int64),
                "seq_num": prefix[SEQ_NUM_COLS].to_numpy(dtype=np.float32),
                "y_action": int(target[TARGET_ACTION]),
                "y_point": int(target[TARGET_POINT]),
                "y_server": final_server,
                "seq_len": len(prefix),
            })

    return samples


def build_test_samples(df: pd.DataFrame) -> List[dict]:
    """
    test：用目前已知整段 rally 預測下一拍 actionId / pointId，
    同時預測 serverGetPoint。
    """
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


# =========================
# Dataset / Collator
# =========================

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

    def __call__(self, batch: List[dict]) -> dict:
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
            "seq_len": torch.tensor(
                [x["seq_len"] for x in batch],
                dtype=torch.long,
            ),
            "rally_uid": torch.tensor(
                [x["rally_uid"] for x in batch],
                dtype=torch.long,
            ),
        }

        if self.is_train:
            out["y_action"] = torch.tensor(
                [x["y_action"] for x in batch],
                dtype=torch.long,
            )
            out["y_point"] = torch.tensor(
                [x["y_point"] for x in batch],
                dtype=torch.long,
            )
            out["y_server"] = torch.tensor(
                [x["y_server"] for x in batch],
                dtype=torch.float32,
            )

        return out


def make_loader(
    samples: List[dict],
    batch_size: int,
    shuffle: bool,
    is_train: bool,
):
    return DataLoader(
        RallyDataset(samples, is_train=is_train),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=RallyCollator(is_train=is_train),
        num_workers=CFG.num_workers,
    )


# =========================
# Model
# =========================

class BaseRallyEncoder(nn.Module):
    def __init__(self, category_maps, hidden_dim, num_layers, dropout):
        super().__init__()

        self.cat_cols = SEQ_CAT_COLS
        self.embeddings = nn.ModuleDict()

        total_emb_dim = 0

        for col in self.cat_cols:
            card = len(category_maps[col]) + 1
            dim = get_emb_dim(card)
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

        return h


class RallyLSTM(nn.Module):
    def __init__(self, category_maps, hidden_dim, num_layers, dropout):
        super().__init__()

        self.encoder = BaseRallyEncoder(
            category_maps,
            hidden_dim,
            num_layers,
            dropout,
        )

        self.action_head = nn.Linear(
            hidden_dim,
            len(category_maps["actionId"]) + 1,
        )

        self.point_head = nn.Linear(
            hidden_dim,
            len(category_maps["pointId"]) + 1,
        )

        self.server_head = nn.Linear(hidden_dim, 1)

    def forward(self, seq_cat, seq_num, seq_len):
        h = self.encoder(seq_cat, seq_num, seq_len)

        action_logits = self.action_head(h)
        point_logits = self.point_head(h)
        server_logit = self.server_head(h).squeeze(-1)

        return action_logits, point_logits, server_logit


# =========================
# Loss weights
# =========================

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


def make_pos_weight(labels: List[int]) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=2)

    if counts[1] == 0:
        return torch.tensor([1.0], dtype=torch.float32)

    return torch.tensor([counts[0] / counts[1]], dtype=torch.float32)


# =========================
# Training / Evaluation
# =========================

def run_epoch(
    model,
    loader,
    optimizer,
    crit_action,
    crit_point,
    crit_server,
    device,
    train: bool,
):
    model.train() if train else model.eval()

    total_loss = 0.0

    action_true, action_pred = [], []
    point_true, point_pred = [], []
    server_true, server_prob = [], []

    for batch in loader:
        seq_cat = batch["seq_cat"].to(device)
        seq_num = batch["seq_num"].to(device)
        seq_len = batch["seq_len"].to(device)

        y_action = batch["y_action"].to(device)
        y_point = batch["y_point"].to(device)
        y_server = batch["y_server"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            action_logits, point_logits, server_logit = model(
                seq_cat,
                seq_num,
                seq_len,
            )

            loss_action = crit_action(action_logits, y_action)
            loss_point = crit_point(point_logits, y_point)
            loss_server = crit_server(server_logit, y_server)

            loss = (
                0.4 * loss_action
                + 0.4 * loss_point
                + 0.2 * loss_server
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()

        total_loss += loss.item() * seq_cat.size(0)

        action_true.extend(y_action.cpu().numpy())
        action_pred.extend(
            action_logits.argmax(dim=1).detach().cpu().numpy()
        )

        point_true.extend(y_point.cpu().numpy())
        point_pred.extend(
            point_logits.argmax(dim=1).detach().cpu().numpy()
        )

        server_true.extend(y_server.cpu().numpy())
        server_prob.extend(
            torch.sigmoid(server_logit).detach().cpu().numpy()
        )

    action_f1 = f1_score(action_true, action_pred, average="macro")
    point_f1 = f1_score(point_true, point_pred, average="macro")

    try:
        server_auc = roc_auc_score(server_true, server_prob)
    except ValueError:
        server_auc = float("nan")

    score = (
        0.4 * action_f1
        + 0.4 * point_f1
        + 0.2 * server_auc
        if not np.isnan(server_auc)
        else float("nan")
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "action_macro_f1": action_f1,
        "point_macro_f1": point_f1,
        "server_auc": server_auc,
        "score": score,
    }


@torch.no_grad()
def predict_test(model, loader, device):
    model.eval()

    rally_uids = []
    action_idx = []
    point_idx = []
    server_prob = []

    for batch in loader:
        seq_cat = batch["seq_cat"].to(device)
        seq_num = batch["seq_num"].to(device)
        seq_len = batch["seq_len"].to(device)

        action_logits, point_logits, server_logit = model(
            seq_cat,
            seq_num,
            seq_len,
        )

        rally_uids.extend(batch["rally_uid"].cpu().numpy())
        action_idx.extend(action_logits.argmax(dim=1).cpu().numpy())
        point_idx.extend(point_logits.argmax(dim=1).cpu().numpy())
        server_prob.extend(torch.sigmoid(server_logit).cpu().numpy())

    return (
        np.asarray(rally_uids),
        np.asarray(action_idx),
        np.asarray(point_idx),
        np.asarray(server_prob),
    )


# =========================
# Main
# =========================

def main():
    set_seed(CFG.random_state)

    out_dir = ensure_dir(CFG.output_dir)
    device = torch.device(CFG.device)

    print(f"Using device: {device}")

    train_df_raw = load_df(CFG.train_path)
    test_df_raw = load_df(CFG.test_path)

    category_maps = build_category_maps(train_df_raw, SEQ_CAT_COLS)
    inverse_maps = {
        col: {v: k for k, v in m.items()}
        for col, m in category_maps.items()
    }

    # =========================
    # Match-based split
    # =========================

    match_ids = train_df_raw[MATCH_COL].drop_duplicates().to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG.val_size,
        random_state=CFG.random_state,
    )

    tr_idx, va_idx = next(splitter.split(match_ids, groups=match_ids))

    tr_matches = set(match_ids[tr_idx])
    va_matches = set(match_ids[va_idx])

    train_df = train_df_raw[train_df_raw[MATCH_COL].isin(tr_matches)].copy()
    valid_df = train_df_raw[train_df_raw[MATCH_COL].isin(va_matches)].copy()

    print("Train matches:", train_df[MATCH_COL].nunique())
    print("Valid matches:", valid_df[MATCH_COL].nunique())
    print("Overlap matches:", set(train_df[MATCH_COL]) & set(valid_df[MATCH_COL]))

    # =========================
    # Scaling / Encoding
    # =========================

    scaler = StandardScaler()

    train_df[SEQ_NUM_COLS] = scaler.fit_transform(train_df[SEQ_NUM_COLS])
    valid_df[SEQ_NUM_COLS] = scaler.transform(valid_df[SEQ_NUM_COLS])

    train_df = apply_category_maps(train_df, category_maps, SEQ_CAT_COLS)
    valid_df = apply_category_maps(valid_df, category_maps, SEQ_CAT_COLS)

    # =========================
    # Build samples
    # =========================

    train_samples = build_train_samples(train_df)
    valid_samples = build_train_samples(valid_df)

    print("Train samples:", len(train_samples))
    print("Valid samples:", len(valid_samples))

    train_loader = make_loader(
        train_samples,
        batch_size=CFG.batch_size,
        shuffle=True,
        is_train=True,
    )

    valid_loader = make_loader(
        valid_samples,
        batch_size=CFG.batch_size,
        shuffle=False,
        is_train=True,
    )

    # =========================
    # Model / Loss / Optimizer
    # =========================

    model = RallyLSTM(
        category_maps,
        CFG.hidden_dim,
        CFG.num_layers,
        CFG.dropout,
    ).to(device)

    if CFG.use_class_weight:
        action_w = make_class_weight(
            [s["y_action"] for s in train_samples],
            len(category_maps["actionId"]) + 1,
        ).to(device)

        point_w = make_class_weight(
            [s["y_point"] for s in train_samples],
            len(category_maps["pointId"]) + 1,
        ).to(device)

        server_pos_weight = make_pos_weight(
            [int(s["y_server"]) for s in train_samples]
        ).to(device)
    else:
        action_w = None
        point_w = None
        server_pos_weight = None

    crit_action = nn.CrossEntropyLoss(weight=action_w)
    crit_point = nn.CrossEntropyLoss(weight=point_w)
    crit_server = nn.BCEWithLogitsLoss(pos_weight=server_pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    # =========================
    # Train
    # =========================

    best_score = -np.inf
    best_state = None

    print("\n========== Training Single RallyLSTM ==========")

    for epoch in range(1, CFG.epochs + 1):
        tr_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            crit_action,
            crit_point,
            crit_server,
            device,
            train=True,
        )

        va_metrics = run_epoch(
            model,
            valid_loader,
            optimizer,
            crit_action,
            crit_point,
            crit_server,
            device,
            train=False,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_metrics['loss']:.4f} | "
            f"valid_loss={va_metrics['loss']:.4f} | "
            f"action_f1={va_metrics['action_macro_f1']:.4f} | "
            f"point_f1={va_metrics['point_macro_f1']:.4f} | "
            f"server_auc={va_metrics['server_auc']:.4f} | "
            f"score={va_metrics['score']:.4f}"
        )

        if not np.isnan(va_metrics["score"]) and va_metrics["score"] > best_score:
            best_score = va_metrics["score"]
            best_state = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No usable model.")

    torch.save(
        {
            "model_state_dict": best_state,
            "category_maps": category_maps,
            "inverse_maps": inverse_maps,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "best_valid_score": best_score,
            "config": CFG.__dict__,
        },
        out_dir / "best_single_model.pt",
    )

    print(f"\nSaved best model. Best valid score = {best_score:.4f}")

    # =========================
    # Full train fine-tuning
    # =========================
    """
    print("\n========== Full Train Fine-tuning ==========")

    full_scaler = StandardScaler()

    train_full_df = train_df_raw.copy()
    test_full_df = test_df_raw.copy()

    train_full_df[SEQ_NUM_COLS] = full_scaler.fit_transform(
        train_full_df[SEQ_NUM_COLS]
    )

    test_full_df[SEQ_NUM_COLS] = full_scaler.transform(
        test_full_df[SEQ_NUM_COLS]
    )

    train_full_df = apply_category_maps(
        train_full_df,
        category_maps,
        SEQ_CAT_COLS,
    )

    test_full_df = apply_category_maps(
        test_full_df,
        category_maps,
        SEQ_CAT_COLS,
    )

    full_train_samples = build_train_samples(train_full_df)
    test_samples = build_test_samples(test_full_df)

    full_train_loader = make_loader(
        full_train_samples,
        batch_size=CFG.batch_size,
        shuffle=True,
        is_train=True,
    )

    test_loader = make_loader(
        test_samples,
        batch_size=CFG.batch_size,
        shuffle=False,
        is_train=False,
    )

    final_model = RallyLSTM(
        category_maps,
        CFG.hidden_dim,
        CFG.num_layers,
        CFG.dropout,
    ).to(device)

    final_model.load_state_dict(best_state)

    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    if CFG.use_class_weight:
        full_action_w = make_class_weight(
            [s["y_action"] for s in full_train_samples],
            len(category_maps["actionId"]) + 1,
        ).to(device)

        full_point_w = make_class_weight(
            [s["y_point"] for s in full_train_samples],
            len(category_maps["pointId"]) + 1,
        ).to(device)

        full_server_pos_weight = make_pos_weight(
            [int(s["y_server"]) for s in full_train_samples]
        ).to(device)
    else:
        full_action_w = None
        full_point_w = None
        full_server_pos_weight = None

    full_crit_action = nn.CrossEntropyLoss(weight=full_action_w)
    full_crit_point = nn.CrossEntropyLoss(weight=full_point_w)
    full_crit_server = nn.BCEWithLogitsLoss(pos_weight=full_server_pos_weight)

    fine_tune_epochs = max(3, CFG.epochs // 3)

    for epoch in range(1, fine_tune_epochs + 1):
        ft_metrics = run_epoch(
            final_model,
            full_train_loader,
            final_optimizer,
            full_crit_action,
            full_crit_point,
            full_crit_server,
            device,
            train=True,
        )

        print(
            f"Full Train Epoch {epoch:02d} | "
            f"loss={ft_metrics['loss']:.4f} | "
            f"action_f1={ft_metrics['action_macro_f1']:.4f} | "
            f"point_f1={ft_metrics['point_macro_f1']:.4f} | "
            f"server_auc={ft_metrics['server_auc']:.4f} | "
            f"score={ft_metrics['score']:.4f}"
        )
        """

    # =========================
    # Build test samples
    # =========================
    test_full_df = test_df_raw.copy()
    test_full_df[SEQ_NUM_COLS] = scaler.transform(test_full_df[SEQ_NUM_COLS])

    test_full_df = apply_category_maps(
        test_full_df,
        category_maps,
        SEQ_CAT_COLS,
    )

    test_samples = build_test_samples(test_full_df)

    test_loader = make_loader(
        test_samples,
        batch_size=CFG.batch_size,
        shuffle=False,
        is_train=False,
    )

    # =========================
    # Load best model
    # =========================

    final_model = RallyLSTM(
        category_maps,
        CFG.hidden_dim,
        CFG.num_layers,
        CFG.dropout,
    ).to(device)

    final_model.load_state_dict(best_state)

    # =========================
    # Predict test
    # =========================

    rally_uids, action_idx, point_idx, server_prob = predict_test(
        final_model,
        test_loader,
        device,
    )

    submission = pd.DataFrame({
        "rally_uid": rally_uids.astype(int),
        "actionId": [
            inverse_maps["actionId"].get(int(x), 0)
            for x in action_idx
        ],
        "pointId": [
            inverse_maps["pointId"].get(int(x), 0)
            for x in point_idx
        ],
        "serverGetPoint": [
            1 if prob >= 0.6 else 0
            for prob in server_prob
        ],
    })

    submission = submission.sort_values("rally_uid").reset_index(drop=True)

    submission.to_csv(
        out_dir / "submission_single_model.csv",
        index=False,
    )

    print(f"\nSaved submission: {out_dir / 'submission_single_model.csv'}")
    print(submission.head())


if __name__ == "__main__":
    main()