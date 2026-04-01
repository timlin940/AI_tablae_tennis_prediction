import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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

@dataclass # 各種設定參數集中在這裡，方便管理和調整
class Config:
    train_path: str = "Data/train.csv"
    test_path: str = "Data/test.csv"
    output_dir: str = "output_data"
    batch_size: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 15
    val_size: float = 0.2
    random_state: int = 42
    use_class_weight: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

GROUP_COL = "rally_uid"
ORDER_COL = "strikeNumber"

SEQ_CAT_COLS = [
    "sex",
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
    "strikeNumber",
    "numberGame",
]

# 預測目標欄位
TARGET_ACTION = "actionId"
TARGET_POINT = "pointId"
TARGET_SERVER = "serverGetPoint"

# 設定隨機種子，確保實驗可重現
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 路徑確保函式，會建立資料夾（如果不存在的話）
def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# 載入資料並排序，確保每個 rally 的資料是按照 strikeNumber 的順序排列的
def load_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values([GROUP_COL, ORDER_COL]).reset_index(drop=True)
    df["score_diff"] = df["scoreSelf"] - df["scoreOther"]
    return df

# 把類別值轉成 embedding index，0 留給 padding 和未知值
def build_category_maps(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[int, int]]:
    maps: Dict[str, Dict[int, int]] = {}
    for col in cat_cols:
        vals = sorted(df[col].dropna().unique().tolist())
        maps[col] = {v: i + 1 for i, v in enumerate(vals)}  # 0 留給 padding 和未知值
    return maps

# 把 DataFrame 中的類別欄位轉成 embedding index，數值欄位保持不變
def apply_category_maps(df: pd.DataFrame, maps: Dict[str, Dict[int, int]], cat_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].map(maps[col]).fillna(0).astype(np.int64)
    return df


def build_train_samples(df: pd.DataFrame) -> List[dict]:
    samples: List[dict] = []
    for rally_uid, g in df.groupby(GROUP_COL, sort=False):
        g = g.sort_values(ORDER_COL).reset_index(drop=True)
        if len(g) < 2:
            continue
        prefix = g.iloc[:-1]
        last_row = g.iloc[-1]
        samples.append({
            "rally_uid": int(rally_uid),
            "seq_cat": prefix[SEQ_CAT_COLS].to_numpy(dtype=np.int64),
            "seq_num": prefix[SEQ_NUM_COLS].to_numpy(dtype=np.float32),
            "seq_len": len(prefix),
            "y_action": int(last_row[TARGET_ACTION]),
            "y_point": int(last_row[TARGET_POINT]),
            "y_server": float(last_row[TARGET_SERVER]),
        })
    return samples


def build_test_samples(df: pd.DataFrame) -> List[dict]:
    samples: List[dict] = []
    for rally_uid, g in df.groupby(GROUP_COL, sort=False):
        g = g.sort_values(ORDER_COL).reset_index(drop=True)
        if len(g) < 1:
            continue
        samples.append({
            "rally_uid": int(rally_uid),
            "seq_cat": g[SEQ_CAT_COLS].to_numpy(dtype=np.int64),
            "seq_num": g[SEQ_NUM_COLS].to_numpy(dtype=np.float32),
            "seq_len": len(g),
        })
    return samples


# =========================
# Dataset / DataLoader
# =========================

class RallyDataset(Dataset):
    def __init__(self, samples: List[dict], is_train: bool) -> None:
        self.samples = samples
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class RallyCollator:
    def __init__(self, is_train: bool) -> None:
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
        seq_len = torch.tensor([x["seq_len"] for x in batch], dtype=torch.long)
        rally_uid = torch.tensor([x["rally_uid"] for x in batch], dtype=torch.long)

        out = {
            "seq_cat": seq_cat,
            "seq_num": seq_num,
            "seq_len": seq_len,
            "rally_uid": rally_uid,
        }
        if self.is_train:
            out["y_action"] = torch.tensor([x["y_action"] for x in batch], dtype=torch.long)
            out["y_point"] = torch.tensor([x["y_point"] for x in batch], dtype=torch.long)
            out["y_server"] = torch.tensor([x["y_server"] for x in batch], dtype=torch.float32)
        return out


def make_loader(samples: List[dict], batch_size: int, shuffle: bool, is_train: bool) -> DataLoader:
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

def get_emb_dim(cardinality: int) -> int:
    return min(32, max(4, int(math.sqrt(cardinality) + 1)))


class RallyLSTM(nn.Module):
    def __init__(self, category_maps: Dict[str, Dict[int, int]], hidden_dim: int, num_layers: int, dropout: float):
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
        self.action_head = nn.Linear(hidden_dim, len(category_maps["actionId"]) + 1)
        self.point_head = nn.Linear(hidden_dim, len(category_maps["pointId"]) + 1)
        self.server_head = nn.Linear(hidden_dim, 1)

    def forward(self, seq_cat: torch.Tensor, seq_num: torch.Tensor, seq_len: torch.Tensor):
        embs = []
        for i, col in enumerate(self.cat_cols):
            embs.append(self.embeddings[col](seq_cat[:, :, i]))
        x = torch.cat(embs + [seq_num], dim=-1)

        packed = pack_padded_sequence(x, seq_len.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(packed)
        h = self.dropout(h_n[-1])

        return (
            self.action_head(h),
            self.point_head(h),
            self.server_head(h).squeeze(-1),
        )


# =========================
# Training utilities
# =========================

def make_class_weight(encoded_labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(encoded_labels), minlength=num_classes)
    counts[0] = 0  # ignore padding idx if output has it
    weights = np.zeros_like(counts, dtype=np.float32)
    valid = counts > 0
    weights[valid] = counts[valid].sum() / counts[valid]
    if valid.any():
        weights[valid] = weights[valid] / weights[valid].mean()
    weights[0] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def competition_score(y_action_true, y_action_pred, y_point_true, y_point_pred, y_server_true, y_server_prob):
    action_f1 = f1_score(y_action_true, y_action_pred, average="macro")
    point_f1 = f1_score(y_point_true, y_point_pred, average="macro")
    try:
        server_auc = roc_auc_score(y_server_true, y_server_prob)
    except ValueError:
        server_auc = float("nan")
    total = float("nan") if np.isnan(server_auc) else 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
    return {
        "action_macro_f1": action_f1,
        "point_macro_f1": point_f1,
        "server_auc": server_auc,
        "score": total,
    }


def run_epoch(model, loader, optimizer, crit_action, crit_point, crit_server, device, train: bool):
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
            action_logits, point_logits, server_logit = model(seq_cat, seq_num, seq_len)
            loss_action = crit_action(action_logits, y_action)
            loss_point = crit_point(point_logits, y_point)
            loss_server = crit_server(server_logit, y_server)
            loss = 0.4 * loss_action + 0.4 * loss_point + 0.2 * loss_server
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * seq_cat.size(0)
        action_true.extend(y_action.cpu().numpy())
        action_pred.extend(action_logits.argmax(dim=1).cpu().numpy())
        point_true.extend(y_point.cpu().numpy())
        point_pred.extend(point_logits.argmax(dim=1).cpu().numpy())
        server_true.extend(y_server.cpu().numpy())
        server_prob.extend(torch.sigmoid(server_logit).detach().cpu().numpy())

    metrics = competition_score(
        np.asarray(action_true), np.asarray(action_pred),
        np.asarray(point_true), np.asarray(point_pred),
        np.asarray(server_true), np.asarray(server_prob),
    )
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def predict_test(model, loader, device):
    model.eval()
    rally_uids, action_idx, point_idx, server_prob = [], [], [], []
    for batch in loader:
        seq_cat = batch["seq_cat"].to(device)
        seq_num = batch["seq_num"].to(device)
        seq_len = batch["seq_len"].to(device)
        a_logit, p_logit, s_logit = model(seq_cat, seq_num, seq_len)
        rally_uids.extend(batch["rally_uid"].cpu().numpy())
        action_idx.extend(a_logit.argmax(dim=1).cpu().numpy())
        point_idx.extend(p_logit.argmax(dim=1).cpu().numpy())
        server_prob.extend(torch.sigmoid(s_logit).cpu().numpy())
    return np.asarray(rally_uids), np.asarray(action_idx), np.asarray(point_idx), np.asarray(server_prob)


# =========================
# Main
# =========================

def main():
    set_seed(CFG.random_state)
    out_dir = ensure_dir(CFG.output_dir)
    device = torch.device(CFG.device)
    print(f"Using device: {device}")

    # 抓取並處理資料
    train_df_raw = load_df(CFG.train_path)
    test_df_raw = load_df(CFG.test_path)

    category_maps = build_category_maps(train_df_raw, SEQ_CAT_COLS)
    inverse_maps = {col: {v: k for k, v in m.items()} for col, m in category_maps.items()}

    # split by rally to avoid leakage
    rally_ids = train_df_raw[GROUP_COL].drop_duplicates().to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=CFG.val_size, random_state=CFG.random_state)
    tr_idx, va_idx = next(splitter.split(rally_ids, groups=rally_ids))
    tr_rallies = set(rally_ids[tr_idx])
    va_rallies = set(rally_ids[va_idx])

    train_df = train_df_raw[train_df_raw[GROUP_COL].isin(tr_rallies)].copy()
    valid_df = train_df_raw[train_df_raw[GROUP_COL].isin(va_rallies)].copy()

    scaler = StandardScaler()
    train_df[SEQ_NUM_COLS] = scaler.fit_transform(train_df[SEQ_NUM_COLS])
    valid_df[SEQ_NUM_COLS] = scaler.transform(valid_df[SEQ_NUM_COLS])

    train_df = apply_category_maps(train_df, category_maps, SEQ_CAT_COLS)
    valid_df = apply_category_maps(valid_df, category_maps, SEQ_CAT_COLS)

    train_samples = build_train_samples(train_df)
    valid_samples = build_train_samples(valid_df)

    train_loader = make_loader(train_samples, CFG.batch_size, shuffle=True, is_train=True)
    valid_loader = make_loader(valid_samples, CFG.batch_size, shuffle=False, is_train=True)

    model = RallyLSTM(category_maps, CFG.hidden_dim, CFG.num_layers, CFG.dropout).to(device)

    if CFG.use_class_weight:
        action_w = make_class_weight([s["y_action"] for s in train_samples], len(category_maps["actionId"]) + 1).to(device)
        point_w = make_class_weight([s["y_point"] for s in train_samples], len(category_maps["pointId"]) + 1).to(device)
    else:
        action_w = None
        point_w = None

    crit_action = nn.CrossEntropyLoss(weight=action_w)
    crit_point = nn.CrossEntropyLoss(weight=point_w)
    crit_server = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    best_score = -np.inf
    best_state = None

    for epoch in range(1, CFG.epochs + 1):
        tr_metrics = run_epoch(model, train_loader, optimizer, crit_action, crit_point, crit_server, device, train=True)
        va_metrics = run_epoch(model, valid_loader, optimizer, crit_action, crit_point, crit_server, device, train=False)

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
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Validation did not produce a usable model.")

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
        out_dir / "best_model.pt",
    )
    print(f"Saved best model. best_valid_score={best_score:.4f}")

    # retrain on full train with scaler fitted on full train
    full_scaler = StandardScaler()
    train_full_df = train_df_raw.copy()
    train_full_df[SEQ_NUM_COLS] = full_scaler.fit_transform(train_full_df[SEQ_NUM_COLS])
    test_full_df = test_df_raw.copy()
    test_full_df[SEQ_NUM_COLS] = full_scaler.transform(test_full_df[SEQ_NUM_COLS])

    train_full_df = apply_category_maps(train_full_df, category_maps, SEQ_CAT_COLS)
    test_full_df = apply_category_maps(test_full_df, category_maps, SEQ_CAT_COLS)

    full_train_samples = build_train_samples(train_full_df)
    test_samples = build_test_samples(test_full_df)

    full_train_loader = make_loader(full_train_samples, CFG.batch_size, shuffle=True, is_train=True)
    test_loader = make_loader(test_samples, CFG.batch_size, shuffle=False, is_train=False)

    final_model = RallyLSTM(category_maps, CFG.hidden_dim, CFG.num_layers, CFG.dropout).to(device)
    final_model.load_state_dict(best_state)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    if CFG.use_class_weight:
        full_action_w = make_class_weight([s["y_action"] for s in full_train_samples], len(category_maps["actionId"]) + 1).to(device)
        full_point_w = make_class_weight([s["y_point"] for s in full_train_samples], len(category_maps["pointId"]) + 1).to(device)
    else:
        full_action_w = None
        full_point_w = None

    full_crit_action = nn.CrossEntropyLoss(weight=full_action_w)
    full_crit_point = nn.CrossEntropyLoss(weight=full_point_w)
    full_crit_server = nn.BCEWithLogitsLoss()

    # light fine-tuning on full train starting from best validation model
    fine_tune_epochs = max(3, CFG.epochs // 3)
    for epoch in range(1, fine_tune_epochs + 1):
        ft_metrics = run_epoch(final_model, full_train_loader, final_optimizer, full_crit_action, full_crit_point, full_crit_server, device, train=True)
        print(f"Full-train fine-tune {epoch:02d} | loss={ft_metrics['loss']:.4f}")

    rally_uids, action_idx, point_idx, server_prob = predict_test(final_model, test_loader, device)

    serverGetPoint_threshold = 0.7
    serverGetPoint_pred = (server_prob >= serverGetPoint_threshold).astype(int)
    submission = pd.DataFrame({
        "rally_uid": rally_uids.astype(int),
        "actionId": [inverse_maps["actionId"].get(int(x), 0) for x in action_idx],
        "pointId": [inverse_maps["pointId"].get(int(x), 0) for x in point_idx],
        "serverGetPoint": serverGetPoint_pred
    }).sort_values("rally_uid").reset_index(drop=True)

    submission.to_csv(out_dir / "submission.csv", index=False)
    print(f"Saved submission: {out_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
