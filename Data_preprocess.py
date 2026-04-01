import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 基本設定
# =========================

GROUP_COL = "rally_uid"
ORDER_COL = "strikeNumber"

# 每個 timestep 會放進模型的類別特徵
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

# 每個 timestep 會放進模型的數值特徵
SEQ_NUM_COLS = [
    "scoreSelf",
    "scoreOther",
    "score_diff",
    "strikeNumber",
    "numberGame",
]

TARGET_ACTION = "actionId"
TARGET_POINT = "pointId"
TARGET_SERVER = "serverGetPoint"


# =========================
# 2. 讀檔 + 基本整理
# =========================

def load_and_sort_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values([GROUP_COL, ORDER_COL]).reset_index(drop=True)
    df["score_diff"] = df["scoreSelf"] - df["scoreOther"]
    return df


# =========================
# 3. 類別 mapping
# =========================

def build_category_maps(
    df: pd.DataFrame,
    cat_cols: List[str]
) -> Dict[str, Dict[int, int]]:
    """
    每個欄位建立: 原始類別值 -> embedding index
    0 保留給 PAD / UNK
    """
    category_maps = {}
    for col in cat_cols:
        unique_values = sorted(df[col].dropna().unique().tolist())
        category_maps[col] = {v: i + 1 for i, v in enumerate(unique_values)}
    return category_maps


def apply_category_maps(
    df: pd.DataFrame,
    category_maps: Dict[str, Dict[int, int]],
    cat_cols: List[str],
) -> pd.DataFrame:
    """
    將類別欄位轉成 embedding index
    未知值一律給 0
    """
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].map(category_maps[col]).fillna(0).astype(np.int64)
    return df


# =========================
# 4. train / test 樣本建立
# =========================

def build_lstm_train_samples(df: pd.DataFrame) -> List[dict]:
    """
    一個 rally -> 一筆 train sample

    input:
        前 T-1 拍 sequence
    target:
        第 T 拍的 actionId / pointId / serverGetPoint
    """
    samples: List[dict] = []

    grouped = df.groupby(GROUP_COL, sort=False)
    for rally_uid, g in grouped:
        g = g.sort_values(ORDER_COL).reset_index(drop=True)

        # 至少要有 2 拍，才有 prefix -> target
        if len(g) < 2:
            continue

        prefix = g.iloc[:-1].copy()
        last_row = g.iloc[-1]

        samples.append({
            "rally_uid": rally_uid,
            "seq_cat": prefix[SEQ_CAT_COLS].to_numpy(dtype=np.int64),
            "seq_num": prefix[SEQ_NUM_COLS].to_numpy(dtype=np.float32),
            "seq_len": len(prefix),
            "y_action": int(last_row[TARGET_ACTION]),
            "y_point": int(last_row[TARGET_POINT]),
            "y_server": float(last_row[TARGET_SERVER]),
        })

    return samples


def build_lstm_test_samples(df: pd.DataFrame) -> List[dict]:
    """
    一個 rally -> 一筆 test sample

    test.csv 沒有預測目標，因此整個 rally 都是可見 prefix。
    """
    samples: List[dict] = []

    grouped = df.groupby(GROUP_COL, sort=False)
    for rally_uid, g in grouped:
        g = g.sort_values(ORDER_COL).reset_index(drop=True)
        if len(g) < 1:
            continue

        samples.append({
            "rally_uid": rally_uid,
            "seq_cat": g[SEQ_CAT_COLS].to_numpy(dtype=np.int64),
            "seq_num": g[SEQ_NUM_COLS].to_numpy(dtype=np.float32),
            "seq_len": len(g),
        })

    return samples


# =========================
# 5. 封裝 preprocess
# =========================

def preprocess_train_for_lstm(train_path: str | Path):
    df = load_and_sort_dataframe(train_path)
    category_maps = build_category_maps(df, SEQ_CAT_COLS)
    df_encoded = apply_category_maps(df, category_maps, SEQ_CAT_COLS)
    samples = build_lstm_train_samples(df_encoded)
    return df_encoded, samples, category_maps


def preprocess_test_for_lstm(test_path: str | Path, category_maps: Dict[str, Dict[int, int]]):
    df = load_and_sort_dataframe(test_path)
    df_encoded = apply_category_maps(df, category_maps, SEQ_CAT_COLS)
    samples = build_lstm_test_samples(df_encoded)
    return df_encoded, samples


# =========================
# 6. Dataset + Collate + DataLoader
# =========================

class RallyDataset(Dataset):
    def __init__(self, samples: List[dict], is_train: bool = True):
        self.samples = samples
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class RallyCollator:
    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def __call__(self, batch: List[dict]) -> dict:
        # 由長到短排序，方便之後接 pack_padded_sequence
        batch = sorted(batch, key=lambda x: x["seq_len"], reverse=True)

        seq_cat_list = [torch.tensor(x["seq_cat"], dtype=torch.long) for x in batch]
        seq_num_list = [torch.tensor(x["seq_num"], dtype=torch.float32) for x in batch]
        seq_len = torch.tensor([x["seq_len"] for x in batch], dtype=torch.long)
        rally_uid = torch.tensor([x["rally_uid"] for x in batch], dtype=torch.long)

        padded_seq_cat = pad_sequence(seq_cat_list, batch_first=True, padding_value=0)
        padded_seq_num = pad_sequence(seq_num_list, batch_first=True, padding_value=0.0)

        batch_dict = {
            "seq_cat": padded_seq_cat,   # [B, T, D_cat]
            "seq_num": padded_seq_num,   # [B, T, D_num]
            "seq_len": seq_len,          # [B]
            "rally_uid": rally_uid,      # [B]
        }

        if self.is_train:
            batch_dict["y_action"] = torch.tensor([x["y_action"] for x in batch], dtype=torch.long)
            batch_dict["y_point"] = torch.tensor([x["y_point"] for x in batch], dtype=torch.long)
            batch_dict["y_server"] = torch.tensor([x["y_server"] for x in batch], dtype=torch.float32)

        return batch_dict

def make_dataloader(samples: List[dict], batch_size: int = 32, shuffle: bool = False, is_train: bool = True) -> DataLoader:
    dataset = RallyDataset(samples=samples, is_train=is_train)
    collator = RallyCollator(is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )


# =========================
# 7. Demo
# =========================

if __name__ == "__main__":
    train_path = Path("Data/train.csv")
    test_path = Path("Data/test.csv")

    train_df_encoded, train_samples, category_maps = preprocess_train_for_lstm(train_path)
    test_df_encoded, test_samples = preprocess_test_for_lstm(test_path, category_maps)

    print("=== Train summary ===")
    print("總筆數:", len(train_df_encoded))
    print("rally 樣本數:", len(train_samples))
    print("第一筆 sample 的 rally_uid:", train_samples[0]["rally_uid"] )
    print("第一筆 sample 的 seq_cat shape:", train_samples[0]["seq_cat"].shape)
    print("第一筆 sample 的 seq_num shape:", train_samples[0]["seq_num"].shape)
    print("第一筆 sample 的 seq_len:", train_samples[0]["seq_len"])
    print("第一筆 sample 的 y_action:", train_samples[0]["y_action"])
    print("第一筆 sample 的 y_point:", train_samples[0]["y_point"])
    print("第一筆 sample 的 y_server:", train_samples[0]["y_server"])

    train_loader = make_dataloader(train_samples, batch_size=32, shuffle=True, is_train=True)
    batch = next(iter(train_loader))

    print("\n=== Train batch summary ===")
    print("seq_cat shape:", batch["seq_cat"].shape)
    print("seq_num shape:", batch["seq_num"].shape)
    print("seq_len shape:", batch["seq_len"].shape)
    print("y_action shape:", batch["y_action"].shape)
    print("y_point shape:", batch["y_point"].shape)
    print("y_server shape:", batch["y_server"].shape)
    print("前5個 seq_len:", batch["seq_len"][:5])

    test_loader = make_dataloader(test_samples, batch_size=32, shuffle=False, is_train=False)
    test_batch = next(iter(test_loader))

    print("\n=== Test batch summary ===")
    print("seq_cat shape:", test_batch["seq_cat"].shape)
    print("seq_num shape:", test_batch["seq_num"].shape)
    print("seq_len shape:", test_batch["seq_len"].shape)
    import pickle
    with open("output_data/category_maps.pkl", "wb") as f:
        pickle.dump(category_maps, f)
    with open("output_data/train_samples.pkl", "wb") as f:
        pickle.dump(train_samples, f)
    with open("output_data/test_samples.pkl", "wb") as f:
        pickle.dump(test_samples, f)
