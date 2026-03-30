import numpy as np
import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent
project_dir = script_dir.parent

train = pd.read_csv(project_dir / "Data" / "train.csv")
test = pd.read_csv(project_dir / "Data" / "test.csv")
