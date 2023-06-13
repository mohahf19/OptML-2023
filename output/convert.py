import pickle
import torch
from pathlib import Path

to_float = lambda x: x.item() if torch.is_tensor(x) else x

num_runs = 5
experiment = "svrg_runs_bs128_lr1e-02"

for run in range(num_runs):
    pickled_data = open(f"{experiment}/{run}/train_data.pkl", "rb")
    results = pickle.load(pickled_data)

    conv_results = dict()
    for k in results.keys():
          conv_results[k] = list(map(to_float, results[k]))

    output_dir = Path(f"converted/{experiment}/{run}")
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "train_data.pkl", "wb") as f:
            pickle.dump(conv_results, f)