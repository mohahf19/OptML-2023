import pickle
import torch
from pathlib import Path

to_float = lambda x: x.item() if torch.is_tensor(x) and x.shape == torch.Size([]) else x
unpack = lambda x: (x[0][0], to_float(x[0][1])) if type(x[0]) == tuple and len(x[0]) == 2 else (x[1], to_float(x[0]))

num_runs = 4
experiment = "saga_p1_runs_bs1_lr1e-02"

for run in range(num_runs):
    pickled_data = open(f"{experiment}/{run}/train_data.pkl", "rb")
    results = pickle.load(pickled_data)

    conv_results = dict()
    for k in results.keys():
          conv_results[k] = list(map(unpack, zip(results[k], list(range(len(results[k]))))))

    output_dir = Path(f"converted/{experiment}/{run}")
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "train_data.pkl", "wb") as f:
            pickle.dump(conv_results, f)