from pathlib import Path

from config import output_dir
from utils import visualize_losses_from_file

output_dir = Path(output_dir)

for file in output_dir.glob("*/*results.pkl"):
    print("Visualizing", file)
    output_path = file.parent / "loss.png"
    try:
        algorithm_name = file.parent.name.upper()
        print(file, str(file.parent))
        visualize_losses_from_file(
            str(file),
            output_path=str(output_path),
            title=f"{algorithm_name} Losses",
        )
    except Exception as e:
        print(e)
        print("Skipping..")
