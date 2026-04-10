import sys
from pathlib import Path

from bellm.dataset.downloaders.allenai_c4 import download_c4
from bellm.dataset.downloaders.open_assistant_oasst2 import download_oasst


def download_foundation_model_datasets(root: Path):
    download_c4(root / "foundation")


def download_instruction_model_datasets(root: Path):
    download_oasst(root / "instruction")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("No destination path argument provided")
        sys.exit(1)

    output_path = Path(sys.argv[1])

    confirmation = input(f"Confirm output destination: {output_path}. [Y/n] ").lower()
    if confirmation == "y":
        output_path.mkdir(parents=True, exist_ok=True)

        download_foundation_model_datasets(output_path)
        download_instruction_model_datasets(output_path)

    else:
        print("Aborting.")
