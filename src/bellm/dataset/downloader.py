import sys
from pathlib import Path

from bellm.dataset.downloaders.allenai_c4 import download_c4
from bellm.dataset.downloaders.open_assistant_oasst2 import download_oasst


def download_foundation_model_datasets(root: Path):
    download_c4(root / "foundation")


def download_instruction_model_datasets(root: Path):
    # psyche/glaiveai-reasoning-v1-20m
    # psyche/MultiSynt-MT-Reasoning
    # DataMuncher-Labs/UltraMath-Reasoning-Small
    # MaLA-LM/mala-code-reasoning
    # https://huggingface.co/datasets/ianncity/KIMI-K2.5-1000000x
    # https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered
    # https://huggingface.co/datasets/Modotte/CodeX-2M-Thinking
    # zake7749/Qwen3-Coder-Next-Open-Code-SFT
    # DCAgent/c1_gpt53_codex
    # ronantakizawa/github-codereview
    # Zigeng/DMax-LLaDA-2.0-Mini-Code-Trajectories
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
