from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset

def load_ncc_dataset():
    """
    Load the NCC dataset from Hugging Face.

    Returns:
        Dataset: The loaded NCC dataset.
    """
    ds = load_dataset("NbAiLab/NCC")
    return ds

