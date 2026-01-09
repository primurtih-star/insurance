from backend.core.dataset_service import load_dataset, add_row

def get_dataset():
    return load_dataset()

def add_dataset_row(entry):
    return add_row(entry)