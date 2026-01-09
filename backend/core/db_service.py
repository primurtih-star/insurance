import json
import os

def read_json(path, default=None):
    """
    Membaca file JSON.
    Jika file tidak ada → kembalikan default atau [].
    """
    if not os.path.exists(path):
        return default if default is not None else []

    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default if default is not None else []


def write_json(path, data):
    """
    Menulis JSON dengan membuat folder secara otomatis.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    return True


def append_json(path, entry):
    """
    Menambahkan satu baris ke json list.
    """
    data = read_json(path, default=[])
    data.append(entry)
    write_json(path, data)
    return True
