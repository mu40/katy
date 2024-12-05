"""IO module."""


import json
import os
import pathlib
import pickle
import torch


def save(data, path):
    """Serialize arbitrary data and write it to a file.

    Save data as a JSON, pickle or PyTorch file. Generally prefer JSON for
    interoperability, readability, and safety. Use PyTorch for speed.

    Parameters
    ----------
    data : obj
        Data.
    path : os.PathLike
        File path, with suffix .json, .pickle, or .pt.

    """
    path = pathlib.Path(path)

    # Indent and add training newline for readability on command line.
    if path.suffix == '.json':
        with open(path, mode='w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            f.write(os.linesep)

    elif path.suffix == '.pickle':
        with open(path, mode='wb') as f:
            pickle.dump(data, f)

    elif path.suffix == '.pt':
        torch.save(data, path)

    else:
        raise ValueError(f'suffix of {path} is not .json, .pickle, or .pt')


def load(path):
    """Deserialize data from a file.

    Load data from a JSON, pickle or PyTorch file.

    Parameters
    ----------
    path : os.PathLike
        File path, with suffix .json, .pickle, or .pt.

    Returns
    -------
    data : obj
        Data.

    """
    path = pathlib.Path(path)

    if path.suffix == '.json':
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    if path.suffix == '.pickle':
        with open(path, mode='rb') as f:
            return pickle.load(f)

    if path.suffix == '.pt':
        return torch.load(path, weights_only=True)

    raise ValueError(f'suffix of {path} is not .json, .pickle, or .pt')
