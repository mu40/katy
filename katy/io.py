"""IO module."""

import functools
import importlib
import json
import pathlib
import torch


def save(data, path):
    """Serialize data and write it to a file.

    Save data as a JSON, JSONL, or PyTorch file.

    Parameters
    ----------
    data : object
        Data. An iterable of JSON-serializable records for JSONL.
    path : os.PathLike
        File path, with suffix .json, .jsonl, or .pt.

    """
    path = pathlib.Path(path)
    temp = path.with_suffix(path.suffix + '.temp')

    try:
        if path.suffix == '.json':
            # Indent and add trailing newline for readability.
            with open(temp, mode='w', encoding='utf-8') as f:
                json.dump(data, f, allow_nan=False, indent=4)
                f.write('\n')

        elif path.suffix == '.jsonl':
            if isinstance(data, (dict, str, bytes)):
                raise TypeError('.jsonl expects an iterable of records')

            # One record per line. Always '\n', disable newline translation.
            with open(temp, mode='w', encoding='utf-8', newline='') as f:
                for record in data:
                    f.write(json.dumps(record, allow_nan=False) + '\n')

        elif path.suffix == '.pt':
            torch.save(data, temp)

        else:
            raise ValueError(f'suffix of "{path}" not .json, .jsonl, or .pt')

    # Clean up temporary file on CTRL+C.
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

    temp.replace(path)


def load(path):
    """Deserialize data from a file.

    Load data from a JSON, JSONL, or PyTorch file.

    Parameters
    ----------
    path : os.PathLike
        File path, with suffix .json, .jsonl, or .pt.

    Returns
    -------
    data : object
        Data. A list for JSONL.

    """
    path = pathlib.Path(path)
    if path.suffix == '.json':
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    if path.suffix == '.jsonl':
        with open(path, encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    if path.suffix == '.pt':
        return torch.load(path, weights_only=True)

    raise ValueError(f'suffix of "{path}" is not .json, .jsonl, or .pt')


def read_colors(lut):
    """Read a FreeSurfer color lookup table.

    The function returns a dictionary associating labels of type `int` with
    dictionaries of keys 'name' and 'color'. Key 'name' will map to a string,
    'color' to a tuple of integers defining RGB values between 0 and 255.

    Parameters
    ----------
    lut : os.PathLike
        Path to lookup table.

    Returns
    -------
    dict
        Color table.

    """
    with open(lut) as f:
        lines = f.read().splitlines()

    lut = {}
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue

        label, name, *color = line.split()
        label = int(label)
        color = tuple(int(c) for c in color[:3])
        lut[label] = {'name': name, 'color': color}

    return lut


@functools.cache
def default_colors():
    """Return the default FreeSurfer color lookup table.

    Returns
    -------
    dict
        Color table.

    """
    root = importlib.resources.files(__package__)
    path = root / 'data' / 'FreeSurferColorLUT.txt'
    return read_colors(path)
