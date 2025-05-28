import json

from pathlib import Path


def read_run_info(file_path: Path) -> dict:
    """Reads the 'run_information.json' file.

    Args:
        file_path (Path): Path of the 'run_information.json' file.

    Returns:
        dict: Content of the 'run_information.json' file.
    """
    with open(file_path) as f:
        run_info = json.load(f)

    return run_info
