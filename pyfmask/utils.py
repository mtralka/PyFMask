from typing import Union
from pathlib import Path


def valdiate_path(
    path: Union[str, Path],
    check_exists: bool = False,
    check_is_file: bool = False,
    check_is_dir=False,
) -> Path:

    valid_path: Path = Path(path) if isinstance(path, str) else path

    if check_exists:
        if not valid_path.exists():
            raise FileExistsError(f"{path} must exist")

    if check_is_file:
        if not valid_path.is_file():
            raise FileNotFoundError(f"{path} must be a file")

    if check_is_dir:
        if not valid_path.is_dir():
            raise ValueError(f"{path} must be a directory")

    return valid_path
