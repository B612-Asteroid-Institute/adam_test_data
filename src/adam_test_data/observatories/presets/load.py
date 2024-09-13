from importlib.resources import files
from typing import Optional

from ..observatory import Observatory


def load_W84(file: Optional[str] = None) -> Observatory:
    """
    Load CTIO's DECam on the 4m Blanco telescope from a JSON file.

    Parameters
    ----------
    file : str, optional
        The path to the JSON file.

    Returns
    -------
    Observatory
        The observatory object.
    """
    if file is None:
        file = str(
            files("adam_test_data").joinpath("observatories", "presets", "W84.json")
        )

    return Observatory.from_json(file)


def load_695(file: Optional[str] = None) -> Observatory:
    """
    Load the Mayall 4m telescope from a JSON file.

    Parameters
    ----------
    file : str, optional
        The path to the JSON file.

    Returns
    -------
    Observatory
        The observatory object.
    """
    if file is None:
        file = str(
            files("adam_test_data").joinpath("observatories", "presets", "695.json")
        )

    return Observatory.from_json(file)


def load_V00(file: Optional[str] = None) -> Observatory:
    """
    Load the Bok 2.3m telescope from a JSON file.

    Parameters
    ----------
    file : str, optional
        The path to the JSON file.

    Returns
    -------
    Observatory
        The observatory object.
    """
    if file is None:
        file = str(
            files("adam_test_data").joinpath("observatories", "presets", "V00.json")
        )

    return Observatory.from_json(file)


def load_X05(file: Optional[str] = None) -> Observatory:
    """
    Load the Rubin Observatory observatory from a JSON file.

    Parameters
    ----------
    file : str, optional
        The path to the JSON file.

    Returns
    -------
    Observatory
        The observatory object.
    """
    if file is None:
        file = str(
            files("adam_test_data").joinpath("observatories", "presets", "X05.json")
        )

    return Observatory.from_json(file)
