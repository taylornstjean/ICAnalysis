import os
import shutil
import analysis.config as config


def listdir_absolute(directory: str) -> list:
    """
    Lists files in a directory with their absolute paths.

    Args:
        directory (str): The path to the directory to list the contents of.

    Returns:
        list: List of absolute paths to the files contained in the directory.
    """

    files = os.listdir(directory)
    absolute_paths = [os.path.abspath(os.path.join(directory, file)) for file in files]
    return absolute_paths


def reset_temp_dir():
    """Clears the temp directory."""

    directory = os.path.join(config.BASE_DIR, "data/temp")
    shutil.rmtree(directory)  # Deletes everything inside the directory
    os.makedirs(directory, exist_ok=True)
