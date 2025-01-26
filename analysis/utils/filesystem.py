import os


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
