import os


def listdir_absolute(directory: str):

    files = os.listdir(directory)
    absolute_paths = [os.path.abspath(os.path.join(directory, file)) for file in files]
    return absolute_paths