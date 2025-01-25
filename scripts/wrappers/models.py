import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        type=str,
        dest="input_file",
        required=True
    )
    parser.add_argument(
        "-o",
        type=str,
        dest="output_file",
        required=True
    )

    args = parser.parse_args()
    return args
