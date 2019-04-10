import os


def get_sentences(input_file):
    lines = []
    with open(input_file, "r") as _f:
        for line in _f:
            lines.append(line)
    return lines


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            print("Unable to create {}".format(dir_path))
