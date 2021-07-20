import os
import shutil
import random
from pathlib import Path


def main():
    cdcp_path = os.path.dirname(os.getcwd()) + "/data/cdcp/original"
    train_dir = cdcp_path + "/train"
    dev_dir = cdcp_path + "/dev"
    test_dir = cdcp_path + "/test"

    train_file_prefix = {f.split(".")[0] for f in os.listdir(train_dir)}

    random.seed(101)

    dev_list = random.sample(list(train_file_prefix), len(train_file_prefix) // 10)

    Path(dev_dir).mkdir(parents=True, exist_ok=True)

    for f in os.listdir(train_dir):

        if f.split(".")[0] in dev_list:
            if '.ann' in f:
                shutil.move(train_dir + "/" + f, dev_dir + "/" + f)
            if '.txt' in f and '.pipe' not in f and '.json' not in f:
                shutil.move(train_dir + "/" + f, dev_dir + "/" + f)

    for f in os.listdir(train_dir):
        if '.txt.json' in f or '.txt.pipe' in f:
            os.remove(train_dir + "/" + f)

    for f in os.listdir(test_dir):
        if '.txt.json' in f or '.txt.pipe' in f:
            os.remove(test_dir + "/" + f)

    train_docs = len({f.split(".")[0] for f in os.listdir(train_dir)})
    dev_docs = len({f.split(".")[0] for f in os.listdir(dev_dir)})
    test_docs = len({f.split(".")[0] for f in os.listdir(test_dir)})

    print("Train docs: {}".format(train_docs))
    print("Dev docs:   {}".format(dev_docs))
    print("Test docs:  {}".format(test_docs))
    print("Total docs: {}".format(train_docs + dev_docs + test_docs))


if __name__ == "__main__":
    main()
