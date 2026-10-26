import logging
import os
import pickle

import coloredlogs
import torchvision
from matplotlib import pyplot as plt

plt.switch_backend('agg')

def get_logger(logger_name=None):
    if logger_name is not None:
        logger = logging.getLogger(logger_name)
        logger.propagate = 0
    else:
        logger = logging.getLogger("taufikxu")
    return logger

def build_logger(path, logger_name=None):
    # FORMAT = "%(asctime)s|%(levelname)s%(message)s"
    FORMAT = "%(message)s"
    DATEF = "%H-%M-%S"
    logging.basicConfig(format=FORMAT)
    logger = get_logger(logger_name)

    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s;%(levelname)s|%(message)s", "%H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    LEVEL_STYLES = dict(
        debug=dict(color="magenta"),
        info=dict(color="green"),
        verbose=dict(),
        warning=dict(color="blue"),
        error=dict(color="yellow"),
        critical=dict(color="red", bold=True),
    )
    coloredlogs.install(
        level=logging.INFO, fmt=FORMAT, datefmt=DATEF, level_styles=LEVEL_STYLES
    )
    return logger


class Logger(object):
    """
    Helper class for logging.
    Arguments:
        path (str): Path to log file.
    """
    def __init__(self, path):
        self.path = path
        self.transcribe = True
        self.transcribe_list = []
        self.logger = build_logger(path)
        self.log_dir = os.path.dirname(path)
        filename = os.path.join(self.log_dir, 'stats.pkl')
        self.stats = dict()
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.stats = pickle.load(f)
        print('Logging to file: ', self.path)

    def log(self, message):
        if self.transcribe:
            for msg in self.transcribe_list:
                self.logger.info(msg)
            self.transcribe_list.clear()
            self.logger.info(message)
        else:
            self.transcribe_list.append(message)
            print(message)

    def log_stats(self, step, cats=None):
        cats = cats or self.stats.keys()
        lines = []
        for cat in cats:
            if cat not in self.stats:
                continue
            parts = [f"|{cat}:"]
            for k, values in self.stats[cat].items():
                last_step, last_val = values[-1]
                if last_step != step:
                    continue
                if isinstance(last_val, float):
                    fmt = f"{k}:{last_val:.2f}%" if "acc" in k else f"{k}:{last_val:.5f}"
                else:
                    fmt = f"{k}:{last_val}"
                parts.append(fmt)
            if len(parts) > 1:
                lines.append(" ".join(parts))
        if lines:
            self.logger.info("\n".join(lines) + "\n")

    def add(self, category, k, v, global_it, unique=False):
        lst = self.stats.setdefault(category, {}).setdefault(k, [])
        if unique:
            lst[:] = [(it, val) for it, val in lst if it != global_it]
        lst.append((global_it, v))

    def save_stats(self, filename=None):
        if filename is None:
            filename = "stat.pkl"
        filename = os.path.join(self.log_dir, filename)
        with open(filename, mode='wb') as f:
            pickle.dump(self.stats, f)
    def save_images(self, imgs, name=None, class_name=None, nrow=10):
        if class_name is None:
            outdir = self.log_dir
        else:
            outdir = os.path.join(self.log_dir, class_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if isinstance(name, str):
            outfile = os.path.join(outdir, "{}.png".format(name))
        else:
            outfile = os.path.join(outdir, "%08d.png" % name)
        torchvision.utils.save_image(imgs, outfile, nrow=nrow)

    def plot_learning_curve(self, cats = ["train", "test", "eval"]):
        figure = plt.figure()
        for cat in cats:
            for key in ["clean_acc", "adversarial_acc"]:
                dat = self.stats.get(cat, {}).get(key)
                if dat:
                    xs, ys = zip(*dat)
                    plt.plot(xs, ys, "-", label=f"{cat}_{key}")

        plt.legend()
        plt.grid()
        filename = os.path.join(self.log_dir, "acc.png")
        figure.savefig(filename)
        plt.close(figure)

    def plot_loss(self, cats = ["training"]):
        figure = plt.figure()
        for cat in cats:
            dat = self.stats.get(cat, {}).get("loss")
            if dat:
                xs, ys = zip(*dat)
                plt.plot(xs, ys, "-", label=cat)
        plt.legend()
        plt.grid()
        out_path = os.path.join(self.log_dir, f"loss.png")
        figure.savefig(out_path)
        plt.close(figure)


