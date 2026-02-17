# -*- coding: utf-8 -*-
# *** indent: 4 spaces ***
"""Description of component."""
import argparse
import logging
import os

import yaml


def eval_fn(cfg: dict, logger: logging.Logger) -> None:
    """Actual operation of the script, based on config dict."""
    logging.info("Nothing to do in this sample script. [END]")
    return


def init_logger(name, log_level="INFO", log_filename=None) -> logging.Logger:
    """Init logger."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger(name)
    handlers = [logging.StreamHandler()]

    if log_filename:
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s | %(module)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logger


def get_config(args: argparse.Namespace, logger: logging.Logger) -> dict:
    """Parse args and config values."""
    cfg_path = args.config
    if not cfg_path:
        logger.warning("No config path was provided. Convert argparse.Namespace to dictionary.")
        cfg = vars(args)
    else:
        with open(cfg_path, "r") as fp:
            cfg = yaml.safe_load(fp)
    return cfg


def parse_args() -> argparse.Namespace:
    """Define and parse command line arguments."""
    parser = argparse.ArgumentParser(description="""This is a sample script with basic settings.""")

    parser.add_argument("--config", "-c", default=None, help="Path to the YAML config file.")

    parser.add_argument(
        "--log",
        "-l",
        help="Logging level to output, supported by python logger: https://docs.python.org/3.8/howto/logging.html",
        default="INFO",
    )

    return parser.parse_args()


def main() -> None:
    """Include all ops here."""
    args = parse_args()
    logger = init_logger("sample script", log_level=args.log, log_filename=os.path.join("logs", "log.txt"))
    cfg = get_config(args, logger)
    eval_fn(cfg, logger)
    return


if __name__ == "__main__":
    main()
