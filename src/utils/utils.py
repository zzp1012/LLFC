import os
import datetime
import argparse
import logging
import random
import numpy as np
import torch

def get_datetime() -> str:
    """get the date.
    Returns:
        date (str): the date.
    """
    datetime_ = datetime.datetime.now().strftime("%m%d-%H%M%S")
    return datetime_


def set_logger(save_path: str) -> None:
    """set the logger.
    Args:
        save_path(str): the path for saving logfile.txt
        name(str): the name of the logger
        verbose(bool): if true, will print to console.

    Returns:
        None
    """
    # set the logger
    logfile = os.path.join(save_path, "logfile.txt")
    logging.basicConfig(filename=logfile,
                        filemode="w+",
                        format='%(name)-12s: %(levelname)-8s %(message)s',
                        datefmt="%H:%M:%S",
                        level=logging.INFO)
    # define a Handler which writes DEBUG messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # tell the handler to use this format
    console.setFormatter(logging.Formatter(
        '%(name)-12s: %(levelname)-8s %(message)s'))
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def get_logger(name:str,
               verbose:bool = True) -> logging.Logger:
    """get the logger.
    Args:
        name (str): the name of the logger
        verbose (bool): if true, will print to console.
    Returns:
        logger (logging.Logger)
    """
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    if not verbose:
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = 0) -> None:
    """set the random seed for multiple packages.
    Args:
        seed (int): the seed.

    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device(device: int) -> torch.device:
    """set GPU device.
    Args:
        device (int) the number of GPU device

    Returns:
        device (torch.device)
    """
    logger = get_logger(__name__)
    if torch.cuda.is_available():
        if device >= torch.cuda.device_count():
            logger.error("CUDA error, invalid device ordinal")
            exit(1)
    else:
        logger.error("Plz choose other machine with GPU to run the program")
        exit(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    device = torch.device("cuda:" + str(device))
    logger.info(device) 
    return device


def log_settings(args: argparse.Namespace, config: dict = {}) -> None:
    """log the settings of the program. 
    Args:
        args (argparse.Namespace): the arguments.
        config (dict): the config.
    """
    logger = get_logger(__name__)
    hyperparameters = {
        **args.__dict__, 
        **{key: value for key, value in config.items() \
            if key.isupper() and type(value) in [int, float, str, bool, dict]}
    }
    logger.info(hyperparameters)


def save_current_src(save_path: str) -> None:
    """save the current src.
    Args:
        save_path (str): the path to save the current src.
        src_path (str): the path to the current src.
    Returns:
        None
    """
    logger = get_logger(__name__)
    logger.info("save the current src")
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.system("cp -r {} {}".format(src_path, save_path))
    script_path = os.path.join(os.path.dirname(src_path), "scripts")
    os.system("cp -r {} {}".format(script_path, save_path))