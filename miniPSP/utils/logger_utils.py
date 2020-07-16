import os
import os.path as osp
import logging
import time

DEFAULT_LEVEL = logging.DEBUG
DEFAULT_LOGGING_DIR = "logs"
fh = None
ch = None


def update_default_level(default_level):
    global DEFAULT_LEVEL
    DEFAULT_LEVEL = default_level

def update_default_logging_dir(default_logging_dir):
    global DEFAULT_LOGGING_DIR
    DEFAULT_LOGGING_DIR = default_logging_dir

def strftime(t = None):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t or time.time()))

def init_fh():
    global fh
    if fh is not None:
        return
    if DEFAULT_LOGGING_DIR is None:
        return
    if not osp.exists(DEFAULT_LOGGING_DIR): os.makedirs(DEFAULT_LOGGING_DIR)
    logging_path = osp.join(DEFAULT_LOGGING_DIR, strftime() + ".log")
    fh = logging.FileHandler(logging_path)
    fh.setLevel(DEFAULT_LEVEL)
    formatter1 = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter1)



def init_ch():
    global ch
    if ch is not None:
        return
    ch = logging.StreamHandler()
    ch.setLevel(DEFAULT_LEVEL)
    formatter2 = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter2)

def get_logger(level=DEFAULT_LEVEL):

    #Logger created
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    #Tensorflow also uses logging leading to double logging. Prevention ->
    logger.propagate = False

    #File handler

    init_fh()
    if fh is not None:
        logger.addHandler(fh)

    #Console handler
    init_ch()
    if ch is not None:
        logger.addHandler(ch)


    return logger
