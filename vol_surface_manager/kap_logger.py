import sys
import os

import logging


def acquire_logger(config):
    format = '%(levelname)8s - %(message)s'
    logging.basicConfig(format=format)
    logging.root.setLevel(logging.WARNING)
    logger = logging.getLogger('KAP')
    if config.LOG == 'INFO':
        logger.setLevel(logging.INFO)
    elif config.LOG == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    else:
        assert(config.LOG == 'ERROR')
        logger.setLevel(logging.ERROR)
    return logger
