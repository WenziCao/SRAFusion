from .log import Logger


def build_logger(cfg):
    logger_instance = Logger(cfg)
    logger = logger_instance.get_logger()
    return logger
