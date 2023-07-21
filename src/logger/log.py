import logging
import time

from src.utils.tool import create_file


class Logger:
    def __init__(self, cfg):
        self.verbosity = cfg.LOGGER.VERBOSITY
        self.name = cfg.LOGGER.NAME
        self.level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        self.log_path = './Logs/'
        create_file(self.log_path)
        self.rq = '{}-{}'.format(cfg.LOGGER.NAME, time.strftime('%Y-%m-%d-%H-%M-%S'))
        self.filename = self.log_path + self.rq + '.log'
        self.formatter = logging.Formatter("%(asctime)s|%(levelname)8s|%(filename)10s|%(lineno)4s|%(message)s")

    def get_logger(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level_dict[self.verbosity])

        fh = logging.FileHandler(self.filename, "w")
        fh.setFormatter(self.formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(self.formatter)
        logger.addHandler(sh)

        return logger


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    # Assuming you have a YAML configuration file named "config.yaml"
    with open('../../config/cfg.yaml') as f:
        config_ = yaml.safe_load(f)
    print(config_)
    # Create an instance of the Logger class
    logger_instance = Logger(EasyDict(config_))

    # Get the logger object
    logger = logger_instance.get_logger()

    # Test the logger by writing some log messages
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
