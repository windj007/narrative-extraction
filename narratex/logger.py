import logging


LOGGER = logging.getLogger()


DEFAULT_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_FORMATTER = logging.Formatter(DEFAULT_FORMAT)


def setup_logger(out_path=None, use_strerr=True, file_level=logging.DEBUG, stderr_level=logging.INFO):
    LOGGER.handlers = []
    LOGGER.setLevel(min(file_level, stderr_level))

    if use_strerr:
        stderr_handler = logging.StreamHandler()
        stderr_handler.setLevel(stderr_level)
        stderr_handler.setFormatter(DEFAULT_FORMATTER)
        LOGGER.addHandler(stderr_handler)

    if out_path is not None:
        file_handler = logging.FileHandler(out_path)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(DEFAULT_FORMATTER)
        LOGGER.addHandler(file_handler)

    LOGGER.info('Logger initialized')

    return LOGGER
