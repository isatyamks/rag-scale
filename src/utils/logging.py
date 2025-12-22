

import logging


def setup_logging(level: str = "INFO", log_format: str = "%(asctime)s [%(levelname)s] %(message)s") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format
    )
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
