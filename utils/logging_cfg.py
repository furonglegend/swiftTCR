import logging
import os
from datetime import datetime


def setup_logger(log_dir, name="experiment"):
    """
    Configure experiment logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(name)
