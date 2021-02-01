"""Script to reproduce all the results presented in the QueRyvolution paper.
"""
import logging

import colorlog

from queryvolution.src.common import setup_logger
from queryvolution.src.result_procedures import (
    understand_variance,
    cluster_size_statistics,
    base_results,
    attack_comparison,
    document_set_results,
    countermeasure_results,
    generalization,
    query_distrib_results,
    apache_by_year,
    similar_dataset_size,
    exact_voc_results,
)

logger = colorlog.getLogger("QueRyvolution")


if __name__ == "__main__":
    setup_logger()
    FORMATTER = logging.Formatter("[%(asctime)s %(levelname)s] %(module)s: %(message)s")
    file_handler = logging.FileHandler("results.log")
    file_handler.setFormatter(FORMATTER)
    logger.addHandler(file_handler)

    procedures = (
        base_results,
        attack_comparison,
        document_set_results,
        countermeasure_results,
        generalization,
        query_distrib_results,
        understand_variance,
        apache_by_year,
        similar_dataset_size,
        exact_voc_results,
    )
    for procedure in procedures:
        logger.info(f"Starting procedure {procedure.__name__}")
        procedure()
        logger.info(f"Procedure {procedure.__name__} ended")
