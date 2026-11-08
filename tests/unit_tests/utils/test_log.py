"""Tests for the logger context fallback."""

import logging

from graphgen.utils.log import get_current_logger


def test_get_current_logger_falls_back_without_context():
    # Outside a Ray worker context no logger is set; the helper must return a
    # usable fallback instead of raising.
    logger = get_current_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "GraphGen"
