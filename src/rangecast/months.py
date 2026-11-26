"""Month indexing. Month 1 is the first month of the data; indices are 1-based."""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd


def month_label(first_month: pd.Period, index: int) -> str:
    """Label such as ``2024-03`` for the 1-based month ``index``."""
    return str(first_month + (index - 1))


def days_in_month(first_month: pd.Period, index: int) -> int:
    period = first_month + (index - 1)
    return calendar.monthrange(period.year, period.month)[1]


def month_of_year(first_month: pd.Period, index: int) -> int:
    """Calendar month 0..11 of the 1-based month ``index``."""
    return (first_month + (index - 1)).month - 1


def days_vector(first_month: pd.Period, count: int) -> np.ndarray:
    return np.array([days_in_month(first_month, i) for i in range(1, count + 1)], dtype=float)
