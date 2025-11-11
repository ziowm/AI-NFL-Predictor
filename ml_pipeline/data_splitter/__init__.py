"""
Data Splitting Module

Splits multi-season data into train, validation, and test sets maintaining temporal order.
"""

from .data_splitter import TemporalDataSplitter

__all__ = ['TemporalDataSplitter']
