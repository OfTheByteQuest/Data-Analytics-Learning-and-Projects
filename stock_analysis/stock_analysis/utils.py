"""Utility functions for stock analysis."""

from functools import wraps
import re

import pandas as pd


def _santize_label(label):
    """
    CLean up a label by removing non-letter, non-space characters and
    putting in all lowercase with underscore replacing spaces.

    Parameters
    ----------
    label : String
        The text you want to fix.

    Returns
    -------
    re.sub(r'[^\w\s]', '', label).lower().replace(' ', '_')

    """
    return re.sub(r'[^\w\s]', '', label).upper().replace(' ', '_')

def label_sanitizer(method):
    """
    Decorator around a method that reutrns a dataframe to
    clean up all labels in said dataframe (column names and index
    name) by removing non-letter, non-space characters and
    putting in all lowercase with underscores replacing spaces.

    Parameters
    ----------
    method : types.FucntionType
        The method to wrap.

    Returns
    -------
    A decorated method or function

    """
    
    @wraps(method)
    def method_wrapper(self, *args, **kwargs):
        df = method(self, *args, **kwargs)
        
        # fix the column names
        df.columns = [_santize_label(col) for col in df.columns]
        
        # fix the index name
        df.index.rename(_santize_label(df.index.name), inplace=True)
        
        return df
    return method_wrapper

