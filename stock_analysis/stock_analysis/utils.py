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

def group_stock(mapping):
    """
    Create a new dataframe with many assets and a new column
    indicating the asset that that row's data belongs to.

    Parameters
    ----------
    mapping : Dictionary
        A key-value mapping of the form
        {asset_name: asset_df}

    Returns
    -------
    A new pandas DataFrame.

    """
    group_df = pd.DataFrame()
    
    for stock, stock_data in mapping.items():
        df = stock_data.copy(deep=True)
        df['name'] = stock
        group_df = group_df.concat(df, sort=True, ignore_index=True)

    group_df.index = pd.to_datetime(group_df.index)
    
    return group_df

def validate_df(columns, instance_method = True):
    """
    Decorator that raises a ValueError if input isn't a pandas
    DataFrame or doesn't contain the proper columns. Note the
    DataFrame must be the first positional argument passed to this method.
    
    """
    def method_wrapper(method):
        @wraps(method)
        def validate_wrapper(self, *args, **kwargs):
            # functions and static methods don't pass self
            # so self is the first postional argument in that case
            df = (self, *args) [0 if not instance_method else 1]
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError('Must pass in a pandas DataFrame')
                
            if columns.difference(df.columns):
                raise ValueError(
                    'DataFrame must contain the following columns:'
                    f'{columns}'
                    )
            return method(self, *args, **kwargs)
        return validate_wrapper
    return method_wrapper


@validate_df(columns={'name'}, instance_method=False)
def describe_group(data):
    """
    Run `describe()` on the asset group created with `group_stocks()`.

    Parameters
    ----------
    data : pd.DataFrame
        The group data resulting from `group_stocks()`

    Returns
    -------
    The transpose of the grouped description statistics.

    """
    return data.groupby('name').describe().T









































