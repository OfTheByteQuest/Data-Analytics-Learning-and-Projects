"""Gathering select stock data."""

import datetime
import re

import pandas as pd
import pandas_datareader.data as web

from .utils import label_sanitizer

class StockReader:
    """Class for reading financial data from websites"""
    
    _index_tickers={
            'SP500': '^GSPC', 'DOW': '^DJI', 'NASDAQ': '^IXIC'
        } # It's a protected/not protected (not name mangled) attribute,
          # which must be used inside the class or it's instances.
    
    def __int__(self, start, end=None):
        """
        Create a StockReader object for reading across a given date
        range.
        
        Parameters:
            - start: The first date to include, as a datetime object
              or a string in the format 'YYYYMMDD'.
            - end: The last date to include, as a datetime object or
               string in the format 'YYYYMMDD'. Default to today if
               not provided.
        """
        
        self.start, self.end = map(
                lambda x: x.strftime('%Y%m%d') if isinstance(
                        x, datetime.date
                    ) else re.sub(r'\D', '',x),
                [start, end or datetime.date.today()]
        )
        if self.start >= self.end:
            raise ValueError('`start` must be before `end`')
            
        @property
        def available_ticker(self):
            """
                Access the names of the indices whose tickers are supported
            """
            return list(self._index_tickers.keys()) # type(self) is not used, as that's a
                                                    # convention for moedifying class variables.
        @classmethod
        def get_index_ticker(cls, index):
            """
               Get the ticker of the specified index, if known.
               
               Parameters:
                   - index: The name of the index; check 'available_tickers'
                            property for full list which inlcudes:
                                - 'SP500' for S&P 500,
                                - 'DOW' for Dow anf Jones Industrial Average,
                                - 'NASDAQ' for NASDAQ Composite Index
              Returns: The ticker as s string if known, otherwise None.
            """
            try:
                index = index.upper()
            except:
                raise ValueError('`index` must be a string')
            return cls._index_ticker.get(index, None)
            
        @label_sanitizer
        def get_ticker_data(self, ticker):
            pass
        
        @label_sanitizer
        def get_bitcoin_data(self):
            """
            Get Bitcoin historical OHLC data from coinmarketcap.com
            for a given date range.

            Returns
            -------
            A dataframe with the bitcoin data

            """
            return pd.read_html(
                'https://coinmarketcap.com/'
                'currencies/bitocoin/historical-data/?'
                'start={}&end={}'.format(self.start, self.end),
                parse_dates=[0],
                index_col=[0]
                )[0].sort_index()
        
        @label_sanitizer
        def get_index_data(self, index='SP500'):
            """
            Get historical OHLC data from Yahoo! Finance for the choosen index
            for given date range.

            Parameters
            ----------
            - index: String
                String representing the index you want data for,
                supported indeices includes:
                    - 'SP500' for S&P 500,
                    - 'DOW' for Dow Jones Industrial Average,
                    - 'NASDAQ' for NASDAQ Composie Index
                Check the `available_ticker` property for more.

            Returns
            -------
            A pandas dataframe with the index data.

            """
            if index not in self.available_tickers:
                raise ValueError(
                    'Index not supported. Available tickers'
                    f"are: {', '.join(self.available_tickers)}"
                    )
            return web.get_data_yahoo(
                self.get_index_ticker(index), self.start, self.end
                )
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            