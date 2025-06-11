"""Gathering select stock data"""

import datetime as dt
import re

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

from .utils import label_sanitizer, scrape_bitcoin_data

# yf.pdr_override()  # Override pandas datareader with yfinance


class StockReader:
    """Class for reading financial data from websites"""

    _index_tickers = {"S&P500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI"}

    def __init__(self, start, end=None):
        """
        Create a StockReader object for reading accross a given date
        range.

        Parameters:
            - start: The first date to include, as a datetime object
              or a string in the format 'YYYYMMDD'.
            - end: The last date to include, as a datetime object or
              string in the format 'YYYYMMDD'. Defaults to today if
              not provided.
        """
        self.start, self.end = map(
            lambda x: x.strftime("%Y%m%d")
            if isinstance(x, dt.date)
            else re.sub(r"\D", "", x),
            [start, end or dt.date.today()],
        )
        if self.start >= self.end:
            raise ValueError(
                f"Start date {self.start} must be before end date {self.end}."
            )

    @property
    def available_tickers(self):
        """
        Access the names of the indices whose tickers are supported.
        """
        return list(self._index_tickers.keys())

    @classmethod
    def get_index_ticker(cls, index):
        """
        Get the ticker of the specified index, if known.

        Parameters:
            -index: The name of the index; check `available_tickers`
                    property for full list which includes:
                        - 'SP500' for S&P 500,
                        - 'DOW' for Dow Jones Industrial Average,
                        - 'NASDAQ' for NASDAQ Composite Index.
        Returns: The ticker as a string if known, otherwise None.
        """
        try:
            index = index.upper()
        except AttributeError:
            raise ValueError("`index` must be a string.")
        return cls._index_tickers.get(index, None)

    @label_sanitizer
    def get_ticker_data(self, ticker):
        """
        Get historical OHLC data for given date range and ticker.

        Parameters:
            - ticker: The stock ticker symbol as a string, e.g. 'AAPL' for Apple Inc.
            Returns: A pandas dataframe of stock data with columns:
                - 'Open': Opening price
                - 'High': Highest price
                - 'Low': Lowest price
                - 'Close': Closing price
                - 'Volume': Volume of trades
        """
        try:
            data = web.DataReader(ticker, "iex", self.start, self.end)
            data.index = pd.to_datetime(data.index)
        except:  # noqa: E722
            data = yf.download(
                ticker,
                dt.datetime.strptime(self.start, "%Y%m%d").strftime("%Y-%m-%d"),
                dt.datetime.strptime(self.end, "%Y%m%d").strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
            data.columns = [col[0] for col in data]
        return data

    @label_sanitizer
    def get_bitcoin_data(self):
        """
        Get bitcoin historical OHLC data from coinmarketcap.com
        for given date range.

        Returns: A pandas dataframe with columns:
            - 'Open': Opening price
            - 'High': Highest price
            - 'Low': Lowest price
            - 'Close': Closing price
            - 'Volume': Volume of trades
        """
        return scrape_bitcoin_data(self.start, self.end).sort_index()

    @label_sanitizer
    def get_index_data(self, index="S&P500"):
        """
        Get historical OHLC data from Yahoo! Finanace for the chosen index
        for given date range.

        Parameters:
            - index: String representing the index name you want data for,
                     supported indices include:
                       - 'SP500' for S&P 500,
                       - 'DOW' for Dow Jones Industrial Average,
                       - 'NASDAQ' for NASDAQ Composite Index.
                     Check the `available_tickers` property for full list.

        Returns: A pandas dataframe with the index data, including columns:
            - 'Open': Opening price
            - 'High': Highest price
            - 'Low': Lowest price
            - 'Close': Closing price
            - 'Volume': Volume of trades
        """

        if index not in self.available_tickers:
            raise ValueError(
                f"Index '{index}' not supported. "
                f"Available indices: {self.available_tickers}."
            )
        sp = yf.download(
            self.get_index_ticker(index),
            dt.datetime.strptime(self.start, "%Y%m%d").strftime("%Y-%m-%d"),
            dt.datetime.strptime(self.end, "%Y%m%d").strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        sp.columns = [col[0] for col in sp]

        return sp
