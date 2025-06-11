"""Classes for technical analysis of assets."""

import math

import pandas as pd

from .utils import validate_df


class StockAnalyser:
    """
    Class for providing metrics for technical analysis of a stock.
    IMPORTNAT: The index of the data is needed to a sorted time-series.
    """

    @validate_df(columns={"open", "close", "low", "close"})
    def __init__(self, df):
        self.data = df

    @property
    def close(self):
        """Get the close column of the data."""
        return self.data.close

    @property
    def pct_change(self):
        """Get the percent change of the close column."""
        return self.close.pct_change()

    @property
    def pivot_point(self):
        """Calculate the pivot point for support/resistance levels."""
        return (self.last_close + self.last_low + self.last_high) / 3

    @property
    def last_close(self):
        """Get the value of the last close in the data."""
        return self.data["close"].iloc[-1]

    @property
    def last_high(self):
        """Get the value of the last close in the data."""
        return self.data["high"].iloc[-1]

    @property
    def last_low(self):
        """Get the value of the last close in the data."""
        return self.data["low"].iloc[-1]

    def resistance(self, level=1):
        """Calculate the resistance at the given level.

        Parameters:
            - level: The resistance level (1, 2, or 3)

        Returns: The resistace value.
        """
        if level == 1:
            return (2 * self.pivot_point) - self.last_low
        elif level == 2:
            return self.pivot_point + (self.last_high - self.last_low)
        elif level == 3:
            return self.last_high + 2 * (self.pivot_point - self.last_low)
        else:
            raise ValueError("Not a valid level.  Must be 1, 2, or 3")

    def support(self, level):
        """
        Calculate the support at the given level.

        Parameters:
            - level: The support level (1, 2, or 3)

        Returns: The support value for the given level.
        """

        if level == 1:
            return (2 * self.pivot_point) - self.last_high
        elif level == 2:
            return self.pivot_point - (self.last_high - self.last_low)
        elif level == 3:
            return self.last_low + 2 * (self.last_high - self.pivot_point)
        else:
            raise ValueError("Not a valid level. Must be 1, 2 or 3.")

    @property
    def _max_periods(self):
        """Get the number of trading periods in the data."""
        return self.data.shape[0]

    def daily_std(self, periods=252):
        """
        Calculate the daily standard deviation of percent change.

        Paramters:
            - periods: The number of periods to use for the
                       calculation; default in 252 for the trading days
                       in a year. Note if you provide a number greater
                       than the number of trading periods in the data,
                       self._max_periods will be used instead.

        Returns: The standard deviation
        """
        return self.pct_change.iloc[
            min(periods, self._max_periods) * -1 :
        ].std()  # Negative indexing is being used to select the most
        # recent data.

    def annualized_volatility(self):
        """Calculate the annualized volatility."""
        return self.daily_std() * math.sqrt(252)

    def volatility(self, periods=252):
        """
        Calculate the rolling volatility.

        Parameters:
            - periods: The number of periods to use for the
                       calculations; default is 252 for the trading
                       days in a year. Note if you provide a number
                       greater than the number of trading periods in
                       the data, self._max_periods will be used instead.

        Returns: A pandas series.
        """
        periods = min(periods, self._max_periods)
        return self.close.rolling(periods).std() / math.sqrt(periods)

    def corr_with(self, other):
        """
        Calculate teh correlations between this dataframe and another.

        Parameters:
            - other: The other dataframe.

        Returns: A pandas series.
        """
        return self.data.corrwith(other)

    def cv(self):
        """
        Calculate the coefficient of variation for the asset. Note
        that the lower this is, the better the risk/retrun tradeoff.
        """
        return self.close.std() / self.close.mean()

    def qcd(self):
        q1, q3 = self.close.quantile([0.25, 0.75])
        return (q3 - q1) / (q3 + q1)

    def beta(self, index):
        """
        Calculate the beta of the asset.

        Parameters:
            - index: The dataframe for the index to compare to.

        Returns: Beta, a float
        """
        index_change = index.close.pct_change()
        beta = self.pct_change.cov(index_change) / index_change.var()
        return beta

    def cummulative_returns(self):
        """
        Calculate the series of cummulative returns for plotting.
        """
        return (1 + self.pct_change).cumprod()

    @staticmethod
    def port_return(df):
        """
        Calculate the return assuming no distribution per share.

        Parameters:
            - df: The asset's dataframe.

        Returns: The return, as a float.
        """
        start, end = df.close.iloc[0], df.close.iloc[-1]
        return (end - start) / start

    def alpha(self, index, r_f):
        """
        Calculate the asset's alpha

        Paramters:
            - index: The index to compare to.
            - r_f: The risk_free rate of return.

        Returns: Alpha, as a float.
        """

        r_f /= 100
        r_m = self.port_return(index)
        beta = self.beta(index)
        r = self.port_return(self.data)
        alpha = r - r_f - beta * (r_m - r_f)
        return alpha

    def is_bear_market(self):
        """
        Determine if a stock is in a bear market, meaning its
        return in the last 2 month is a decline of 20% or more.
        """
        end_date = self.data.index.max()
        start_date = end_date - pd.DateOffset(months=2)
        recent_data = self.data.iloc[self.data.index >= start_date]
        return self.port_return(recent_data) <= -0.2

    def is_bull_run(self):
        """
        Determine of a stock is in a bull market, meaning its
        return in the last 2 months is a increase of 20% or more.
        """
        end_date = self.data.index.max()
        start_date = end_date - pd.DateOffset(months=2)
        recent_data = self.data.iloc[self.data.index >= start_date]
        return self.port_return(recent_data) >= 0.2

    def sharpe_ratio(self, r_f):
        """
        Calculate the asset's shapre ratio.

        Parameters:
            - r_f: The risk-free rate of return.

        Returns: The sharpe ratio, as a float.
        """
        return (self.cummulative_returns().iloc[-1] - r_f) / (
            self.cummulative_returns().std()
        )

    def __getitem__(self, attr):
        return getattr(self, attr)


class AssetGroupAnalyzer:
    """Analyze many assets in a dataframe."""

    @validate_df(columns={"open", "high", "low", "close"})
    def __init__(self, df, group_by="name"):
        self.data = df
        self.group_by = group_by

        if group_by not in self.data.columns:
            raise ValueError(f'`groub_by` column "{group_by}" not in dataframe.')
        self.analyzers = self._composition_handler()

    def _composition_handler(self):
        """
        Create a dictionry mapping each group to its analyzer,
        taking advantage of composition instead of inheritance.
        """
        return {
            group: StockAnalyser(data)
            for group, data in self.data.groupby(self.group_by)
        }

    def analyze(self, func_name, **kwargs):
        """
        Run a StockAnalyzer mathod on all assets in the group.

        Parameters:
            - func_name: The name of the method to run.
            - kwargs: Additional keyword arguments to pass to the
                      function.

        Returns: A dictionary mapping each asset to the result of the
                 calculation of that fuction.
        """
        if not hasattr(StockAnalyser, func_name):
            raise ValueError(f'StockAnalyzer has no "{func_name}" method.')
        if not kwargs:
            kwargs = {}

        return {
            group: analyzer[func_name](**kwargs)  # Works since, __getitem__ is defined
            for group, analyzer in self.analyzers.items()
        }
