"""Visualize financial instruments."""

import math
# from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


from .utils import validate_df


class Visualizer:
    """Base visualizer class not intended for direct use."""

    @validate_df(columns={"open", "high", "low", "close"})
    def __init__(self, df):
        """Visualizer has a pandas dataframe as an attribute"""
        self.data = df

    @staticmethod
    def add_reference_line(ax, x=None, y=None, **kwargs):
        """
        Static method for adding reference lines to plots

        Parameters:
            - ax: matplotlib Axes object to add the reference line to.
            - x, y: The x, y value or numpy-array to draw the line as a
                    single value or numpy-array like structure.
                    - For horizontal: pass only `y`
                    - For vertical: pass only `x`
                    - For AB line: pass both `x` and `y`
            - kwargs: Additional keyword arguments to pass down.

        Returns: The Axes object with the reference line added.
        """
        if x is not None and y is not None:
            # When both x and y are provided, assume they are array-like and plot as a line.
            try:
                if hasattr(x, "shape") or hasattr(y, "shape"):
                    ax.plot(x, y, **kwargs)
                else:
                    raise ValueError("x and y must be array-like for a reference line.")
            except Exception as e:
                raise ValueError("Error plotting AB line: " + str(e))
        elif x is not None:
            ax.axvline(x, **kwargs)
        elif y is not None:
            ax.axhline(y, **kwargs)
        else:
            raise ValueError("Either x or y must be provided.")
        ax.legend()
        return ax

    @staticmethod
    def shade_region(ax, x=tuple(), y=tuple(), **kwargs):
        """
        Static method for shading a region in a plot.

        Parameters:
            - ax: Matplotlib Axes object to shade the region to.
            - x: Tuple with the `xmin` and `xmax` bounds for the
                 rectangle drawn vertically.
            - y: Tuple with the `ymin` and `ymax` bounds for the
                 rectangle drawn horizontally.
            - kwargs: Additional keyword arguments to pass down.
        Returns: The Axes object with the shaded region added.
        """

        if not x and not y:
            raise ValueError("Either x or y must be provided.")
        elif x and y:
            ax.add_patch(
                Rectangle(
                    (x[0], y[0]),  # lower left corner
                    x[1] - x[0],  # width
                    y[1] - y[0],  # height
                    **kwargs,
                )
            )
        elif x and not y:
            ax.axvspan(*x, **kwargs)
        elif not x and y:
            ax.axhspan(*y, **kwargs)
        return ax

    @staticmethod
    def _iter_handler(items):
        """
        Static method for making a list out of a item of it isn't a
        list or tuple already.

        Parameters:
            - items: The varibale to make sure it is a list.

        Returns: The input as a list or tuple.
        """
        if not isinstance(items, (list, tuple)):
            items = [items]
        return items

    def _window_calc_func(self, column, periods, name, func, named_arg, **kwargs):
        """
        To be implemented by subclasses. Define how to add lines
        resulting from window claculations.
        """
        raise NotImplementedError("To be implemented by subclasses!")

    def moving_average(self, column, periods, **kwargs):
        """
        Add line(s) for the moving avergae of a column

        Parameters:
            - column: The name of the column to plot.
            - periods: The rule or list of rules for resampling,
                       like '20D' for 20-day periods.
            - kwatgs: Additional arguments to pass down to the
                      plotting function.

        Returns: A matplotlib Axes object.
        """
        return self._window_calc_func(
            column,
            periods,
            name="MA",
            func=pd.DataFrame.rolling,
            named_arg="window",
            **kwargs,
        )

    def exp_smoothing(self, column, periods, **kwargs):
        """
        Add line(s) for the exponential smoothed moving average
        of a column.

        Parameters:
            - column: The names of the column to plot.
            - periods: The spanc or list of spans for smoothing,
                       like 20 for 20-period smoothing.
            - kwargs: Additional arguments to pass down to the
                      plotting function.

        Returns: A matplotlib Axes object.
        """
        return self._window_calc_func(
            column,
            periods,
            name="EWMA",
            func=pd.DataFrame.ewm,
            named_arg="span",
            **kwargs,
        )

    def evolution_over_time(self, column, **kwargs):
        """
        To be implemented by subclasses to create line plots.
        """
        raise NotImplementedError("To be implemented by subclasses!")

    def boxplot(self, **kwargs):
        """
        To be implemented by subclasses to create boxplots.
        """
        raise NotImplementedError("To be implemented by subclasses!")

    def histogram(self, column, **kwargs):
        """
        To be implemented by subclasses to create histograms.
        """
        raise NotImplementedError("To be implemented by subclasses!")

    def after_hours_trades(self):
        """
        To be implemented by subclasses.
        """
        raise NotImplementedError("To be implemented by subclasses!")

    def pairplot(self):
        """
        To be implemented by subclasses.
        """
        raise NotImplementedError("To be implemented by subclasses!")


class StockVisualizer(Visualizer):
    """Visualizer for a single stock."""

    def evolution_over_time(self, column, **kwargs):
        """
        Visualizer the evolution over time of a column.

            Parameters:
                - column: The name of the column to visualize.
                - kwargs: Additional keyword arguements to pass
                          down to the plotting function.

                Returns: A matplotlib Axes object.
        """
        return self.data.plot(
            y=column,
            **kwargs,
        )

    def boxplot(self, **kwargs):
        """
        Generate box plots for all the columns.

        Parameters:
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns: A metplotlib Axes object.
        """
        return self.data.plot(kind="box", **kwargs)

    def histogram(self, column, **kwargs):
        """
        Generate the hsitogram for a given column.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns: A matplotlib Axes object.
        """
        return self.data.plot.hist(
            y=column,
            **kwargs,
        )

    def trade_volume(self, tight=False, **kwargs):
        """
        Visualize the trade volume and closing price.

        Parameters:
            - tight: Whether or not to attempt to match up the
                     resampled bar plot on the bpttom to the line plot
                     on the top.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns: A matplotlib Axes object.
        """
        _, axes = plt.subplots(2, 1, figsize=(15, 15))
        self.data.close.plot(
            ax=axes[0],
            title="Closing Price",
        ).set_ylabel("Price ($)")
        monthly_volume = self.data.volume.resample("1M").sum()
        monthly_volume.index = monthly_volume.index.strftime("%b\n%Y")
        monthly_volume.plot(
            kind="bar", ax=axes[1], color="blue", rot=0, title="Volume Traded"
        ).set_ylabel("Volume Traded")
        if tight:
            axes[0].set_xlim(self.data.index.min(), self.data.index.max())
            axes[1].set_xlim(-0.25, axes[1].get_xlim()[1] - 0.25)
        return axes

    def after_hours_trades(self):
        """
        Visualize the effects of after-hours trading on this asset.

        Returns: A matplotlib Axes object.
        """
        after_hours = self.data.open - self.data.close.shift()

        monthly_effect = after_hours.resample("1M").sum()
        fig, axes = plt.subplots(1, 2, figsize=(15, 3))

        after_hours.plot(
            ax=axes[0], title="After hours trading\n(Open Price - Prior Day's Close)"
        ).set_ylabel("Price")

        monthly_effect.index = monthly_effect.index.strftime("%b")
        monthly_effect.plot(
            ax=axes[1],
            kind="bar",
            rot=90,
            title="After hours trading monthly effects",
            color=np.where(monthly_effect >= 0, "g", "r"),
        ).axhline(0, color="black", linewidth=1)
        axes[1].set_ylabel("Price")
        return axes

    def open_to_close(self, figsize=(10, 4)):
        """
        Visulalize the daily change froopen to close price.

        Parameter:
            -figsize: A tuple of (width, height) for plot
                      dimensions.

        Returns: A matplotlib Figure object.
        """
        is_higher = self.data.close - self.data.open > 0

        fig = plt.figure(figsize=figsize)

        for exclude_mask, color, label in zip(
            (is_higher, np.invert(is_higher)), ("g", "r"), ("price rose", "price fell")
        ):
            plt.fill_between(
                self.data.index,
                self.data.open,
                self.data.close,
                figure=fig,
                where=exclude_mask,
                color=color,
                label=label,
            )
        plt.suptitle("Daily price change (open to close)")
        plt.legend()
        plt.xlabel("Data")
        plt.ylabel("Price")
        plt.close()
        return fig

    def fill_between_other(self, other_df, figsize=(10, 4)):
        """
        Visualize the differenc in closing price between assets.

        Parameters:
            - other_df: The dataframe with the other asset's data.
            - figsize: A tuple  of (width, height) for the plot
                       dimensions.

        Return: A matplotlib Figure object.
        """
        is_higher = self.data.close - other_df.close > 0

        fig = plt.figure(figsize=figsize)

        for exclude_mask, color, label in zip(
            (is_higher, np.invert(is_higher)),
            ("g", "r"),
            ("asset is higher", "asset is lower"),
        ):
            plt.fill_between(
                self.data.index,
                self.data.close,
                other_df.close,
                figure=fig,
                where=exclude_mask,
                color=color,
                label=label,
            )
        plt.suptitle("Differential between asset closing price (this - other)")
        plt.legend()
        plt.close()
        return fig

    def _window_calc_func(self, column, periods, name, func, named_arg, **kwargs):
        """
        Helper method for plotting a series and adding reference lines
        using a window calculation.

        Parameters:
            - column: The name of the column to plot.
            - periods: The rule/span or list of them to pass to the
                       resampling/smoothing function, like '20D' for 20-day
                       peridos (resampling) or 20 for a 20-day span (smoothing)
            - name: The name of the window claculation function (to show in
                    the legend).
            - func: The window calculation function.
            - named_arg: The name of the arguments `periods` is being
                         passed as.
            - kawargs: Additional arguments to pass down to the plotting
                       function.

        Returns: A matplotlib Axes object.
        """
        ax = self.data.plot(y=column, **kwargs)
        for period in self._iter_handler(periods):
            self.data[column].pipe(func, **{named_arg: period}).mean().plot(
                ax=ax,
                linestyle="--",
                label=f"""{period if isinstance(period, str) else str(period) + "D"} {
                    name
                }""",
            )
        plt.legend()
        return ax

    def pariplot(self, **kwargs):
        """
        Generate a seaborn pairplot for this asset.

        Parameters"
            - kwargs:  Keyword arguments to pass down to
                       `sns.pairplot()`

        Returns: A seaborn pariplot.
        """
        return sns.pairplot(self.data, **kwargs)

    def jointplot(self, other, column, **kwargs):
        """
        Generate a seaorn jointplot for the given column in asset
        compared to another asset.

        Parameter:
            - other: The other asset's dataframe.
            - column: The column name to use for the comparison.
            - kwargs: Keyword arguments to pass down to
                      `sns.jointplot()`

        Returns: A seaborn jointplot.
        """
        return sns.jointplot(x=self.data[column], y=other[column], **kwargs)

    def correlation_heatmap(self, other):
        """
        Plot the correlations between the same column between this
        asset and another one with a heatmap.

        Parameters:
            - other: The other dataframe.

        Returns: A seaborn heatmap.
        """
        corrs = self.data.corrwith(other)
        corrs = corrs[~pd.isnull(corrs)]
        size = len(corrs)
        matrix = np.zeros((size, size), float)
        for i, corr in zip(range(size), corrs):
            matrix[i][i] = corr

        # create mask to only show diagonal
        mask = np.ones_like(matrix)
        np.fill_diagonal(mask, 0)

        return sns.heatmap(
            matrix,
            annot=True,
            center=0,
            mask=mask,
            xticklabels=self.data.columns,
            yticklabels=self.data.columns,
        )


class AssetGroupVisualizer(Visualizer):
    """
    Class for visualizing groups of asset in a single dataframe.
    """

    # override for group visuals
    def __init__(self, df, group_by="name"):
        """This object keeps track of which column to group by."""
        super().__init__(df)
        self.group_by = group_by

    def evolution_over_time(self, column, **kwargs):
        """
        Visualize the evolution over time of a column for all assets.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down to the
                      plotting function.

        Returns: A matplotlib Axes object.
        """
        if "ax" not in kwargs:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        else:
            ax = kwargs.pop("ax")
        return sns.lineplot(
            x=self.data.index,
            y=column,
            hue=self.group_by,
            data=self.data,
            ax=ax,
            **kwargs,
        )

    def boxplot(self, column, **kwargs):
        """
        Generate boxplots for a given column in all assets.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns: A matplotlib Axes object.
        """
        return sns.boxplot(x=self.group_by, y=column, data=self.data, **kwargs)

    def _get_layout(self):
        """
        Helper method for getting an autolayout of subplots.

        Returns: The matplotlib Figure and Axes objects to plot with.
        """
        subplots_needed = self.data[self.group_by].nunique()
        rows = math.ceil(subplots_needed / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        if rows > 1:
            axes = axes.flatten()
        if subplots_needed < len(axes):
            # remove excess axes from autolayout
            for i in range(subplots_needed, len(axes)):
                # can't use comprehension here
                fig.delaxes(axes[i])
        return fig, axes

    def histogram(self, column, **kwargs):
        """
        Generate the histogram of a given column for all assets using displot.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns: A seaborn FacetGrid.
        """
        return sns.displot(data=self.data, x=column, col=self.group_by, **kwargs)

    def _window_calc_func(self, column, periods, name, func, named_arg, **kwargs):
        """
        Helper method for plotting a series and adding reference lines
        using a window calculation.

        Prameters:
            - column: The name of the column to plot.
            - periods: The rule/span or list of them to pass to the
              resampling/smoothing function, like, '20D' for 20-day
              periods (resampling) or 20 for 20-day span (smoothing)
            - name: The name of the window calculation (to show in
              the legend).
            - func: The window calculation function.
            - named_arg: The name of the arguement `periods` is being
              passed as.
            - kwargs: Additional arguments to pass down to the plotting
              function.

        Returns: A matplotlob Axes object.
        """
        fig, axes = self._get_layout()
        for ax, asset_name in zip(axes, self.data[self.group_by].unique()):
            subset = self.data[self.data[self.group_by] == asset_name]
            ax = subset.plot(y=column, ax=ax, label=asset_name, **kwargs)
            for period in self._iter_handler(periods):
                subset[column].pipe(func, **{named_arg: periods}).mean().plot(
                    ax=ax,
                    linestyle="--",
                    label=f"""{
                        period if isinstance(period, str) else str(period) + "D"
                    }{name}""",
                )
            ax.legend()
        return ax

    def after_hours_trade(self):
        """
        Visualize the effects of after hours trading on this asset.
        Returns: A matplotlib Axes obejct.
        """
        num_categorize = self.data[self.group_by].nunique()
        fig, axes = plt.subplots(num_categorize, 2, figsize=(15, 8 * num_categorize))
        for ax, (name, data) in zip(axes, self.data.groupby(self.group_by)):
            after_hours = data.open - data.close.shift()
            monthly_effect = after_hours.resample("1M").sum()

            after_hours.plot(ax=ax[0], title=f"{name} Open Price - Price Day's Close")
            ax[0].set_ylabel("Price")

            monthly_effect.index = monthly_effect.index.strftime("%b")
            monthly_effect.plot(
                ax=ax[1],
                kind="bar",
                title=f"{name} after hous trading monthly effect",
                color=np.where(monthly_effect >= 0, "g", "r"),
            ).axhline(0, color="black", linewidth=1)
            ax[1].set_ylabel("Price")

        return axes

    def pairplot(self, **kwargs):
        """
        Generate a seaborn pairplot for this asset group.
        Parameters:
            - kwargs: Keyword arguments to pass down to
                      `sns.pairplot()`
        Returns: A seaborn pairplot
        """
        return sns.pairplot(
            self.data.pivot_table(
                values="close", index=self.data.index, columns="name"
            ),
            diag_kind="kde",
            **kwargs,
        )

    def heatmap(self, pct_change=False, **kwargs):
        """
        Generate a seaborn heatmap for correlations between assets.

        Parameters:
            - pct_change: Whether or not to show the correlations of
                          the daily percent change in price or just
                          use the closing price.
            - kwargs: Keyword arguments to pass down to
                      `sns.heatmap()`
        Returns: A seaborn heatmap
        """
        pivot = self.data.pivot_table(
            values="close", index=self.data.index, columns="name"
        )
        if pct_change:
            pivot = pivot.pct_change()
        return sns.heatmap(pivot.corr(), annot=True, center=0, **kwargs)
