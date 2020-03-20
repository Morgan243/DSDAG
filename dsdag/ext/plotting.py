import matplotlib
from matplotlib import pyplot as plt
#from dsdag.core.parameter import BaseParameter
#from dsdag.core.op import OpVertex
from dsdag.core.op import opvertex as opvertex
from dsdag.core.op import parameter as parameter
import numpy as np

@opvertex
class Subplots:
    nrows=parameter(1)
    ncols=parameter(1)
    sharex=parameter(False)
    sharey=parameter(False)
    #squeeze=parameter(True)
    subplot_kw=parameter(None)
    gridspec_kw=parameter(None)
    figsize=parameter((12, 6))
    _never_cache = True
    #fig_kw = dsdag.core.parameter.BaseParameter(dict())
    def run(self):
        from matplotlib import pyplot as plt
        #self.fig, self.axs = plt.subplots(**{k: v.value for k, v in self.get_parameters()})
        self.fig, self.axs = plt.subplots(nrows=self.nrows, ncols=self.ncols,
                                          sharex=self.sharex, sharey=self.sharey,
                                          figsize=self.figsize,
                                          subplot_kw=self.subplot_kw, gridspec_kw=self.gridspec_kw,
                                          squeeze=False)
        self.axs = [self.axs[r][c] for r in range(self.nrows) for c in range(self.ncols)]
        return self.axs

@opvertex
class HistMulti:
    title = parameter('')
    figsize = parameter((8, 5))
    xlabel = parameter('x')
    ylabel = parameter('y')
    logx = parameter(False)
    logy = parameter(False)
    passthrough = parameter(False, "If False, returns axis of plot rather ")
    # @staticmethod
    def hist_compare(self, **hist_data):
        fig, ax = plt.subplots(figsize=self.figsize)
        hist_kwargs = dict(bins=20, alpha=.4, density=False)

        for d_name, d in hist_data.items():
            ax.hist(d, label=d_name, **hist_kwargs)

        ax.legend(fontsize=15)
        ax.set_title(self.title, fontsize=18)
        ax.set_xlabel(self.xlabel, fontsize=16)
        ax.set_ylabel(self.ylabel, fontsize=16)
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)
        ax.tick_params(labelsize=15)

        if self.logy:
            ax.set_yscale('log')
        if self.logx:
            ax.set_xscale('log')

        if self.passthrough:
            return hist_data
        else:
            return ax

from scipy.interpolate import griddata

@opvertex
class ContourPlot:
    sample_ixes = parameter(None)
    title = parameter('')
    bands = parameter(None)
    value_column = parameter(None)
    dot_color_column = parameter(None)
    cbar = parameter(False)

    _never_cache = True

    @staticmethod
    def contour_plot(_s_df, value_column,
                     dot_color=None,
                     levels=10,
                     cmap=plt.cm.hsv,
                     vmin=None, vmax=None,
                     alpha=1.,
                     ax=None, cbar=False, title='',
                     figsize=(6, 4)):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        points = _s_df[['x', 'y']].values
        values = _s_df[value_column].values

        # define 2d grid of existing values
        _s_2d_df = _s_df.dropna().pivot(index='x', columns='y', values=value_column)
        # restack without dropping NaN to get the unique grid
        _s_stacked_df = _s_2d_df.stack(dropna=False).rename(value_column).reset_index()

        xi = _s_stacked_df[['x', 'y']].values

        zi = griddata(points,
                      values,
                      xi, method='linear')

        vmax = zi[~np.isnan(zi)].max() if vmax is None else vmax
        vmin = zi[~np.isnan(zi)].min() if vmin is None else vmin
        #print((vmin, vmax))
        zi = np.clip(zi, vmin, vmax)
        zi = zi.reshape(_s_2d_df.shape)

        ctr = ax.contourf(_s_2d_df.index.values,
                          _s_2d_df.columns.values,
                          zi.T, cmap=cmap, alpha=alpha,
                          levels=levels,
                          vmax=vmax, vmin=vmin)
        if cbar:
            plt.colorbar(ctr, ax=ax)

        ax.set_title(title)

        return ax


    def run(self, df, ax=None, value_column=None, ):
        value_column = self.value_column if value_column is None else value_column
        plt_df = df.dropna()

        ax = self.contour_plot(plt_df, value_column=value_column, title=self.title, ax=ax,
                              dot_color=plt_df[value_column] if self.dot_color_column is None else plt_df[
                                  self.dot_color_column],
                              cbar=self.cbar)
        return ax
