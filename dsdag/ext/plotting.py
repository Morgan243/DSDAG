import matplotlib
from matplotlib import pyplot as plt
from dsdag.core.parameter import BaseParameter
from dsdag.core.op import OpVertex
import numpy as np

class Subplots(OpVertex):
    nrows=BaseParameter(1)
    ncols=BaseParameter(1)
    sharex=BaseParameter(False)
    sharey=BaseParameter(False)
    #squeeze=BaseParameter(True)
    subplot_kw=BaseParameter(None)
    gridspec_kw=BaseParameter(None)
    figsize=BaseParameter((12, 6))
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

class HistMulti(OpVertex):
    title = BaseParameter('')
    figsize = BaseParameter((8, 5))
    xlabel = BaseParameter('x')
    ylabel = BaseParameter('y')
    logx = BaseParameter(False)
    logy = BaseParameter(False)
    passthrough = BaseParameter(False, "If False, returns axis of plot rather ")
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

from dsdag.core.op import OpVertexAttr, opvertex, opattr
from dsdag.core.op import opattr as parameter

@opvertex
class ContourPlot(OpVertexAttr):
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
                     ax=None, cbar=False, title=''):

        from matplotlib.mlab import griddata
        if ax is None:
            fig, ax = plt.subplots()

        x = _s_df['x']
        y = _s_df['y']
        z = _s_df[value_column]

        # define grid.
        _s_2d_df = _s_df.dropna().pivot(index='x', columns='y', values=value_column)

        _s_stacked_df = _s_2d_df.stack(dropna=False).rename(value_column).reset_index()

        # return _s_2d_df
        xi = _s_2d_df.index.tolist()
        yi = _s_2d_df.columns.tolist()
        zi = griddata(x, y, z, xi, yi, interp='linear')

        vmax = abs(zi).max() if vmax is None else vmax
        vmin = -abs(zi).max() if vmin is None else vmin
        zi = np.clip(zi, vmin, vmax)

        ctr = ax.contourf(xi, yi, zi, levels,
                          cmap=cmap, alpha=alpha,
                          vmax=vmax, vmin=vmin)
        if cbar:
            #plt.colorbar(mappable=ctr)
            plt.colorbar(ctr, ax=ax)

        #plt_s_df = _s_df.sort_values(value_column).tail(5)
        #ax.scatter(x=plt_s_df['x'], y=plt_s_df['y'],
        #           cmap='green', s=5, alpha=0.6,
        #           c=dot_color,
        #           )

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
