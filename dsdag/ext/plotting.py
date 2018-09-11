import matplotlib
from dsdag.core.parameter import BaseParameter
from dsdag.core.op import OpVertex

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
        fig, ax = matplotlib.pyplot.subplots(figsize=self.figsize)
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
