from dsdag.core.parameter import BaseParameter
from dsdag.core.op import OpVertex

import dask
from dask.diagnostics import ProgressBar
import inspect



class DaskParallel(OpVertex):
    """
    Wraps an Op in a dask.delayed object
    """
    parallel_op = BaseParameter()
    op_kwargs = BaseParameter(dict())

    def _node_color(self):
        return '#2fbc2d'

    def _node_shape(self):
        #return 'doublecircle'
        return 'doubleoctagon'

    def requires(self):
        if inspect.isclass(self.parallel_op):
            # instantiate with provided keyword args
            self.inst_op = self.parallel_op(**self.op_kwargs)
        else:
            # already instantiated
            self.inst_op = self.parallel_op

        return self.inst_op.requires()

    def run(self, *args, **kwargs):
        return dask.delayed(self.inst_op.run)(*args, **kwargs)

    def get_name(self):
        #orig_name = super(DaskParallel, self).get_name()
        try:
            orig_name = self.parallel_op.get_name()
            return "DaskParallel_%s" % orig_name
        except:
            return super(DaskParallel, self).get_name()


class DaskCollect(OpVertex):
    """
    Calls dask.compute on a collection of dask delayed objects
    """
    dask_progress_bar = BaseParameter(True,
                                      "Include a diagnostic Progressbar from dask")
    num_workers = BaseParameter(4, "Number of dask workers")
    scheduler = BaseParameter('processes', "Dask scheduler option")

    def _node_color(self):
        return '#2fbc2d'

    def run(self, *ops):
        compute_kwargs = dict(scheduler=self.scheduler,
                              num_workers=self.num_workers)
        if self.dask_progress_bar:
            with ProgressBar():
                results = dask.compute(*ops, **compute_kwargs)
        else:
            results = dask.compute(*ops, **compute_kwargs)

        return results