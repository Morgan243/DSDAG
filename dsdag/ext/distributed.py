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
        return (self.inst_op, dask.delayed(self.inst_op.run)(*args, **kwargs))

    def get_name(self):
        return self.parallel_op.get_name()

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
        self.ops = list()
        self.delays = list()
        for _op, _delay in ops:
            self.ops.append(_op)
            self.delays.append(_delay)

        compute_kwargs = dict(scheduler=self.scheduler,
                              num_workers=self.num_workers)
        if self.dask_progress_bar:
            with ProgressBar():
                results = dask.compute(*self.delays, **compute_kwargs)
        else:
            results = dask.compute(*self.delays, **compute_kwargs)

        return results

def DaskParallelOps(ops=None, kw_ops=None, scheduler='threads', num_workers=2):
    # can't do both
    assert ops is not None or kw_ops is not None
    p_ops = [DaskParallel(parallel_op=o)
                for o in ops]
    c_op = DaskCollect(scheduler=scheduler, num_workers=num_workers)(*p_ops)
    c_op.unpack_output = True
    return c_op
