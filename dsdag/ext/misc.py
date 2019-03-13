#from dsdag.core.op import OpVertex
from dsdag.core.op import OpVertexAttr as OpVertex
#from dsdag.core.parameter import BaseParameter, UnhashableParameter
from dsdag.core.op import opattr, opvertex
import dsdag




@opvertex
class InputOp(OpVertex):
    obj = opattr(help_msg="The object to wrap and return")

    def run(self):
        return self.obj

@opvertex
class VarOp(OpVertex):
    obj = opattr(help_msg="The object to wrap and return")

    def _node_color(self):
        return '#b70043'

    def _node_shape(self):
        return 'box'

    def requires(self):
        return dict()

    def run(self):
        return self.obj


@opvertex
class LambdaOp(OpVertex):
    f = opattr(help_msg="Function that is applied to the input")

    def _node_color(self):
        return '#d65768'

    def requires(self):
        raise NotImplementedError("Incomplete LambdaOp - must be applied")

    def run(self, *args, **kwargs):
        return self.f(*args, **kwargs)

@opvertex
class Select(OpVertex):
    i = opattr(0)
    _never_cache = True
    def run(self, l):
        return l[self.i]