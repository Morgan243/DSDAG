#from dsdag.core.op import OpVertex
#from dsdag.core.op import OpVertexAttr as OpVertex
#from dsdag.core.parameter import BaseParameter, UnhashableParameter
from dsdag.core.op import parameter, OpK
from dsdag.core.op import opvertex2 as opvertex
from uuid import uuid4
import dsdag


Collect = OpK.from_callable(lambda *args, **kwargs: list(args),
                                 callable_name='collect',
                                 input_arguments=[])

@opvertex
class InputOp:
    obj = parameter(help_msg="The object to wrap and return")

    def run(self):
        return self.obj

@opvertex
class VarOp:
    obj = parameter(help_msg="The object to wrap and return")

    def _node_color(self):
        return '#b70043'

    def _node_shape(self):
        return 'box'

    def requires(self):
        return dict()

    def run(self):
        return self.obj


@opvertex
class VarOp2:
    obj = parameter(help_msg="The object to wrap and return")

    def __hash__(self):
        try:
            h = hash(self.obj)
        except TypeError:
            self.hash_id = getattr(self, 'hash_id', hash(str(uuid4())))
            h = self.hash_id
        return h

    def _node_color(self):
        return '#b70043'

    def _node_shape(self):
        return 'box'

    def requires(self):
        return dict()

    def run(self):
        return self.obj

@opvertex
class LambdaOp:
    f = parameter(help_msg="Function that is applied to the input")

    def _node_color(self):
        return '#d65768'

    def requires(self):
        raise NotImplementedError("Incomplete LambdaOp - must be applied")

    def run(self, *args, **kwargs):
        return self.f(*args, **kwargs)

@opvertex
class Select:
    i = parameter(0)
    _never_cache = True
    def run(self, l):
        return l[self.i]