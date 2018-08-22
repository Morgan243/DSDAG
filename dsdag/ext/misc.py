from dsdag.core.op import OpVertex
from dsdag.core.parameter import BaseParameter, UnhashableParameter

class InputOp(OpVertex):
    obj = BaseParameter(help_msg="The object to wrap and return")

    def run(self):
        return self.obj

class VarOp(OpVertex):
    obj = UnhashableParameter(help_msg="The object to wrap and return")

    def _node_color(self):
        return '#b70043'

    def _node_shape(self):
        return 'box'

    def requires(self):
        return dict()

    def run(self):
        return self.obj


class LambdaOp(OpVertex):
    f = BaseParameter(help_msg="Function that is applied to the input")

    def _node_color(self):
        return '#d65768'

    def requires(self):
        raise NotImplementedError("Incomplete LambdaOp - must be applied")

    def run(self, *args, **kwargs):
        return self.f(*args, **kwargs)