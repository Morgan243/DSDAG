from dsdag.core.op import OpVertex
from dsdag.core.parameter import BaseParameter, UnhashableParameter

class VarOp(OpVertex):
    obj = UnhashableParameter(help_msg="The object to wrap and return")

    def requires(self):
        return dict()

    def run(self):
        return self.obj


class LambdaOp(OpVertex):
    f = BaseParameter(help_msg="Function that is applied to the input")

    def requires(self):
        raise NotImplementedError("Incomplete LambdaOp - must be applied")

    def run(self, o):
        return self.f(o)