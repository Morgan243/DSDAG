from .context import dsdag
import unittest

OpVertex = dsdag.core.op.OpVertexAttr
opvertex = dsdag.core.op.opvertex
opattr = dsdag.core.op.opattr

#BaseParameter = dsdag.core.parameter.BaseParameter
#UnhashableParameter = dsdag.core.parameter.UnhashableParameter
df_templates = dsdag.ext.dataframe
DAG = dsdag.core.dag.DAG
LambdaOp = dsdag.ext.misc.LambdaOp
VarOp = dsdag.ext.misc.VarOp

@opvertex
class Bar(OpVertex):
    def requires(self):
        return dict()

    def run(self):
        return "Bar"

@opvertex
class Foo(OpVertex):
    def requires(self):
        return dict(bar=Bar())

    def run(self, bar):
        return "Foo" + bar

@opvertex
class ProvideInt(OpVertex):
    magic_num = opattr(default=42)

    def run(self):
        return self.magic_num

@opvertex
class AddOp(OpVertex):
    magic_num = opattr(default=42)

    def requires(self):
        return dict(x=ProvideInt(magic_num=self.magic_num),
                    y=ProvideInt(magic_num=self.magic_num))

    def run(self, x, y):
        self.ratio = (x + y)/self.magic_num

        return x+y


from pprint import pprint
class TestAttrsDAG(unittest.TestCase):
    def test_build(self):
        ret = DAG([AddOp(magic_num=11)])
        print(ret)
        print("=======")
        pprint(ret.dep_map)
        print("=======")
        pprint(ret.topology)

        t = ret()
        print(t)

        ret = DAG([Foo()])
        print(ret)
        print("=======")
        pprint(ret.dep_map)
        print("=======")
        pprint(ret.topology)

        t = ret()
        print("--OUTPUT--")
        print(t)

    def test_lambda_callable(self):
        collect = OpVertex.from_callable(lambda *args, **kwargs: list(args),
                                         callable_name='collect',
                                         input_arguments=[])
        c_0 = collect()(1, 2, 3)
        c_1 = collect()(2, 3, 4)

        self.assertTrue(c_0 != c_1)

        t_d = DAG([c_0, c_1])
        # No dedup for these input ops
        self.assertEqual(len(t_d.all_ops), 8)


