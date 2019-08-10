from .context import dsdag
import unittest

from dsdag.core.op import OpVertexAttr, opattr, opvertex
OpK = dsdag.core.op.OpK
OpVertex = dsdag.core.op.OpVertexAttr

#OpVertex = dsdag.core.op.OpVertexAttr
opvertex = dsdag.core.op.opvertex2
opattr = dsdag.core.op.opattr

#BaseParameter = dsdag.core.parameter.BaseParameter
#UnhashableParameter = dsdag.core.parameter.UnhashableParameter
df_templates = dsdag.ext.dataframe
DAG = dsdag.core.dag.DAG2
LambdaOp = dsdag.ext.misc.LambdaOp
VarOp = dsdag.ext.misc.VarOp

@opvertex
class Bar(object):
    def requires(self):
        return dict()

    def run(self):
        return "Bar"

@opvertex
class Foo(object):
    def requires(self):
        return dict(bar=Bar())

    def run(self, bar):
        return "Foo" + bar

@opvertex
class ProvideInt(object):
    magic_num = opattr(default=42)

    def run(self):
        return self.magic_num

@opvertex
class AddOp(object):
    magic_num = opattr(default=42)

    def requires(self):
        return dict(x=ProvideInt(magic_num=self.magic_num),
                    y=ProvideInt(magic_num=self.magic_num))

    def run(self, x, y):
        self.ratio = (x + y)/self.magic_num

        return x+y

from pprint import pprint
import tempfile
import shutil
class TestAttrsDAG(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

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

    def test_basic_op(self):
        op = AddOp(100)(5, 5)
        dag = DAG([op])
        res = dag()
        self.assertEqual(res, 10)

    def test_lambda_callable(self):
        collect = OpK.from_callable(lambda *args, **kwargs: list(args),
                                         callable_name='collect',
                                         input_arguments=[])
        c_0 = collect()(1, 2, 3)
        c_1 = collect()(2, 3, 4)

        self.assertTrue(c_0 != c_1)

        t_d = DAG([c_0, c_1])
        # Op Detects that literal input is hashable, so makes it an input Op
        # automatically allowing it to be dedupped - 6 ops rather than 8
        self.assertEqual(len(t_d.all_ops), 6)

    def test_op_naming(self):
        add_op11 = AddOp(magic_num=11, name='Magic_11_AddOpp')
        add_op3 = AddOp(magic_num=3, name='Magic_3_AddOpp')
        add_op4 = AddOp(magic_num=5, name='Magic_3_AddOpp')

        add_op = AddOp(name='Agg')(x=add_op11, y=add_op3)

        dag = add_op.build()
        res = dag()
        print(res)
        self.assertEqual(add_op.get_name(), 'Agg')
        self.assertEqual(add_op3.get_name(), 'Magic_3_AddOpp')
        # Op 4 is not in a dag, so it doesn't have a uncollided name (provided by dag)
        self.assertEqual(add_op4.get_name(), 'Magic_3_AddOpp')

        dag1 = DAG([add_op3, add_op4])
        self.assertEqual(dag1.get_dag_unique_op_name(add_op3), 'Magic_3_AddOpp')
        # Now name will be different
        self.assertEqual(dag1.get_dag_unique_op_name(add_op4), 'Magic_3_AddOpp_1')

        # Should fail
        with self.assertRaises(KeyError):
            dag.get_dag_unique_op_name(add_op4)
