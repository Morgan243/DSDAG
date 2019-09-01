from .context import dsdag
import unittest
from mock import patch

#from dsdag.core.op import OpVertexAttr, parameter, opvertex
OpK = dsdag.core.op.OpK
#OpVertex = dsdag.core.op.OpVertexAttr

#OpVertex = dsdag.core.op.OpVertexAttr
opvertex = dsdag.core.op.opvertex2
opattr = dsdag.core.op.parameter

#BaseParameter = dsdag.core.parameter.BaseParameter
#UnhashableParameter = dsdag.core.parameter.UnhashableParameter
df_templates = dsdag.ext.dataframe
DAG = dsdag.core.dag.DAG
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
        logger = self.get_logger()
        logger.info("Adding %d and %d" % (x, y))
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

        completions = ret._ipython_key_completions_()
        self.assertTrue('Foo' in completions)

    def test_dag_op_getter(self):
        op = AddOp(magic_num=11, name='important_op')
        dag = op.build()
        self.assertEqual(op, dag['important_op'])

    def test_basic_op(self):
        op = AddOp(100)(5, 5)
        dag = DAG([op])
        res = dag()
        self.assertEqual(res, 10)

    def test_getting_input_ops(self):
        @opvertex
        class InputUseAddOp:
            def run(self, x):
                #inputs = self.get_input_ops()
                inputs = self.opk.dag.get_op_input(self)
                x_op = inputs['x']

                return x_op.ratio

        #t_ratio = InputUseAddOp().build()()
        t_ratio = InputUseAddOp()(x=AddOp()).build()()
        self.assertEqual(t_ratio, 2)

    def test_lambda_callable(self):
        collect = OpK.from_callable(lambda *args, **kwargs: list(args),
                                         callable_name='collect',
                                         input_arguments=[])
        c_0 = collect()
        c_0 = c_0(1, 2, 3)

        c_1 = collect()
        c_1 = c_1(2, 3, 4)

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

    def test_duplicate_operations(self):
        @opvertex
        class T1:
            def requires(self):
                return dict(x=ProvideInt(magic_num=5), y=ProvideInt(magic_num=5))

            def run(self, x, y):
                return x * y

        @opvertex
        class T2:
            def requires(self):
                return dict(a=ProvideInt(magic_num=6), b=ProvideInt(magic_num=5))

            def run(self, a, b):
                return a + b

        t1 = T1()
        t2 = T2()

        d = DAG([t1, t2])
        # Only need 4 ops: T1, T2, int 5, int 6
        self.assertEqual(len(d.all_ops), 4)
        r = d()
        self.assertFalse(t1 == t2)
        self.assertEqual(r, [25, 11])

        print(r)


        var = VarOp(obj=5, name='int_5')
        f = lambda x: x + 1
        o1 = LambdaOp(f=f, name='main_lambda')(var)
        o2 = LambdaOp(f=f, name='main_lambda')(var)
        o3 = LambdaOp(f=f, name='main_lambda')(var)

        o4 = LambdaOp(f=sum)(o1, o2, o3)
        dag = o4.build()
        self.assertEqual(o1, o2, msg='Identical lambda ops not considered the same!')
        self.assertEqual(o2, o3, msg='Identical lambda ops not considered the same!')

        #key_names = [o.unique_cls_name for o in dag.dep_map.keys()]
        key_names = [o.get_name() for o in dag.dep_map.keys()]

        #detail_dep_is_in_map = {o.unique_cls_name: {d.unique_cls_name:d.unique_cls_name in key_names for d in deps}
        #                 for o, deps in dag.dep_map.items()}
        detail_dep_is_in_map = {o.get_name(): {d.get_name():d.get_name() in key_names for d in deps}
                                for o, deps in dag.dep_map.items()}

        dep_is_in_map = {_name: all(dep_dict.values())
                    for _name, dep_dict in detail_dep_is_in_map.items()}

        self.assertTrue(all(dep_is_in_map.values()))

    def test_dag_op_getter(self):
        op = AddOp(magic_num=11, name='important_op')
        dag = op.build()

        self.assertEqual(op, dag['important_op'])

    def test_caching(self):
        @opvertex
        class Canary:
            canary = 0

        @opvertex
        class CacheTestOpA:
            def run(self, x):
                Canary.canary +=1
                return [x*.5]

        @opvertex
        class CacheTestOpB:
            def run(self, x):
                Canary.canary +=1
                return x*5

        op_a = CacheTestOpA()(10)
        op_b = CacheTestOpB()(op_a)
        #lambda_op = LambdaOp(f=lambda a, b: a+b)(op_b, op_b)
        LambdaAdd = OpK.from_callable(lambda a, b: a + b,
                                      input_arguments=[0, 1],
                                      callable_name='lambda_add')
        lambda_op = LambdaAdd()
        lambda_op = lambda_op(op_b, op_a)

        dag = lambda_op.build(write_to_cache=True,
                             read_from_cache=True,
                             logger='DEBUG')
        # First run
        res = dag()
        self.assertEqual(Canary.canary, 2)
        Canary.canary = 0

        res = dag()
        self.assertEqual(Canary.canary, 0)

        self.assertTrue(isinstance(dag.cache, dict))
        dag.clear_cache()
        self.assertEqual(len(dag.cache), 0)

        ## Cache eviction
        LambdaSum = OpK.from_callable(lambda s: sum(s, 0),
                                      input_arguments=[0],
                                      callable_name='lambda_sum')
        lambda_sum = LambdaSum()(op_b)
        dag = lambda_sum.build(write_to_cache=True, read_from_cache=True,
                                          cache_eviction=True, logger='DEBUG')
        # First run
        res = dag()
        self.assertEqual(Canary.canary, 2)
        Canary.canary = 0

        res = dag()
        self.assertEqual(Canary.canary, 0)

    def test_clear_cache_per_dag(self):
        @opvertex
        class Canary:
            canary = 0

        @opvertex
        class CacheTestOpA:
            def run(self, x):
                Canary.canary +=1
                return [x*.5]

        @opvertex
        class CacheTestOpB:
            def run(self, x):
                Canary.canary +=1
                return x*5


        op_a = CacheTestOpA()(10)
        op_b = CacheTestOpB()(op_a)
        a_dag = DAG([op_a], write_to_cache=True)
        b_dag = DAG([op_b], write_to_cache=True)

        a_res = a_dag()
        b_res = b_dag()

        self.assertTrue(len(DAG._CACHE), 2)
        a_dag.clear_cache(only_ops_in_dag=True)
        self.assertTrue(len(DAG._CACHE), 1)

    def test_single_op_requires(self):
        @opvertex
        class SingleRequires:
            def requires(self):
                return ProvideInt(magic_num=10)

            def run(self, i):
                return i*10

        dag = SingleRequires().build()
        res = dag()

        self.assertEqual(res, 100)

    def test_op_parameter_ordering(self):
        @opvertex
        class OpOrderTest:
            first = opattr(1)
            second = opattr(2)
            third = opattr(3)

            def run(self):
                return self.first*1 + self.second/2 + self.third/3

        dag = OpOrderTest(1, 2, 3).build()
        res = dag()

        self.assertEqual(res, 3)

    def test_arg_unpacking(self):
        @opvertex
        class ListReturner:
            def run(self):
                return list(range(5))

        @opvertex
        class NeedsArgs:
            def requires(self):
                op = ListReturner()
                self.set_unpack_input(op)
                self.unset_unpack_input(op)
                self.set_unpack_input(op)
                return op

            def run(self, a1, a2, a3, a4, a5):
                return a1 + a2 + a3 + a4 + a5

        dag = DAG(NeedsArgs())
        res = dag()
        self.assertEqual(res, 10)

    def test_kwarg_unpacking(self):
        @opvertex
        class DictReturner:
            def run(self):
                return {k:v for k, v in zip('abc', [1, 2, 3])}

        @opvertex
        class NeedsKwargs:
            def requires(self):
                op = DictReturner()
                self.set_unpack_input(op)
                return op

            def run(self, a, b, c):
                return a + b + c

        dag = DAG(NeedsKwargs())
        res = dag()
        self.assertEqual(res, 6)

    def test_lazy_op(self):
        # Foo Bar may have been used before, so give custom name to id
        op = Foo()(Bar(name='bar_test'))
        dag = DAG(['bar_test', op])
        bar, foo = dag()
        self.assertEqual(bar, "Bar")
        self.assertEqual(foo, "FooBar")


    @patch("matplotlib.pyplot.show")
    def test_basic_viz(self, mock_show):
        mock_show.return_value = None
        op_a = AddOp()
        op_b = AddOp()

        dag = DAG([op_a, op_b])

        dag.viz()
        res = dag()
        dag.plot()

    def test_timings(self):
        op_a = AddOp()
        op_b = AddOp()

        dag = DAG([op_a, op_b])

        res = dag()

        times = dag.timings()
        self.assertEqual(len(times), 2)


    def test_browse(self):
        op_a = AddOp()
        op_b = AddOp()

        dag = DAG([op_a, op_b])

        res = dag()

        dag.browse()

    def test_op_repr(self):
        op_a = AddOp()
        print(repr(op_a))

    def test_no_dag_logger(self):

        op_a = AddOp()
        logger = op_a.get_logger()
