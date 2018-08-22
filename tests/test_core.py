from .context import dsdag
import unittest
from pprint import pprint

OpVertex = dsdag.core.op.OpVertex
BaseParameter = dsdag.core.parameter.BaseParameter
UnhashableParameter = dsdag.core.parameter.UnhashableParameter
df_templates = dsdag.ext.pandas
DAG = dsdag.core.dag.DAG
LambdaOp = dsdag.ext.misc.LambdaOp
VarOp = dsdag.ext.misc.VarOp

class Bar(OpVertex):
    def requires(self):
        return dict()

    def run(self):
        return "Bar"

class Foo(OpVertex):
    def requires(self):
        return dict(bar=Bar())

    def run(self, bar):
        return "Foo" + bar

class ProvideInt(OpVertex):
    magic_num = BaseParameter(default=42)

    def run(self):
        return self.magic_num

class AddOp(OpVertex):
    magic_num = BaseParameter(default=42)

    def requires(self):
        return dict(x=ProvideInt(magic_num=self.magic_num),
                    y=ProvideInt(magic_num=self.magic_num))

    def run(self, x, y):
        self.ratio = (x + y)/self.magic_num

        return x+y


class TestDSDAGBuild(unittest.TestCase):
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

    def test_op_naming(self):
        add_op11 = AddOp(magic_num=11, name='Magic_11_AddOpp')
        add_op3 = AddOp(magic_num=3, name='Magic_3_AddOpp')

        add_op = AddOp(name='Agg')(x=add_op11, y=add_op3)

        dag = add_op.build()
        res = dag()
        print(res)

    def test_with_requires(self):
        ## Basic Correctness
        for mx, my in [(1, 2), (5, 9), (10, 35)]:
            new_t = AddOp()(x=ProvideInt(magic_num=mx),
                            y=ProvideInt(magic_num=my))

            d = DAG([new_t()])
            t = d()
            self.assertEqual(t, mx+my)

        ## Args version
        for mx, my in [(1, 2), (5, 9), (10, 35)]:
            new_t = AddOp()(ProvideInt(magic_num=mx),
                            ProvideInt(magic_num=my))

            d = DAG([new_t()])
            t = d()
            self.assertEqual(t, mx+my)

        ## Auto Op Wrap Version
        for mx, my in [(1, 2), (5, 9), (10, 35)]:
            new_t = AddOp()(mx,
                            my)

            d = DAG([new_t()])
            t = d()
            self.assertEqual(t, mx+my)

    def test_with_params(self):
        for mn in range(3, 25, 2):
            new_t = AddOp.with_params(magic_num=BaseParameter(mn))
            d = DAG([new_t()])
            t = d()
            self.assertEqual(t, mn*2)

    def test_duplicate_operations(self):
        class T1(OpVertex):
            def requires(self):
                return dict(x=ProvideInt(magic_num=5), y=ProvideInt(magic_num=5))

            def run(self, x, y):
                return x * y

        class T2(OpVertex):
            def requires(self):
                return dict(a=ProvideInt(magic_num=6), b=ProvideInt(magic_num=5))

            def run(self, a, b):
                return a + b

        t1 = T1()
        t2 = T2()

        d = DAG([t1, t2])
        # T1, T2, int 5, int 6
        self.assertEqual(len(d.all_ops), 4)
        r = d()
        self.assertFalse(t1 == t2)
        self.assertEqual(r, [25, 11])

        print(r)

    def test_deduplicating_operations(self):
        var = VarOp(obj=5, name='int_5')
        f = lambda x: x + 1
        o1 = LambdaOp(f=f, name='main_lambda')(var)
        o2 = LambdaOp(f=f, name='main_lambda')(var)
        o3 = LambdaOp(f=f, name='main_lambda')(var)

        o4 = LambdaOp(f=sum)(o1, o2, o3)
        dag = o4.build()
        self.assertEqual(o1, o2, msg='Identical lambda ops not considered the same!')
        self.assertEqual(o2, o3, msg='Identical lambda ops not considered the same!')

        key_names = [o.unique_cls_name for o in dag.dep_map.keys()]

        detail_dep_is_in_map = {o.unique_cls_name: {d.unique_cls_name:d.unique_cls_name in key_names for d in deps}
                         for o, deps in dag.dep_map.items()}

        dep_is_in_map = {_name: all(dep_dict.values())
                    for _name, dep_dict in detail_dep_is_in_map.items()}

        self.assertTrue(all(dep_is_in_map.values()))

    def test_duplicate_op_name(self):
        pass

    def test_dataframe_template(self):
        import pandas as pd
        df_a = pd.DataFrame(dict(key=range(10),
                                 a=range(20, 30)))
        df_b = pd.DataFrame(dict(key=range(10),
                                 b=range(33, 43)))

        class ProvideDF(OpVertex):
            df = UnhashableParameter()
            def run(self):
                return self.df

        a = ProvideDF(df=df_a)
        b = ProvideDF(df=df_b)
        op = df_templates.Merge(key='key')(a, b)
        dag = op.build()
        merge_df = dag()
        print(merge_df)

    def test_dag_op_getter(self):
        op = AddOp(magic_num=11, name='important_op')
        dag = op.build()

        self.assertEqual(op, dag['important_op'])

    def test_caching(self):
        pass

    def test_getting_input_ops(self):
        class InputUseAddOp(OpVertex):
            def run(self, x):
                inputs = self.get_input_ops()
                x_op = inputs['x']

                return x_op.ratio

        #t_ratio = InputUseAddOp().build()()
        t_ratio = InputUseAddOp()(x=AddOp()).build()()
        self.assertEqual(t_ratio, 2)


class TestParameterTypes(unittest.TestCase):
    def test_datetime_param(self):
        pass