import dsdag.ext.data_provider
import dsdag.ext.misc
import dsdag.ext.dataframe
import dsdag.ext.sql
import dsdag.ext.modeling
import dsdag.ext.plotting
import dsdag.ext.op_doc
import dsdag.ext.distributed


try:
    import pandas
except ImportError as e:
    pandas = None

def from_module(mod, post_call_func=None, input_arguments=None, eager_call=False, name_suffix='', verbose=False):
    _ops = dict()
    for dname, d in ((dname, getattr(mod, dname)) for dname in dir(mod)):
        ia = input_arguments.get(dname) if isinstance(input_arguments, dict) else input_arguments
        try:
            _ops[dname] = dsdag.core.op.Opk.from_callable(d,
                                                           post_call_func=post_call_func,
                                                           callable_name=dname+name_suffix,
                                                           input_arguments=ia,
                                                           eager_call=eager_call)
        except:
            if verbose:
                print("Could not Opify %s" % dname)
            pass
    return type('pd_ops', (object,), _ops )

if pandas is not None:
    _pd_ops = dict()
    for dname, d in ((dname, getattr(pandas, dname)) for dname in dir(pandas)):
        if callable(d):
            #print(dname)
            _pd_ops[dname] = dsdag.core.op.OpK.from_callable(d)
    pd = type('pd_ops', (object,), _pd_ops )