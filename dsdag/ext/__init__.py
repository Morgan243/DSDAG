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

if pandas is not None:
    _pd_ops = dict()
    for dname, d in ((dname, getattr(pandas, dname)) for dname in dir(pandas)):
        if callable(d):
            #print(dname)
            _pd_ops[dname] = dsdag.core.op.OpVertexAttr.from_callable(d)
    pd = type('pd_ops', (object,), _pd_ops )