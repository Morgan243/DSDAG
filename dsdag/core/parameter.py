import uuid
import pandas as pd

class _UnsetParameter(object):
    pass

class BaseParameter(object):
    _reserved_kw = ('name',)
    def __init__(self, default=_UnsetParameter(),
                 help_msg=None):
        self.is_unset = True
        self.help_msg = help_msg
        self.default_value = default
        self.uid = str(uuid.uuid4())
        #self.set_value(default)
        self.value = default
        self._cached_decoded = None

    def __eq__(self, other):
        if isinstance(other, BaseParameter) or issubclass(type(other), BaseParameter):
            return self._value == other._value
        else:
            return self._value == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(BaseParameter), self._value))

    def set_value(self, v):
        self.is_unset = isinstance(v, _UnsetParameter)
        if not self.is_unset:
            self._value = self.encode_value(v)
        else:
            self._value = v

    def get_value(self):
        self.is_unset = isinstance(self._value, _UnsetParameter)
        if not self.is_unset:
            if self._cached_decoded is None:
                self._cached_decoded = self.decode_value(self._value)
            return self._cached_decoded
        else:
            return self._value

    value = property(fget=get_value, fset=set_value, fdel=None, doc=None)

    def encode_value(self, v):
        return  v

    def decode_value(self, enc_v):
        return enc_v

    def value_repr(self):
        return str(self._value)

    def __repr__(self):
        try:
            type_name = type(self.value).__name__
        except:
            type_name = str(type(self._value))

        return "(%s)%s" % (type_name, self.value_repr())

class UnhashableParameter(BaseParameter):
    unhashable_value_id = 0
    def __init__(self, *args, **kwargs):
        super(UnhashableParameter, self).__init__(*args, **kwargs)

    def set_value(self, v):
        super(UnhashableParameter, self).set_value(v)
        self.uid += "-" + str(UnhashableParameter.unhashable_value_id)
        UnhashableParameter.unhashable_value_id += 1

    def __hash__(self):
        return self.uid

    def __eq__(self, other):
        if isinstance(other, UnhashableParameter) or issubclass(type(other), UnhashableParameter):
            return self.uid == other.uid
        else:
            return False

    def __repr__(self):
        try:
            type_name = type(self.value).__name__
        except:
            type_name = str(type(self.value))

        return "(%s)%s" % (type_name, self.uid)

    def value_repr(self):
        return hash(self)

class DatetimeParameter(BaseParameter):
    def encode_value(self, v):
        return pd.to_datetime(v)

class SQLQueryParameter(BaseParameter):
    def value_repr(v):
        return "'%s ...'" % str(v)[:10].strip()

class RepoTreeParameter(BaseParameter):

    def __hash__(self):
        return hash((type(BaseParameter), self._value))

    def encode_value(self, v):
        import interactive_data_tree as idt
        if isinstance(v, idt.RepoLeaf):
            parent_rt_path = v.parent_repo.idr_prop['repo_root']
            leaf_name = v.name
            return (parent_rt_path, leaf_name)
        elif isinstance(v, idt.RepoTree):
            parent_rt_path = v.idr_prop['repo_root']
            return parent_rt_path
        else:
            raise ValueError("%s expects %s, got %s"
                             % (self.__class__.__name__,
                                idt.RepoLeaf.__name__,
                                str(type(v))))


    def decode_value(self, enc_v):
        import interactive_data_tree as idt
        #if len(self._value) == 2:
        if isinstance(self._value, tuple):
            parent_rt_path, leaf_name = self._value
            return idt.RepoTree(parent_rt_path)[leaf_name]
        else:
            parent_rt_path = self._value
            return idt.RepoTree(parent_rt_path)




class DataFrameParameter(BaseParameter):
    pass


def ParamCopy(op, *params, **overrides):
    if len(params) == 0:
        params = op.scan_for_op_paramters(overrides)
    else:
        params = {p:getattr(op, p)
                        for p in params}
    return type('ParamCopyMixin', (object,), params)

def ParamMixin_old(*ops, **kwops):

    params = [o.scan_for_op_parameters(overrides=None) for o in ops]
    params += [o.scan_for_op_parameters(overrides=None) for i in kwops.values()]
    params = {p_n:p for o in params for p_n, p in o.items()}

    return  type('ParamMixin', (object,), params)
