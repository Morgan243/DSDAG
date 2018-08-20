import uuid
import pandas as pd

class _UnsetParameter(object):
    pass

class BaseParameter(object):
    _reserved_kw = ('name',)
    def __init__(self, default=_UnsetParameter,
                 help_msg=None):
        self.is_unset = True
        self.help_msg = help_msg
        self.default_value = default
        self.uid = str(uuid.uuid4())
        self.set_value(default)

    def __eq__(self, other):
        if isinstance(other, BaseParameter) or issubclass(type(other), BaseParameter):
            return self.value == other.value
        else:
            return self.value == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(BaseParameter), self.value))

    def set_value(self, v):
        self.is_unset = isinstance(v, _UnsetParameter)
        if not self.is_unset:
            self.value = self.parse_value(v)
        else:
            self.value = v

    def parse_value(self, v):
        return v

    def value_repr(self):
        return str(self.value)

    def __repr__(self):
        try:
            type_name = type(self.value).__name__
        except:
            type_name = str(type(self.value))

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
    def parse_value(self, v):
        return pd.to_datetime(v)

class SQLQueryParameter(BaseParameter):
    def value_repr(v):
        return "'%s ...'" % str(v)[:10].strip()

import interactive_data_tree as idt
class RepoTreeParameter(BaseParameter):
    def parse_value(self, v):
        return idt.RepoTree(v)

class DataFrameParameter(BaseParameter):
    pass
