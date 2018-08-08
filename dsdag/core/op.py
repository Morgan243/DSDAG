import abc
from toposort import toposort
import copy
import interactive_data_tree as idt
import time
import logging
import pandas as pd

from parameter import BaseParameter, UnhashableParameter

class OpMeta(type):
    existing = dict()
    def __new__(cls, name, parents, dct):
        _cls =  super(OpMeta, cls).__new__(cls, name, parents, dct)

        if _cls not in cls.existing:
            docs = name
            docs += '\n\n'
            docs += 'Parameters\n'
            docs += '----------\n'
            docs += "\n".join("%s : %s" % (k, str(param.help_msg))
                              for k, param in
                              _cls.scan_for_op_parameters(overrides=dict()).items())
            docs += '\n\n\n'

            _cls.__doc__ = docs
            cls.existing[_cls] = _cls
        else:
            _cls = cls.existing[_cls]

        cnt = sum(1 if c.__name__ == _cls.__name__ else 0 for c in cls.existing.keys())
        if cnt > 1:
            _cls.unique_name = "%s%d" % (str(_cls.__name__), cnt - 1)
        else:
            _cls.unique_name = str(_cls.__name__)

        return _cls

class OpVertex(object):
    __metaclass__ = OpMeta
    _never_cache = False

    # Map types + parameters to instances - don't duplicate
    existing_ops = dict()
    def __init__(self, **kwargs):
        # For parameters:
        #   (1) An attribute stores the full (Base)Parameter Instance
        #       - store under parameters
        #   (2) The value of each parameter is assigned to self

        # Filled out during build call
        self._built = False
        self._requirements = dict()
        self._dag = None
        self._user_kwargs = kwargs
        self._parameters = self.scan_for_op_parameters(overrides=self._user_kwargs)
        for p_n, p, in self._parameters.items():
            setattr(self, p_n, p.value)
        self.__update_doc_str()

        self._cacheable = not self._never_cache

    def __update_doc_str(self):
        docs = self.__class__.__name__
        docs += '\n'

        docs += "\n".join("%s : %s" % (k, str(param.help_msg))
                          for k, param in self._parameters.items())
        self.__doc__ = docs

    def __call__(self, *args, **kwargs):
        return self.with_requires(*args, **kwargs)

    def __repr__(self):
        params = self.get_parameters()
        repr = ", ".join("%s=%s" % (str(k), str(params[k]))
                        for k in sorted(params.keys()))
        return self.__class__.__name__ + "(" + repr + ")"

    def __hash__(self):
        #return hash((self.requirements, self.__parameters))
        p_tuples = tuple([(k, repr(v)) for k, v in self._parameters.items()])
        r_tuples = tuple([(k, repr(v)) for k, v in self._requirements.items()])
        return hash((type(self), p_tuples, r_tuples))

    def req_match(self, other):
        for req_name, req_o in self._requirements.items():
            for other_req_name, other_req_o in other.requirements.items():
                if (req_name != other_req_name) or (req_o != other_req_o):
                    return False
        return True

    def __eq__(self, other):
        if not isinstance(other, OpVertex) and not issubclass(type(other), OpVertex):
            return False

        self_params = self.get_parameters()
        other_params = other.get_parameters()

        if len(self_params) != len(other_params):
            return False

        for p_k, p_v in self_params.items():
            if p_k not in other_params:
                return False

            if not(p_v == other_params[p_k]):
                return False

        if not self.req_match(other):
            return False

        if type(self) != type(other):
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def _set_dag(self, dag):
        from dsdag.core.dag import DAG
        if not isinstance(dag, DAG) or not issubclass(type(dag), DAG):
            msg = "Expected a DAG object or derived, got %s" % str(type(dag))
            raise ValueError(msg)
        self._dag = dag

    def get_logger(self):
        return self._dag.logger

    def set_cacheable(self, is_cacheable):
        self._cacheable = is_cacheable

    def get_input_ops(self):
        return self._dag.get_op_input(self)

    @classmethod
    def passthrough_params(cls, dest_cls, cls_name, on_conflict='error', skip_params=None,
                           requirement_name=None):
        params = cls.scan_for_op_parameters(overrides=dict())
        attrs_params = dict()
        for p_name, p in params.items():
            if skip_params is not None and p_name in skip_params:
                continue

            if hasattr(dest_cls, p_name):
                if on_conflict == 'use_src':
                    #setattr(new_cls, p_name, p)
                    attrs_params[p_name] = p
                elif on_conflict == 'use_dest':
                    pass
                else:
                    raise ValueError("Param with name %s already exists on %s" % (p_name, str(dest_cls)))
            else:
                #setattr(new_cls, p_name, p)
                attrs_params[p_name] = p

        if requirement_name is not None:
            def req_closure(self):
                return {requirement_name: cls(**{
                    _pname: getattr(self, _pname)
                    for _pname, _p in attrs_params.items()
                })}
            attrs_params['requires'] = req_closure

        new_cls = OpMeta(cls_name, (dest_cls,),
                         attrs_params)

        return new_cls

    @classmethod
    def passthrough_requirements(cls, skip_params=None, on_conflict='error', **pass_reqs):
        """"
        A Pass through requirement inherits it's parent's parameters and constructs a requires method that
        instatiate those requirements and passes the param values through (i.e. passthrough)
        """
        # Mix inputs (mixin)?
        #params = cls.scan_for_op_parameters(overrides=dict())
        #attrs_params = dict()
        dst_cls = cls
        for r_name, r_op in pass_reqs.items():
            dst_cls = r_op.passthrough_params(dst_cls, "Passthrough" + cls.__name__,
                                              on_conflict=on_conflict, skip_params=skip_params)


    @classmethod
    def scan_for_op_parameters(cls, overrides):
        """ Scans the class for BaseParameter types and returns mapping to them"""
        params = dict()
        for o_n in dir(cls):
            o = getattr(cls, o_n)
            if isinstance(o, BaseParameter) or issubclass(type(o), BaseParameter):
                params[o_n] = copy.copy(o)

                if o_n in overrides:
                    params[o_n].set_value(overrides[o_n])
        return params

    def with_requires(self, *args, **kwargs):
        if len(kwargs) == 0 and len(args) == 0:
            return self
        elif len(kwargs) != 0:
            req_ret = kwargs
        elif len(args) != 0:
            from dsdag.ext.misc import VarOp
            # not supports *args just yet
            req_ret = list(args)
            req_ret = [a if isinstance(a, OpVertex) or issubclass(a.__class__, OpVertex)
                       else VarOp(obj=a)
                       for a in req_ret]

        else:
            msg = "Mix of *args and **kwargs not supported (yet?)"
            raise ValueError(msg)

        def closure(self):
            return req_ret
        # Define new Op class that has correct requires
        new_class = OpMeta(self.unique_name, (type(self),),
                           {'requires':closure})

        return new_class(**self._user_kwargs)

    @classmethod
    def with_params(cls, op_name=None, **kwargs):
        for k, v in kwargs.items():
            assert (isinstance(v, BaseParameter)
                    or issubclass(type(v), BaseParameter))
        name = (cls.__name__ + "WithParams") if op_name is None else op_name
        new_c = type(name,
                     (cls,),
                     kwargs)
        return new_c

    def get_parameters(self):
        return self._parameters

    def requires(self):
        return dict()

    def run(self):
        raise NotImplementedError("Implement run")

    def build(self, **dag_kwargs):
        from dsdag.core.dag import DAG
        return DAG(self, **dag_kwargs)

    def op_nb_viz(self, op_out, viz_out=None):
        raise NotImplementedError()
        from IPython.display import display
        import ipywidgets as widgets
        if viz_out is None:
            viz_out = widgets.Output()

        viz_out.append_display_data("op_nb_viz not implemented!")
        return viz_out





##############
if __name__ == """__main__""":
    pass
