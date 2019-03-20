import abc
from toposort import toposort
import copy
import uuid
import types
import interactive_data_tree as idt
import time
import logging
import pandas as pd

from dsdag.core.parameter import BaseParameter, UnhashableParameter
import collections

def make_param_doc_str(param_name, param_help, wrap_len=80,
                       optional=False,
                       none_msg='<no help message provided>'):
    base_str = (param_name + ' : '
                + ('(OPTIONAL)' if optional else '')
                + (param_help if param_help is not None else none_msg))

    pad_len = len(param_name) + 3 # +2 is for space plus ':'
    step_size = wrap_len - pad_len

    lines = [base_str[:wrap_len]]
    lines += [(' ' * pad_len) + base_str[i: i + step_size ].strip()
              for i in range(wrap_len, len(base_str), step_size)]


    return "\n".join(lines)

class OpMeta(type):
    """Updates doc strings on creation of new op by scanning op parameters"""

    # Prepare is only supported in python 3+ :(
    @classmethod
    def __prepare__(cls, name, bases):
        return collections.OrderedDict()

    def __new__(cls, name, parents, dct):
        # not actually ordered in python 2.*
        dct['__ordered_params__'] = [k for k, v in dct.items()
                                    if isinstance(v, BaseParameter) or issubclass(type(v), BaseParameter)]
        parent_params = {p: getattr(p, '__ordered_params__', list())
                            for p in parents }

        parent_params.update({_k: _v for k, v in parent_params.items()
                                for _k, _v in getattr(k, '__parents_params__', dict()).items()
                                if len(_v) > 0})
        dct['__parents_params__'] = parent_params

        _cls =  super(OpMeta, cls).__new__(cls, name, parents, dct)
        original_docs = _cls.__doc__
        docs = _cls.__doc__.strip() if _cls.__doc__ is not None else name
        docs += '\n\n'
        docs += 'Parameters\n'
        docs += '----------\n'
        docs += "\n".join(make_param_doc_str(k,
                                             getattr(_cls, k).help_msg,
                                             optional=getattr(_cls, k).optional)
                                for k in dct['__ordered_params__'])

        if any(len(_p_names) > 0 for _p_names in _cls.__parents_params__.values()):
            #docs += '\n\n\n------------------\n'
            docs += '\n\n\n[Inherited Parameters]'
        for parent_cls, param_names in _cls.__parents_params__.items():
            if len(param_names) == 0:
                continue
            heading = '\n\n%s' % parent_cls.__name__
            heading += '\n' + ('-'*len(heading)) + '\n'
            docs += heading
            docs += "\n".join(make_param_doc_str(k,
                                                 getattr(parent_cls, k).help_msg,
                                                 optional=getattr(parent_cls, k).optional)
                              for k in param_names)
        docs += '\n\n\n'

        _cls.__doc__ = docs
        _cls.__original_doc__ = original_docs

        return _cls

    #def __call__(cls, *args, **kwargs):
        # Deduplication of ops should probably happen here?
        #o = type.__call__(cls, *args, **kwargs)
        #return o

class OpVertex(object):
    __metaclass__ = OpMeta
    _never_cache = False
    _instance_id_map = dict()
    _given_name_cnt_map = dict()
    _closure_map = dict()
    def __new__(cls, *args, **kwargs):
        obj = super(OpVertex, cls).__new__(cls)

        # TODO: align %args with ordered dict args

        # For parameters:
        #   (1) An attribute stores the full (Base)Parameter Instance
        #       - store under parameters
        #   (2) The value of each parameter is assigned to self

        # Filled out during build call
        obj._built = False
        obj.req_hash = None
        obj._unpack_ops = set()
        obj.unpack_output = False
        obj._downstream = dict()

        obj._dag = None
        obj._user_args = args
        if len(obj._user_args) > 0:
            raise ValueError("Ops do not currently support non-keyword constructor arguments")

        obj._user_kwargs = kwargs
        obj._parameters = obj.scan_for_op_parameters(overrides=obj._user_kwargs)
        obj._runtime_parameters = {k:v for k, v in obj._parameters.items() if v.runtime}
        obj._name = kwargs.get('name', None)
        #if obj._name is not None:
        #    name_iid = obj.__class__._given_name_cnt_map.get(obj._name, 0)
        #    new_name = obj._name
        #    if name_iid > 0:
        #        new_name = obj._name + '_' + str(name_iid)
        #    obj.__class__._given_name_cnt_map[obj._name] = name_iid + 1
        #    obj._name = new_name

        #obj.unique_cls_name = str(obj.__class__.__name__)
        #iid = obj.__class__._instance_id_map.get(obj.unique_cls_name)
        #if iid is not None:
        #    obj.__class__._instance_id_map[obj.unique_cls_name] += 1
        #    obj.unique_cls_name += '_%s' % str(iid)
        #else:
        #    obj.__class__._instance_id_map[obj.unique_cls_name] = 1

        for p_n, p, in obj._parameters.items():
            setattr(obj, p_n, p.value)

        obj._cacheable = not obj._never_cache

        return obj

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def __repr__(self):
        params = self.get_parameters()
        repr = ", ".join("%s=%s" % (str(k), str(params[k]))
                        for k in sorted(params.keys()))
        if self._name is not None:
            repr += ', name=\'' + self._name + '\''
        return self.__class__.__name__ + "(" + repr + ")"
        #return self._name + "(" + repr + ")"

    def __hash__(self):
        p_tuples = tuple([(k, repr(self._parameters[k]))
                          for k in sorted(self._parameters.keys())])
        #r_tuples = tuple([(k, repr(self._requirements[k]))
        #                  for k in sorted(self._requirements.keys())])
        if self.req_hash is not None:
            r = self.req_hash
        else:
            r = self.requires.__func__

        return hash((type(self), p_tuples, r))
        #return hash((self.unique_cls_name, p_tuples, r_tuples))

    def _req_match(self, other):
        return self.requires.__func__ == other.requires.__func__

    def _param_match(self, other):
        self_params = self.get_parameters()
        other_params = other.get_parameters()

        if len(self_params) != len(other_params):
            return False

        for p_k, p_v in self_params.items():
            if p_k not in other_params:
                return False

            if not(p_v == other_params[p_k]):
                return False

        return True

    def __hash__eq__(self, other):
        return hash(self) == hash(other)

    def __eq__(self, other):
        if not isinstance(other, OpVertex) and not issubclass(type(other), OpVertex):
            return False

        if not self._param_match(other):
            return False

        if not self._req_match(other):
            return False

        if not isinstance(self, type(other)):# and not issubclass(type(self), type(other)):
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

    def _node_color(self):
        return 'lightblue2'

    def _node_style(self):
        return 'filled'

    def _node_shape(self):
        return 'oval'

    def _get_viz_attrs(self):
        return dict(color=self._node_color(),
                    style=self._node_style(),
                    shape=self._node_shape())

    def set_downstream(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            self._downstream[k] = v

        return self

    def get_logger(self, log_level='WARN'):
        if self._dag is not None:
            l = self._dag.get_op_logger(self)
        else:
            l = logging.getLogger()
            l.setLevel(log_level)
        return l

    def set_unpack_input(self, op):
        self._unpack_ops.add(op)
        return op

    def unset_unpack_input(self, op):
        if op in self._unpack_ops:
            self._unpack_ops.remove(op)
        return op

    def is_unpack_required(self, op):
        return op in self._unpack_ops

    def set_name(self, name):
        self._name = name

    def get_name(self):
        #return self._name if self._name is not None else self.unique_cls_name
        return self._name if self._name is not None else self.__class__.__name__

    def set_cacheable(self, is_cacheable):
        self._cacheable = is_cacheable

    def get_input_ops(self):
        return self._dag.get_op_input(self)

    @classmethod
    def passthrough_params(cls, dest_cls, cls_name,
                           on_conflict='error', skip_params=None,
                           requirement_name=None):
        """



        :param dest_cls:
        :param cls_name:
        :param on_conflict:
        :param skip_params:
        :param requirement_name:
        :return:
        """
        params = cls.scan_for_op_parameters(overrides=dict())
        attrs_params = dict()
        for p_name, p in params.items():
            if skip_params is not None and p_name in skip_params:
                continue

            if hasattr(dest_cls, p_name):
                if on_conflict == 'use_src':
                    attrs_params[p_name] = p
                elif on_conflict == 'use_dest':
                    pass
                else:
                    raise ValueError("Param with name %s already exists on %s" % (p_name, str(dest_cls)))
            else:
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
    def passthrough_requirements(cls, skip_params=None,
                                 on_conflict='error', **pass_reqs):
        """"
        A Pass through requirement inherits it's parent's parameters and constructs a requires method that
        instatiate those requirements and passes the param values through (i.e. passthrough)
        """
        dst_cls = cls
        for r_name, r_op in pass_reqs.items():
            dst_cls = r_op.passthrough_params(dst_cls, "Passthrough" + cls.__name__,
                                              on_conflict=on_conflict, skip_params=skip_params)
        return dst_cls

    @classmethod
    def scan_for_op_parameters(cls, overrides=None):
        """ Scans the class for BaseParameter types and returns mapping to them"""
        params = dict()
        for o_n in dir(cls):
            o = getattr(cls, o_n)
            if isinstance(o, BaseParameter) or issubclass(type(o), BaseParameter):
                if o_n in BaseParameter._reserved_kw:
                    raise ValueError("Parameter cannot use reserved keyword '%s'" % o_n)
                params[o_n] = copy.copy(o)

                if overrides is not None and o_n in overrides:
                    params[o_n].set_value(overrides[o_n])
        return params

    @classmethod
    def copy_op_parameters(cls, other, default_overrides=None):
        params = cls.scan_for_op_parameters(overrides=default_overrides)
        for o_n, o in params.items():
            setattr(other, o_n, o)

    def apply(self, *args, **kwargs):
        if len(kwargs) == 0 and len(args) == 0:
            return self
        elif len(kwargs) != 0:
            from dsdag.ext.misc import VarOp
            req_ret = kwargs
            req_hash = hash(tuple((k, v if isinstance(v, OpVertex) or issubclass(v.__class__, OpVertex) else VarOp(obj=v))
                        for k, v in kwargs.items()))
        elif len(args) != 0:
            from dsdag.ext.misc import VarOp
            req_ret = list(args)
            req_ret = [a if isinstance(a, OpVertex) or issubclass(a.__class__, OpVertex)
                       else VarOp(obj=a)
                       for a in req_ret]
            req_hash = hash(tuple(req_ret))

        else:
            msg = "Mix of *args and **kwargs not supported (yet?)"
            raise ValueError(msg)

        for r in req_ret:
            if hasattr(r, '_downstream'):
                self.set_downstream(**r._downstream)

        if req_hash not in self.__class__._closure_map:
            def closure(self):
                return req_ret
            self.__class__._closure_map[req_hash] = closure

        # Use descriptor protocol: https://docs.python.org/2/howto/descriptor.html
        #self.requires = closure.__get__(self)
        self.req_hash = req_hash
        self.requires = self.__class__._closure_map[self.req_hash].__get__(self)

        return self

    @classmethod
    def with_params(cls, op_name=None, **kwargs):
        for k, v in kwargs.items():
            if not (isinstance(v, BaseParameter) or issubclass(type(v), BaseParameter)):
                raise ValueError("New parameters must all be derived from %s" % BaseParameter.__name__)
        name = (cls.__name__ + "WithParams") if op_name is None else op_name
        new_c = OpMeta(name, (cls,), kwargs)
        return new_c

    def get_parameters(self):
        return self._parameters

    def requires(self):
        return dict()

    def run(self):
        raise NotImplementedError("Implement run")

    def build(self,
              read_from_cache=False,
              write_to_cache=False,
              cache=None,
              cache_eviction=False,
              force_downstream_rerun=True,
              pbar=True,
              live_browse=False,
              logger=None):

        from dsdag.core.dag import DAG
        return DAG(required_outputs=self,
                   read_from_cache=read_from_cache,
                   write_to_cache=write_to_cache,
                   cache=cache,
                   cache_eviction=cache_eviction,
                   force_downstream_rerun=force_downstream_rerun,
                   pbar=pbar,
                   live_browse=live_browse,
                   logger=logger)

    def op_nb_viz(self, op_out, viz_out=None):
        raise NotImplementedError()
        from IPython.display import display
        import ipywidgets as widgets
        if viz_out is None:
            viz_out = widgets.Output()

        viz_out.append_display_data("op_nb_viz not implemented!")
        return viz_out

import attr

def opattr(default=attr.NOTHING, validator=None,
           #repr=True, cmp=True, hash=None,
           init=True,
           #convert=None,
           metadata=None, type=None,
           converter=None, factory=None,
           kw_only=False, help_msg=None):
    return attr.ib(default=default, validator=validator, repr=True, cmp=True, hash=None,
                   init=init, convert=None, metadata=metadata, type=type, converter=converter,
                   factory=factory, kw_only=kw_only)

def opvertex(cls):
    cls._name = opattr(cls.__name__, init=True, kw_only=True)
    return attr.s(cls, cmp=False, these=None)

#@attr.s(cmp=False)#frozen=True)#hash=True)
class _OpVertexAttr(object):
    _runtime_parameters = dict()#attr.ib(factory=dict, init=False)
    _never_cache = False #attr.ib(False, init=False)
    _closure_map = dict()
    def __attrs_post_init__(self):
        obj = self

        # TODO: align %args with ordered dict args

        # For parameters:
        #   (1) An attribute stores the full (Base)Parameter Instance
        #       - store under parameters
        #   (2) The value of each parameter is assigned to self

        # Filled out during build call
        #obj._built = False
        #obj.req_hash = None
        obj._unpack_ops = set()
        obj.unpack_output = False
        obj._downstream = dict()

        obj._dag = None
        fields = attr.fields(self.__class__)
        self.parameters_ = {f.name: getattr(self, f.name) for f in fields}
        import collections
        from uuid import uuid4
        self._param_hashable = tuple([(k, v if isinstance(v, collections.Hashable) else str(uuid4()))
                                for k, v in self.get_parameters().items()])
        self._req_hashable = self.requires.__func__ if isinstance(self.requires.__func__, collections.Hashable) else uuid4()
        #self._op_hash = hash((type(self), self._param_hashable, self._req_hashable))

        #obj._user_args = args
        #if len(obj._user_args) > 0:
        #    raise ValueError("Ops do not currently support non-keyword constructor arguments")

        #obj._user_kwargs = kwargs
        #obj._parameters = obj.scan_for_op_parameters(overrides=obj._user_kwargs)
        #obj._runtime_parameters = {k:v for k, v in obj._parameters.items() if v.runtime}
        #obj._name = kwargs.get('name', None)

        #for p_n, p, in obj._parameters.items():
        #    setattr(obj, p_n, p.value)

        obj._cacheable = not obj._never_cache

        return obj

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def __repr__(self):
        params = self.get_parameters()
        repr = ", ".join("%s=%s" % (str(k), str(params[k]))
                        for k in sorted(params.keys()))
        if self._name is not None:
            repr += ', name=\'' + self._name + '\''
        return self.__class__.__name__ + "(" + repr + ")"
        #return self._name + "(" + repr + ")"

    @classmethod
    def from_callable(cls, callable, post_call_func=None, input_arguments=[0], eager_call=False,
                      callable_name=None, never_cache=False):
        import inspect
        input_arguments = list() if input_arguments is None else input_arguments
        if callable_name is not None:
            pass
        elif hasattr(callable, '__name__'):
            callable_name = callable.__name__
        elif hasattr(callable, '__class__'):
            callable_name = callable.__class__.__name__
        else:
            callable_name = str(callable)

        try:
            sig = inspect.signature(callable)
        except ValueError as e:
            #print("Cannot obtain signature of %s" % callable_name)
            sig = None

        _attrs = dict()
        if sig is not None:
            for p_name, p in sig.parameters.items():
                #print("%s=%s (kw_only=%s" % (p_name, str(p.default),
                #                             str(p.kind == p.KEYWORD_ONLY)))
                _attr = attr.ib(default=p.default, init=True,
                                kw_only=p.kind == p.KEYWORD_ONLY)
                _attrs[p_name] = _attr

        _attr = attr.ib(default=callable_name, init=True,
                        kw_only=True)
        if 'name' in _attrs:
            op_name_key = '_%s_name' % callable_name
        else:
            op_name_key = '_name'
        _attrs[op_name_key] = _attr

        tmp_cls = type(callable_name,
                       (_OpVertexAttr,), _attrs)
        #tmp_cls._name = callable_name
        _op = tmp_cls

        attr.s(_op, cmp=False, these=None)
        if sig is not None and post_call_func is None:
            pos_inputs = [ia for ia in input_arguments if isinstance(ia, int)]
            kw_inputs = [ia for ia in input_arguments if ia not in pos_inputs]
            def _run(s, *args, **kwargs):
                _kwargs = {k: kwargs.get(k, getattr(s, k))
                           for i, (k, v) in enumerate(sig.parameters.items())
                            if i not in pos_inputs}
                if len(args) != len(pos_inputs):
                    print("Appears to be missing positional arguments")
                    print("Expected %d args" % len(pos_inputs))
                    print("But Got %d args" % len(args))
                #_kwargs.update({k:kwargs[k] for k in kw_inputs })
                #_args = [a for i, a in enumerate(args)]

                return callable(*args, **_kwargs)
        elif post_call_func is not None:
            pos_inputs = [ia for ia in input_arguments if isinstance(ia, int)]
            kw_inputs = [ia for ia in input_arguments if ia not in pos_inputs]

            if eager_call:
                def _run(s, *args, **kwargs):
                    _kwargs = {k: getattr(s, k)
                               for i, (k, v) in enumerate(sig.parameters.items())
                                if i not in pos_inputs}
                    #if len(args) != len(pos_inputs):
                    #    print("Appears to be missing positional arguments")
                    #    print("Expected %d args" % len(pos_inputs))
                    #    print("But Got %d args" % len(args))
                    #_kwargs.update({k:kwargs[k] for k in kw_inputs })
                    _args = [getattr(s, k) for i, k in enumerate(sig.parameters.keys()) if i in pos_inputs]

                    s.call_results = getattr(s, 'call_results', callable_name(*_args, **_kwargs))
                    return post_call_func(s, s.call_results , *args, **kwargs)
            else:
                def _run(s, *args, **kwargs):
                    _kwargs = {k: getattr(s, k)
                               for i, (k, v) in enumerate(sig.parameters.items())
                                if i not in pos_inputs}
                    #if len(args) != len(pos_inputs):
                    #    print("Appears to be missing positional arguments")
                    #    print("Expected %d args" % len(pos_inputs))
                    #    print("But Got %d args" % len(args))
                    #_kwargs.update({k:kwargs[k] for k in kw_inputs })
                    _args = [getattr(s, k) for i, k in enumerate(sig.parameters.keys()) if i in pos_inputs]

                    return post_call_func(s, (callable, _args, _kwargs), *args, **kwargs)
        else:
            def _run(s, *_args, **_kwargs): return callable(*_args, **_kwargs)
        setattr(_op, 'run', _run)
        return _op

    def old__hash__(self):
        p_tuples = tuple([(k, repr(self._parameters[k]))
                          for k in sorted(self._parameters.keys())])
        if self.req_hash is not None:
            r = self.req_hash
        else:
            r = self.requires.__func__

        return hash((type(self), p_tuples, r))

    def __hash__(self):
        self._op_hash = hash((type(self), self._param_hashable, self._req_hashable))
        return self._op_hash

    def _req_match(self, other):
        return self.requires.__func__ == other.requires.__func__

    def _param_match(self, other):
        self_params = self.get_parameters()
        other_params = other.get_parameters()

        if len(self_params) != len(other_params):
            return False

        for p_k, p_v in self_params.items():
            if p_k not in other_params:
                return False

            if not(p_v == other_params[p_k]):
                return False

        return True

    def _set_dag(self, dag):
        from dsdag.core.dag import DAG
        if not isinstance(dag, DAG) or not issubclass(type(dag), DAG):
            msg = "Expected a DAG object or derived, got %s" % str(type(dag))
            raise ValueError(msg)
        self._dag = dag

    def _node_color(self):
        return 'lightblue2'

    def _node_style(self):
        return 'filled'

    def _node_shape(self):
        return 'oval'

    def _get_viz_attrs(self):
        return dict(color=self._node_color(),
                    style=self._node_style(),
                    shape=self._node_shape())

    def set_downstream(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            self._downstream[k] = v

        return self

    def get_logger(self, log_level='WARN'):
        if self._dag is not None:
            l = self._dag.get_op_logger(self)
        else:
            l = logging.getLogger()
            l.setLevel(log_level)
        return l

    def set_unpack_input(self, op):
        self._unpack_ops.add(op)
        return op

    def unset_unpack_input(self, op):
        if op in self._unpack_ops:
            self._unpack_ops.remove(op)
        return op

    def is_unpack_required(self, op):
        return op in self._unpack_ops

    def set_name(self, name):
        self._name = name

    def get_name(self):
        #return self._name if self._name is not None else self.unique_cls_name
        return self._name if self._name is not None else self.__class__.__name__

    def set_cacheable(self, is_cacheable):
        self._cacheable = is_cacheable

    def get_input_ops(self):
        return self._dag.get_op_input(self)

    def map(self, ops, suffix='map'):
        #return [copy(self)(o) for o in ops]
        return [self.new(name='%s__%s%d' %(self.get_name(), suffix,  i))(o)
                for i, o in enumerate(ops)]

    def new(self, name=None):
        from copy import copy, deepcopy
        c = deepcopy(self)
        c.set_name(name=name)
        return c


    def apply(self, *args, **kwargs):
        cls = (OpVertexAttr, _OpVertexAttr)
        if len(kwargs) == 0 and len(args) == 0:
            return self
        elif len(kwargs) != 0:
            from dsdag.ext.misc import VarOp, InputOp
            # var and input are basically the same?
            auto_op = lambda _obj: VarOp(obj=_obj) if not isinstance(_obj, collections.Hashable) else InputOp(obj=_obj)
            req_ret = kwargs
            req_hash = hash(tuple((k, v if isinstance(v, cls) or issubclass(v.__class__, cls) else auto_op(v))
                        for k, v in kwargs.items()))
        elif len(args) != 0:
            from dsdag.ext.misc import VarOp, InputOp
            # var and input are basically the same?
            auto_op = lambda _obj: VarOp(obj=_obj) if not isinstance(_obj, collections.Hashable) else InputOp(obj=_obj)
            req_ret = list(args)
            req_ret = [a if isinstance(a, cls) or issubclass(a.__class__, cls)
                       else auto_op(a)
                       for a in req_ret]
            req_hash = hash(tuple(req_ret))

        else:
            msg = "Mix of *args and **kwargs not supported (yet?)"
            raise ValueError(msg)

        for r in req_ret:
            if hasattr(r, '_downstream'):
                self.set_downstream(**r._downstream)

        if req_hash not in self.__class__._closure_map:
            def closure(self):
                return req_ret
            self.__class__._closure_map[req_hash] = closure

        # Use descriptor protocol: https://docs.python.org/2/howto/descriptor.html
        #self.requires = closure.__get__(self)
        #self.req_hash = req_hash
        self._req_hashable = req_hash
        self.requires = self.__class__._closure_map[self._req_hashable].__get__(self)

        return self

    def get_parameters(self):
        return self.parameters_

    def requires(self):
        return dict()

    def run(self):
        raise NotImplementedError("Implement run")

    def build(self,
              read_from_cache=False,
              write_to_cache=False,
              cache=None,
              cache_eviction=False,
              force_downstream_rerun=True,
              pbar=True,
              live_browse=False,
              logger=None):

        from dsdag.core.dag import DAG
        return DAG(required_outputs=self,
                   read_from_cache=read_from_cache,
                   write_to_cache=write_to_cache,
                   cache=cache,
                   cache_eviction=cache_eviction,
                   force_downstream_rerun=force_downstream_rerun,
                   pbar=pbar,
                   live_browse=live_browse,
                   logger=logger)


@opvertex
class OpVertexAttr(_OpVertexAttr):
    pass

class UpackingOp(OpVertex):
    pass

##############
if __name__ == """__main__""":
    @attr.s
    class Foo(OpVertexAttr):
        x = attr.ib(5)
        y = attr.ib(10)

    foo = Foo(x=66)