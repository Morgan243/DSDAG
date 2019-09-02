import abc
from toposort import toposort
import copy
import uuid
import types
import time
import logging
import inspect
import pandas as pd
import types
from uuid import uuid4
import attr

from dsdag.core.parameter import parameter
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


default_viz_props = dict(node_color='lightblue2',
                           node_style='filled',
                           node_shape='oval')
@attr.s(cmp=False)
class OpK(object):
    run_callable = attr.ib()
    requires_callable = attr.ib()
    name = attr.ib('NONAME')

    dag = attr.ib(None)
    never_cache = attr.ib(False)
    cacheable = attr.ib(True)
    ops_to_unpack = attr.ib(attr.Factory(set))
    unpack_output = attr.ib(False)
    #downstream = attr.ib(dict())
    downstream = attr.ib(attr.Factory(dict))
    node_viz_kws = attr.ib(attr.Factory(lambda : copy.deepcopy(default_viz_props)))

    closure_map = dict()

    def __attrs_post_init__(obj):

        fields = attr.fields(obj.__class__)
        obj.parameters_ = {f.name: getattr(obj, f.name) for f in fields}
        # Filter out OpK
        #obj.parameters_ = {k: o for k, o in obj.parameters_.items()
        #                   if not isinstance(o, OpK) }
        obj.param_hashable = tuple([(k, v if isinstance(v, collections.Hashable) else str(uuid4()))
                                    for k, v in obj.parameters_.items()])

        obj.set_requires(obj.requires_callable)

        obj.cacheable = not obj.never_cache
        # only partially implemented in past versions, stub out for now
        obj.runtime_parameters = dict()

        return obj

    @staticmethod
    def is_op(obj):
        return hasattr(obj, 'opk') and isinstance(obj.opk, OpK)

    @staticmethod
    def op_nb_viz(op_out, viz_out=None):
        import ipywidgets as widgets
        if viz_out is None:
            viz_out = widgets.Output()

        viz_out.append_display_data("op_nb_viz not implemented!")
        return viz_out

    def get_requires(self, parent):
        return getattr(parent, self.requires_callable_name, self.requires_callable)

    def get_parameters(self):
        return self.parameters_

    def is_unpack_required(self, op):
        return op in self.ops_to_unpack

    def set_downstream(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.downstream[k] = v

        return self

    def apply(self, *args, **kwargs):
        from dsdag.ext.misc import VarOp2 as VarOp
        if len(kwargs) == 0 and len(args) == 0:
            return self
        elif len(kwargs) > 0 and len(args) > 0:
            msg = "Mix of *args and **kwargs not supported (yet?)"
            raise ValueError(msg)
        if len(kwargs) != 0:
            # var and input are basically the same?
            auto_op = lambda _obj: VarOp(obj=_obj) #if not isinstance(_obj, collections.Hashable) else InputOp(obj=_obj)
            req_ret = {k:(v if OpK.is_op(v) else auto_op(v))
                        for k, v in kwargs.items()}
            req_hash = hash(tuple(k, v) for k, v in req_ret.items())
        if len(args) != 0:
            # var and input are basically the same?
            auto_op = lambda _obj: VarOp(obj=_obj) #if not isinstance(_obj, collections.Hashable) else InputOp(obj=_obj)
            req_ret = list(args)
            req_ret = [a if OpK.is_op(a) else auto_op(a)
                       for a in req_ret]
            req_hash = hash(tuple(req_ret))

        for r in req_ret:
            if hasattr(r, 'downstream'):
                self.set_downstream(**r.downstream)

        if req_hash not in self.__class__.closure_map:
            #def closure(*args, **kwargs):
            #    return req_ret
            #opk.__class__.closure_map[req_hash] = closure
            self.__class__.closure_map[req_hash] = lambda *args, **kwargs: req_ret

        # Use descriptor protocol: https://docs.python.org/2/howto/descriptor.html
        #self.requires = closure.__get__(self)
        #self.req_hash = req_hash
        #opk._req_hashable = req_hash
        #opk.requires = opk.__class__.closure_map[opk._req_hashable].__get__(opk)
        self.set_requires(self.__class__.closure_map[req_hash],
        #opk.__class__.closure_map[req_hash].__get__(opk),
                         hash_of_requires=req_hash,
                         name_of_method='_auto_requires')

        return self

    def set_requires(self, requires_callable, hash_of_requires=None,
                     name_of_method='requires'):
        self.requires_callable = requires_callable
        self.requires_callable_name = name_of_method if name_of_method is not None else None

        if self.requires_callable is not None:
            if self.requires_callable_name is None:
                self.requires_callable_name = self.requires_callable.__name__

            if hash_of_requires is None:
                try:
                    self.req_hashable = inspect.getsource(self.requires_callable)
                except:
                    print("Cannot get source of requires: %s" % str(self.requires_callable))
                    self.req_hashable = uuid4()
            else:
                self.req_hashable = hash_of_requires
        elif hash_of_requires is None:
            self.req_hashable = ('requires_nothing',)
        elif hash_of_requires is not None:
            self.req_hashable = hash_of_requires

    def __hash__(self):
        self._op_hash = hash((type(self),
                              #self.param_hashable,
                              self.req_hashable))
        return self._op_hash

    def _set_dag(self, dag):
        from dsdag.core.dag import DAG
        if not isinstance(dag, DAG) or not issubclass(type(dag), DAG):
            msg = "Expected a DAG object or derived, got %s" % str(type(dag))
            raise ValueError(msg)
        self.dag = dag

    #def get_name(self):
        #return self._name if self._name is not None else self.unique_cls_name
        #return self._name if self._name is not None else self.__class__.__name__

    @staticmethod
    def from_callable(callable,
                      input_arguments=[0],
                      callable_name=None):
        """
        Generate a class from a callable that can be used as an Op in a DAG.

        :param callable:
        :param input_arguments:
        :param callable_name:
        :return:
        """
        input_arguments = list() if input_arguments is None else input_arguments

        ###
        # NAME
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
                _attrs[p_name] = parameter(default=p.default, init=True,
                                  kw_only=p.kind == p.KEYWORD_ONLY)

        #_op = attr.make_class(callable_name, _attrs,
        #                      bases=(OpParent,), #(OpParent,),
        #                      cmp=False)
        #_attrs['_name'] = parameter(callable_name, init=True, kw_only=True)

        _op = type(callable_name, (object,), _attrs)

        if sig is not None:
            #kw_inputs = [ia for ia in input_arguments if ia not in pos_inputs]
            def _run(s, *args, **kwargs):
                #pos_inputs = [ia for ia in (input_arguments) if isinstance(ia, int)]
                pos_inputs = range(len(list(args)))
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

        else:
            def _run(s, *_args, **_kwargs): return callable(*_args, **_kwargs)

        _op.run = _run
        #_op.run = types.MethodType(_run, _op)
        #_op.opk = OpK(_run, None)
        # make_class already initializes
        _op = opvertex2(_op, name=False)

        return _op


class OpParent(object):
    def __call__(self, *args, **kwargs):
        self.opk = self.opk.apply(*args, **kwargs)
        return self

    def __repr__(self):
        params = self.opk.get_parameters()
        repr = ", ".join("%s=%s" % (str(k), str(params[k]))
                        for k in sorted(params.keys()))
        if self._name is not None:
            repr += ', name=\'' + self._name + '\''
        return self.__class__.__name__ + "(" + repr + ")"

    def __hash__(self):
        fields = attr.fields(self.__class__)
        parameters_ = {f.name: getattr(self, f.name) for f in fields}
        # Filter out OpK
        parameters_ = {k: o for k, o in parameters_.items()
                           if not isinstance(o, OpK) }
        param_hashable = tuple([(k, v if isinstance(v, collections.Hashable) else str(uuid4()))
                                    for k, v in parameters_.items()])

        return hash((self.opk, param_hashable))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def _set_dag(self, dag):
        self.opk._set_dag(dag)

    def get_name(self):

        return getattr(self, '_name', self.opk.name)

    def get_logger(self, log_level='WARN'):
        if self.opk.dag is not None:
            l = self.opk.dag.get_op_logger(self)
        else:
            l = logging.getLogger()
            l.setLevel(log_level)
        return l

    def set_unpack_input(self, op):
        self.opk.ops_to_unpack.add(op)
        return op

    def unset_unpack_input(self, op):
        if op in self.opk.ops_to_unpack:
            self.opk.ops_to_unpack.remove(op)
        return op

    def requires(self):
        return dict()

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


def opvertex2(cls, run_method='run', requires_method='requires',
              unpack_run_return=False, ops_to_unpack=None,
              name=True, node_viz_kws=None, extra_attrs=None):
    ops_to_unpack = set() if ops_to_unpack is None else ops_to_unpack
    node_viz_kws = default_viz_props if node_viz_kws is None else node_viz_kws
    extra_attrs = dict() if extra_attrs is None else extra_attrs

    if isinstance(name, bool) and name:
        cls._name = parameter(cls.__name__, init=True, kw_only=True)

    opk_f = lambda self: OpK(getattr(self, run_method),
                             getattr(self, requires_method, None),
                             unpack_output=unpack_run_return,
                             ops_to_unpack=ops_to_unpack,
                             node_viz_kws=node_viz_kws,
                             name=cls.__name__)

    cls.opk = parameter(default=attr.Factory(opk_f, takes_self=True),
                        init=True, kw_only=True)
    attr_cls = attr.s(cls,
                      cmp=False, these=None)

    if not issubclass(cls, OpParent):
        attr_cls = attr.make_class(cls.__name__,
                                   attrs=[],# attrs=attr_cls.__attrs_attrs__
                                   #bases=(OpParent, attr_cls),
                                   bases=(attr_cls, OpParent),
                                   cmp=False,
                                   )


    return attr_cls


