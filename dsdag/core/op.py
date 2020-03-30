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


default_viz_props = dict(color='lightblue2',
                         style='filled',
                         shape='oval')
@attr.s(cmp=False)
class OpK(object):
    run_callable = attr.ib()
    requires_callable = attr.ib()
    parameters = attr.ib(None)
    parent_cls = attr.ib(None)

    name = attr.ib('NONAME')

    dag = attr.ib(None)
    never_cache = attr.ib(False)
    cacheable = attr.ib(True)
    ops_to_unpack = attr.ib(attr.Factory(set))
    unpack_output = attr.ib(False)
    #downstream = attr.ib(dict())
    downstream = attr.ib(attr.Factory(dict))
    node_viz_kws = attr.ib(attr.Factory(lambda: copy.deepcopy(default_viz_props)))

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
    def auto_op(obj):
        from dsdag.ext.misc import VarOp2 as VarOp
        return obj if OpK.is_op(obj) else VarOp(obj=obj)
        #auto_op = lambda _obj: VarOp(obj=_obj)  # if not isinstance(_obj, collections.Hashable) else InputOp(obj=_obj)

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
        if len(kwargs) == 0 and len(args) == 0:
            return self
        elif len(kwargs) > 0 and len(args) > 0:
            msg = "Mix of *args and **kwargs not supported (yet?)"
            raise ValueError(msg)
        if len(kwargs) != 0:
            req_ret = {k: OpK.auto_op(v) for k, v in kwargs.items()}
            req_hash = hash(tuple(k, v) for k, v in req_ret.items())
        if len(args) != 0:
            req_ret = [OpK.auto_op(a) for a in list(args)]
            req_hash = hash(tuple(req_ret))

        for r in req_ret:
            if hasattr(r, 'opk') and isinstance(r.opk, OpK):
                self.set_downstream(**r.opk.downstream)

        req_f = lambda *args, **kwargs: req_ret

        # Use descriptor protocol: https://docs.python.org/2/howto/descriptor.html
        self.set_requires(req_f,
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
        _op = opvertex(_op, name=True)

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
                           if not isinstance(o, OpK)}

        param_hashable = list()
        uuids = getattr(self.opk, 'uuids', dict())

        import hashlib
        for pname, p in parameters_.items():
            if isinstance(p, (pd.Series, pd.DataFrame)):
                h = hashlib.sha256(pd.util.hash_pandas_object(p, index=True).values).hexdigest()
            elif isinstance(p, collections.Hashable):
                h = hash(p)
            else:
                h = uuids.get(pname, uuid4())
                uuids[pname] = h

            param_hashable.append((pname, h))


        param_hashable = tuple(param_hashable)
        self.opk.uuids = uuids

        return hash((self.opk, param_hashable))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def _set_dag(self, dag):
        self.opk._set_dag(dag)

    def __rshift__(self, shift):
        if isinstance(shift, OpParent):
            return shift(self)
        elif hasattr(shift, 'opk'):
            return shift.opk.apply(self)
        elif isinstance(shift, (tuple, list)):
            if not all(isinstance(s, OpParent) for s in shift):
                raise TypeError("All elements in a tuple/list must be derived from OpParent")
            return OpCollection([self >> s for s in shift])
        elif callable(shift):
            from dsdag.ext.misc import LambdaOp
            return self >> LambdaOp(f=shift, name=shift.__name__)
        else:
            raise ValueError("Unknown shift apply for %s" % str(type(shift)))

    def get_name(self):
        #return getattr(self, '_name', self.opk.name)
        return self.opk.name

    def set_name(self, name):
        self.opk.name = name

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

    def set_downstream(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.opk.downstream[k] = v

        return self

    def map(self, ops, suffix='map'):
        #return [copy(self)(o) for o in ops]
        return [self.new(name='%s__%s%d' % (self.get_name(),
                                            suffix,  i))(o)
                for i, o in enumerate(ops)]

    def new(self, name=None):
        from copy import copy, deepcopy
        c = deepcopy(self)
        c.set_name(name=name if name is not None else self.get_name())
        return c

    def unwind(self):
        deps_to_resolve = [self]
        while len(deps_to_resolve):
            dep_op = deps_to_resolve[0]
            deps_to_resolve = deps_to_resolve[1:]
            yield dep_op

            reqs = dep_op.opk.get_requires(dep_op)()
            reqs = list(reqs.values() if isinstance(reqs, dict) else reqs)
            deps_to_resolve += reqs

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


def opvertex(cls=None, run_method='run', requires_method='requires',
             unpack_run_return=False, ops_to_unpack=None,
             name=True,
             node_color='lightblue', node_style='filled', node_shape='oval',
             #node_viz_kws=None,
                          extra_attrs=None,
             auto_subclass=True):
    if cls is None:
        def _ov(cls=None, run_method=run_method, requires_method=requires_method,
                unpack_run_return=unpack_run_return, ops_to_unpack=ops_to_unpack,
                name=name, node_color=node_color, node_style=node_style, node_shape=node_shape,
                extra_attrs=extra_attrs, auto_subclass=auto_subclass):
            return opvertex(cls=cls, run_method=run_method, requires_method=requires_method,
                            unpack_run_return=unpack_run_return, ops_to_unpack=ops_to_unpack,
                            name=name, node_color=node_color, node_style=node_style, node_shape=node_shape,
                            extra_attrs=extra_attrs, auto_subclass=auto_subclass)
        return _ov

    ops_to_unpack = set() if ops_to_unpack is None else ops_to_unpack
    #node_viz_kws = default_viz_props if node_viz_kws is None else node_viz_kws
    node_viz_kws = dict(color=node_color, style=node_style, shape=node_shape)
    extra_attrs = dict() if extra_attrs is None else extra_attrs

    if isinstance(name, bool) and name:
        cls.name = parameter(cls.__name__, init=True, kw_only=True)

    opk_f = lambda self: OpK(getattr(self, run_method),
                             getattr(self, requires_method, None),
                             unpack_output=unpack_run_return,
                             ops_to_unpack=ops_to_unpack,
                             node_viz_kws=node_viz_kws,
                             parent_cls=cls,
                             parameters=attr.fields(cls),
                             # If the OpParent has a name Set, take that
                             name=getattr(self, 'name', cls.__name__))

    cls.opk = parameter(default=attr.Factory(opk_f, takes_self=True),
                        init=True, kw_only=True)
    attr_cls = attr.s(cls,
                      cmp=False,
                      these=None)

    if not issubclass(cls, OpParent) and auto_subclass:
        attr_cls = attr.make_class(cls.__name__,
                                   attrs=[],# attrs=attr_cls.__attrs_attrs__
                                   #bases=(OpParent, attr_cls),
                                   bases=(attr_cls, OpParent),
                                   cmp=False,
                                   )

    return attr_cls


@attr.s
class OpCollection:
    ops = attr.ib()

    #def apply(self):
    def __call__(self, *args, **kwargs):
        self.ops = [o.opk.apply(*args, **kwargs) for o in self.ops]
        return self


    def __rshift__(self, shift):
        """
        Many to one
        :param self:
        :param shift:
        :return:
        """
        if isinstance(shift, OpParent):
            return shift(*self.ops)
            #return OpCollection([shift.new()(o) for o in self.ops])

    def __lshift__(self, shift):
        """
        Many to many
        :param self:
        :param shift:
        :return:
        """
        if isinstance(shift, OpParent):
            return OpCollection([shift.new()(o) for o in self.ops])

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
        return DAG(required_outputs=self.ops,
                   read_from_cache=read_from_cache,
                   write_to_cache=write_to_cache,
                   cache=cache,
                   cache_eviction=cache_eviction,
                   force_downstream_rerun=force_downstream_rerun,
                   pbar=pbar,
                   live_browse=live_browse,
                   logger=logger)


