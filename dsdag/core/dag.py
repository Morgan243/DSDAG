import logging
import interactive_data_tree as idt
import time
from toposort import toposort
import pandas as pd #TODO: this dep should not be in core dag
from dsdag.core.op import OpVertex
import inspect
from collections import Counter

class DAG(object):
    _CACHE=dict()
    _default_log_level = 'INFO'
    def __init__(self, required_outputs,
                 read_from_cache=False,
                 write_to_cache=False,
                 cache=None,
                 cache_eviction=False,
                 force_downstream_rerun=True,
                 pbar=True,
                 live_browse=False,
                 logger=None):
        """
        Build a DAG that produces outputs from a set of Ops. After construction, calling the DAG will
        commence processing and return the required outputs from the relevant Ops.

        :param required_outputs: (OpVertex) Ops whose output is to be returned/materialized
        :param read_from_cache: (bool) True if Op outputs should be read from the provided cache
        :param write_to_cache: (bool) True if Op outputs should be written to the provided cache
        :param cache: (RepoTree) Caching location
        :param cache_eviction: (bool) If True, only the Op outputs required for the dag at each step are kept in the cache
        :param force_downstream_rerun: (bool) True if all dependent outputs should be rerun after an Op is run
        :param pbar: (bool) Use tqdm progress bar
        :param live_browse: (bool) Experimental
        :param logger: (logger or string level) Pass logger level (e.g. 'INFO', 'WARN') to set default level or pass
                        custom logger
        """
        #if isinstance(logger, basestring):
        if isinstance(logger, str):
            log_level = logger
            logger = None
        else:
            log_level = DAG._default_log_level

        if logger is None:
            logger = self._create_dag_logger('DAG', log_level)

        self.logger = logger
        self.log_level = log_level
        self.op_loggers = dict()

        ####
        self.live_browse = live_browse

        self.read_from_cache = read_from_cache
        self.write_to_cache = write_to_cache
        self.using_cache = (self.read_from_cache or self.write_to_cache)
        self.force_downstream_rerun = force_downstream_rerun

        self.cache_eviction = cache_eviction
        if self.using_cache and (cache is None):
            logger.info("Using dict cache")
            self.cache = DAG._CACHE
        elif cache is not None:
            self.cache = cache
        else:
            logger.info("Using dict cache")
            self.cache = DAG._CACHE


        if not isinstance(required_outputs, (list, tuple)):
            self.required_outputs = [required_outputs]
        elif isinstance(required_outputs, tuple):
            self.required_outputs = list(required_outputs)
        elif isinstance(required_outputs, list):
            self.required_outputs = required_outputs
        else:
            msg = "Expected required outputs to be a single object, list, or tuple.\n"
            msg += "Got %s" % str(type(required_outputs))
            raise ValueError(msg)

        self.output_ordering_map = {ro: i for i, ro in enumerate(self.required_outputs)}
        self.lazy_outputs = [ro for ro in self.required_outputs if isinstance(ro, str)]
        if len(self.lazy_outputs) > 0:
            self.logger.info("Lazy required outputs found: %s"
                             % ", ".join(self.lazy_outputs))
            self.required_outputs = list(set(self.required_outputs) - set(self.lazy_outputs))


        self.pbar = pbar

        self.op_name_counts = dict()# Counter(o.get_name() for o in self.all_ops.values())
        self.op_suffixes = dict()
        self.all_ops, self.dep_map, self.input_op_map = self.build(self.required_outputs)


        self.runtime_parameters = dict()
        for op in self.all_ops:
            for p_name, param in op._runtime_parameters.items():
                self.runtime_parameters[p_name] = param

        self.name_to_op_map = {o.get_name():o for o in self.all_ops.values()}
        for lo in self.lazy_outputs:
            if lo not in self.name_to_op_map:
                self.logger.error("Lazy Op %s is not in resulting build Ops: %s"
                                    % (lo, "\n" + "\n".join(self.name_to_op_map.keys())))
                raise ValueError("Lazy Op %s could not be resolved after build" % lo)

            self.logger.info("Adding %s to outputs" % lo)
            self.required_outputs.append(self.name_to_op_map[lo])
            self.output_ordering_map[self.name_to_op_map[lo]] = self.output_ordering_map[lo]
            del self.output_ordering_map[lo]
            self.required_outputs = list(sorted(self.required_outputs,
                                                key=self.output_ordering_map.get))

        self.dep_sets = {p:set(d.values()) if isinstance(d, dict) else set(d)
                         for p, d in self.dep_map.items()}
        self.topology = list(toposort(self.dep_sets))

        self.outputs = dict()
        self.dag_start_time = None
        self.system_utilization = dict()
        self.start_and_finish_times = dict()
        self.all_requirements = [d for t in self.topology for d in t]
        self.completed_ops = dict()
        self._call_args = None
        self._call_kwargs = None

    def __call__(self, *args, **kwargs):
        self._call_args = args
        self._call_kwargs = kwargs
        return self.run_dag(*args, **kwargs)

    def __getitem__(self, item):
        return self.completed_ops.get(item, self.name_to_op_map[item])

    def _create_dag_logger(self, name, log_level):
        logger = logging.getLogger(name=name)
        logger.setLevel(log_level)

        # Don't add extra handlers - will see duplicated outputs
        if len(logger.handlers) == 0:
            # let the logger fitler - stream should output anything it gets (for now...)
            # In future, allow finer grained control - per op basis?
            handler_level = 'DEBUG'
            self._log_stream_handler = getattr(self, '_log_stream_handler',
                                               logging.StreamHandler())
            self._log_stream_handler.setLevel(handler_level)
            self._log_formatter = getattr(self, '_log_formatter',
                                          logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self._log_stream_handler.setFormatter(self._log_formatter)
            logger.addHandler(self._log_stream_handler)
        return logger

    def _ipython_key_completions_(self):
        return list(self.name_to_op_map.keys())

    def _record_utilization(self):
        import os
        import psutil
        process = psutil.Process(os.getpid())
        t = time.time()
        self.system_utilization[t] = process.memory_info().rss

    def get_op_logger(self, op):
        if op not in self.op_loggers:
            l = self._create_dag_logger(self.get_dag_unique_op_name(op), self.log_level)
            self.op_loggers[op] = l
        return self.op_loggers[op]

    def build(self, required_outputs):
        if not isinstance(required_outputs, list):
            msg = "DAG's build method only accepts a list of outputs"
            raise ValueError(msg)

        deps_to_resolve = required_outputs
        dep_map = dict()
        all_ops = dict()
        input_op_map = dict()
        # Iterate rather than recurse
        # deps_to_resolve treated like a FIFO queue
        while len(deps_to_resolve) > 0:
            # Pop a dep
            o = deps_to_resolve[0]
            deps_to_resolve = deps_to_resolve[1:]
            #####-----
            # If Op already registered, move on
            if any(o == _o for _o in all_ops):
                continue

            # Give each Op a reference to this DAG
            o._set_dag(self)

            # If name is duplicated, register a unique suffix for the op
            o_name = o.get_name()
            self.op_name_counts[o_name] = self.op_name_counts.get(o_name, 0) + 1
            if self.op_name_counts[o_name] > 1:
                self.op_suffixes[o] = '_%d' % (self.op_name_counts[o_name] - 1)

            # Ops stored in a dict for easy lookup
            all_ops[o] = o

            try:
                # Try producing the input Ops that need to be given to this Op
                dep_map[o] = o.requires()
            except:
                print("Error producing requires for %s" % o.get_name())
                raise

            # requires can return a map (for kwargs) or a list (for args)
            if isinstance(dep_map[o], (list, tuple)):
                if isinstance(dep_map[o], tuple):
                    dep_map[o] = list(dep_map[o])
                deps_to_resolve += dep_map[o]
            elif isinstance(dep_map[o], dict):
                deps_to_resolve += list(dep_map[o].values())
            else:
                from dsdag.core.op import OpVertexAttr
                t = (OpVertex, OpVertexAttr)
                if not isinstance(dep_map[o], t) and not issubclass(type(dep_map[o]), t):

                    msg = "%s requires returned %s - Op requires must return a list or dict of OpVertices or a single OpVertex"
                    raise ValueError(msg % (str(o), str(dep_map[o])))
                # Treat single op returns like a list with only one element
                dep_map[o] = [dep_map[o]]
                deps_to_resolve += dep_map[o]

            # With every new Op, we want to check that this Op isn't
            # already being satisfied.

            # Go through all ops that have been processed at this point
            for o in all_ops.keys():
                if isinstance(dep_map[o], dict):
                    #For this operations dependencies (dict)
                    for req_k in dep_map[o].keys():
                        # If this is already satisified
                        if dep_map[o][req_k] in all_ops:
                            # Take the existing (resolved) op and overwrite
                            # this ops dependency to it # However, the key object is not overwritten - so explicitly delete and update
                            # TODO: Is this still necessary?
                            _o = all_ops[dep_map[o][req_k]]
                            del(all_ops[_o])
                            dep_map[o][req_k] = _o
                            all_ops[_o] = _o
                elif isinstance(dep_map[o], list):
                    #For this operations dependencies (list)
                    for i, req_k in enumerate(dep_map[o]):
                        if req_k in all_ops:
                            _o = all_ops[dep_map[o][i]]
                            dep_map[o][i] = _o
                            del all_ops[_o]
                            all_ops[_o] = _o
                else:
                    msg = "Found unsupported Op dependency object %s" % type(o)
                    raise ValueError(msg)

        import dsdag
        input_op_map = {k: o for k, o in all_ops.items() if isinstance(o, dsdag.ext.misc.InputOp)}
        return all_ops, dep_map, input_op_map

    def exec_process(self, process,
                     tpid,
                     proc_exec_kwargs=None,
                     proc_args=None):
        p = process

        if not self.pbar:
            self.logger.info("%s Executing %s" % (tpid, process.__class__.__name__))

        self._record_utilization()
        start_t = time.time()
        if proc_exec_kwargs is None and proc_args is None:
            self.outputs[process] = p.run()
        elif proc_exec_kwargs is not None and proc_args is None:
            self.outputs[process] = p.run(**proc_exec_kwargs)
        elif proc_exec_kwargs is None and proc_args is not None:
            self.outputs[process] = p.run(*proc_args)
        elif proc_exec_kwargs is not None and proc_args is not None:
            self.outputs[process] = p.run(*proc_args, **proc_exec_kwargs)

        finish_t = time.time()
        self._record_utilization()
        self.start_and_finish_times[process] = (start_t, finish_t)

        if not self.pbar:
            self.logger.info("%s %s completed in %.2fs" % (tpid,
                                                           process.__class__.__name__,
                                                           finish_t - start_t))
        self.completed_ops[p.get_name()] = p

    def set_runtime(self, **kwargs):
        for k, v in kwargs.items():
            self.runtime_parameters[k] = None

    def collect_op_inputs(self, op):
        dependencies = self.dep_map.get(op, dict())
        process_name = op.get_name()
        proc_args, proc_kwargs = list(), dict()

        if isinstance(dependencies, dict):
            for k, v in dependencies.items():
                if v not in self.outputs:
                    msg = "The process %s has a missing dependency:" % process_name
                    msg += "%s=%s" % (k, v.__class__.__name__)
                    self.logger.error(msg=msg)
                    return -1
                if isinstance(self.outputs[v], idt.RepoLeaf):
                    self.outputs[v] = self.outputs[v].load()

                if op.is_unpack_required(v) or v.unpack_output:
                    if isinstance(self.outputs[v], dict):
                        proc_kwargs.update(self.outputs[v])
                    else:
                        proc_args += list(self.outputs[v])
                else:
                    proc_kwargs[k] = self.outputs[v]

        elif isinstance(dependencies, list):
            for _i, v in enumerate(dependencies):
                if v not in self.outputs:
                    msg = "The process %s (%s) has a missing dependency:" % (process_name,
                                                                             op.__class__.__name__)
                    msg += "%s=%s" % ("*arg[%d]" % _i, v.__class__.__name__)
                    msg += "\n" + str(v)
                    msg += "\n" + str(self.get_dag_unique_op_name(v))
                    msg += "\n" + str(hash(v))
                    self.logger.error(msg=msg)
                    return -1
                if isinstance(self.outputs[v], idt.RepoLeaf):
                    self.outputs[v] = self.outputs[v].load()

                if op.is_unpack_required(v) or v.unpack_output:
                    if isinstance(self.outputs[v], dict):
                        proc_kwargs.update(self.outputs[v])
                    else:
                        proc_args += list(self.outputs[v])
                else:
                    proc_args.append(self.outputs[v])

        return proc_args, proc_kwargs

    def dependencies_in_cache(self, process):
        process_name = self.get_dag_unique_op_name(process)

        import inspect

        #src_hash = hash(inspect.getsource(process.__class__))

        if isinstance(self.cache, dict) and process in self.cache:
            process_in_cache = True
        elif isinstance(self.cache, idt.RepoTree) and process_name in self.cache:
            import hashlib
            m = hashlib.sha256()
            m.update(str(process).encode('utf-8'))
            #m.update(str(process._req_hashable))
            h = str(m.digest())

            l = self.cache[process_name]
            #process_src = inspect.getsource(process.__class__)
            #for c_k, c_v in l.read_metadata()
            cache_md = l.read_metadata()
            hash_match = cache_md['op_hash'] == h

            #params_match = cache_md.get('op_params', list()) == process._param_hashable
            #src_match = cache_md.get('op_class_src_hash', 0) == hash(process_src)

            # TODO: what if requires was different?
            #process_in_cache = params_match and src_match
            process_in_cache = hash_match
        else:
            process_in_cache = False


        #process_in_cache = ((isinstance(self.cache, dict) and process in self.cache)
        #                    or
        #                    (isinstance(self.cache, idt.RepoTree)
        #                     and process_name in self.cache
        #                     and hash(process) == self.cache[process_name].md['op_hash']))
        return process_in_cache

    def find_min_topology(self):#, ops, topology=[]):
        #dep_satisfied = lambda dep: (dep._cacheable and
        #                             self.using_cache and
        #                             self.cache is not None and
        #                            (dep in self.cache or dep.get_name() in self.cache))
        #dep_satisfied = self.dependencies_in_cache()

        dep_map = dict()
        deps_to_resolve = list(self.required_outputs)
        while len(deps_to_resolve) > 0:
            o = deps_to_resolve[0]
            deps_to_resolve = deps_to_resolve[1:]

            reqs = self.dep_map[o]
            # No matter what, the dag needs to see the requirements
            dep_map[o] = reqs

            # reqs can be kwargs or args (list)
            _iter_reqs = reqs.values() if isinstance(reqs, dict) else reqs
            # But only need to resolve those that are not satisfied already
            deps_to_resolve += [r for r in _iter_reqs if not self.dependencies_in_cache(r)]


        dep_sets = {p: set(d.values()) if isinstance(d, dict) else set(d)
                    for p, d in dep_map.items()}
        topology = list(toposort(dep_sets))
        return topology

    def run_dag(self, *args, **kwargs):
        self.dag_start_time = time.time()
        computed = list()

        topology = self.find_min_topology()
        run_reqs = [o for l in topology for o in l]

        if self.pbar:
            from tqdm.auto import tqdm
            self.tqdm_bar = tqdm(total=len(run_reqs))

        # For each ter
        for i, ind_processes in enumerate(topology):
            ind_processes = list(ind_processes)
            # For each dependency in the iter
            for j, process in enumerate(ind_processes):
                if self.live_browse and len(self.outputs) > 0:
                    self.browse()

                process_name = self.get_dag_unique_op_name(process)

                if self.pbar:
                    self.tqdm_bar.set_description(process_name)
                else:
                    self.logger.info(process_name)

                # May be a list of dependencies (*args) or a dict (**kwargs)
                dependencies = self.dep_map.get(process, dict())
                dep_values = dependencies.values() if isinstance(dependencies, dict) else dependencies

                tpid = "[%d.%d]" % (i, j)
                #process_in_cache = ((isinstance(self.cache, dict) and process in self.cache)
                #                    or
                #                    (isinstance(self.cache, idt.RepoTree)
                #                     and process_name in self.cache
                #                     and hash(process) == self.cache[process_name].md['op_hash']))
                process_in_cache = self.dependencies_in_cache(process)
                load_from_cache = (process not in self.required_outputs  # always run specified vertices
                                   # Op is set to cacheable (default)
                                   and getattr(process, '_cacheable', False)
                                   # If None of an Ops depends has been (re)computed
                                   # and we're not forcing all downstream to run
                                   and (not any(p in computed for p in dep_values)
                                        or not self.force_downstream_rerun)
                                    # DAG is set to read from cache
                                   and self.read_from_cache
                                    # Proces is in the cache (check dict or IDT)
                                   and process_in_cache
                                   #and (process_name in self.cache or process in self.cache)
                                   )
                if process in self.input_op_map:
                    pass

                if load_from_cache:
                    if not self.pbar:
                        self.logger.info("%s Will use cached output of %s"
                                         % (tpid, process_name))
                    if isinstance(self.cache, idt.RepoTree):
                        self.outputs[process] = self.cache[process_name]
                    else:
                        self.outputs[process] = self.cache[process]
                else:
                    proc_args, proc_kwargs = self.collect_op_inputs(process)

                    self.exec_process(process=process,
                                      proc_args=proc_args,
                                      proc_exec_kwargs=proc_kwargs,
                                      tpid=tpid)
                    computed.append(process)

                if self.pbar:
                    self.tqdm_bar.update(1)

                future_deps = list()
                #####
                # Go through future Ops to determine dependencies

                # Go through remaining processes at this level
                for p in ind_processes[j+1:]:
                    deps = self.dep_map.get(p, dict())
                    dep_iter = deps if isinstance(deps, list) else deps.values()
                    future_deps += list(d for d in dep_iter)

                # Go through future levels
                ### Include this level - so really we are removing leftovers from the previous level
                for topo in topology[i:]:
                    for p in topo:
                        deps = self.dep_map.get(p, dict())
                        vals = deps.values() if isinstance(deps, dict) else deps
                        future_deps += list(d for d in vals)
                future_deps = set(future_deps)
                self.logger.debug("%d Future deps" % len(future_deps))

                 # No reason to save the cached output back
                if self.write_to_cache and not load_from_cache and process._cacheable:
                    if not self.pbar:
                        self.logger.info("Persisting output of %s" % process_name)
                    if isinstance(self.cache, idt.RepoTree):
                        import hashlib
                        m = hashlib.sha256()
                        m.update(str(process).encode('utf-8'))
                        #m.update(str(process._req_hashable))
                        h = str(m.digest())
                        self.cache.save(self.outputs[process],
                                            name=process_name,
                                            author='dag',
                                        op_hash=h,
                                            #param_names=list(proc_kwargs.keys()),
                                        #op_params=process._param_hashable,
                                        #op_class_src_hash=hash(inspect.getsource(process.__class__)),
                                            auto_overwrite=True)
                    elif isinstance(self.cache, dict):
                        self.cache[process] = self.outputs[process]

                for k in list(self.outputs.keys()):
                    if k not in future_deps and k not in self.required_outputs:
                        self.logger.debug("Deleting future dep %s" % k.__class__.__name__)
                        del self.outputs[k]

                        in_cache = (k.__class__.__name__ in self.cache or k in self.cache)
                        if k._cacheable and self.cache_eviction and in_cache:
                            self.logger.debug("Removing unnecessary op from cache %s"
                                              % k.__class__.__name__)
                            if isinstance(self.cache, idt.RepoTree):
                                self.cache[k.__class__.__name__].delete('dag')
                            else:
                                del self.cache[k]

        if self.pbar:
            self.tqdm_bar.close()

        if self.live_browse and len(self.outputs) > 0:
            self.browse()

        return self.get_op_output(self.required_outputs)

    def get_op_input(self, node):
        return self.dep_map[node]

    def get_op_output(self, nodes=None):
        if nodes is None:
            nodes = self.required_outputs
        o = [self.outputs[n] for n in nodes]
        if len(o) == 1:
            return o[0]
        else:
            return o

    def get_dag_unique_op_name(self, o, on_missing='error'):
        if on_missing == 'error':
            #if o not in self.op_suffixes:
            if o not in self.all_ops:
                raise KeyError("Op %s not in DAG %s" % (str(o), str(self)))
            n = o.get_name() + self.op_suffixes.get(o, '')
        elif on_missing == 'ignore':
           n = o.get_name() + self.op_suffixes.get(o, '')
        else:
            raise ValueError("on_missing parameter must be on of {'error', 'ignore'}, but got '%s'"
                             % str(on_missing))
        return n

    def clear_cache(self):
        self.logger.info("clearing cache")
        if isinstance(self.cache, idt.RepoTree):
            for k in self.name_to_op_map.keys():
                if k in self.cache:
                    self.logger.info("Removing %s from repo cache" % k)
                    self.cache[k].delete('dag')
        elif self.cache is not None:
            for k in self.all_ops:
                if k in self.cache:
                    self.logger.info("Removing %s from cache" % k)
                    del self.cache[k]
        return self

    def viz(self, fontsize='10',
            color_mapping=None,
            cache_color=None,
            #format='png',
            friendly_names=True,
            rankdir='UD',
            return_dot_object=False):
        from graphviz import Digraph
        import tempfile

        merged = self.dep_map

        dot = Digraph(comment='chassis viz', format='png',
                      graph_attr=dict(rankdir=rankdir),
                      node_attr={'fontsize': fontsize})
        dot.node_attr.update(color='lightblue2', style='filled')

        def _get_name(m):
            return self.get_dag_unique_op_name(m)
            #if friendly_names:
            #    return m.get_name()
            #else:
            #    return m.unique_cls_name

        for m, deps in merged.items():
            n = _get_name(m)

            node_attrs = m._get_viz_attrs()
            if color_mapping is not None and n in color_mapping:
                node_attrs['color'] = color_mapping[n]

            if cache_color is not None and m in self.cache:
                node_attrs['color'] = cache_color


            dot.node(n, **node_attrs)

        for m, deps in merged.items():
            n = _get_name(m)
            vals = deps.values() if isinstance(deps, dict) else deps
            for d in vals:
                dn = _get_name(d)
                dot.edge(dn, n)

        if return_dot_object:
            return dot
        else:
            from IPython.display import Image, display
            import uuid
            img_name = dot.render(cleanup=True,
                                  directory=tempfile.gettempdir(),
                                  filename=str(uuid.uuid4()))

            display(Image(filename=img_name))

    def browse(self, show_dag=True):
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        clear_output()
        tab_child_widgets = list()
        tab_titles = list()

        # Ordering of this is based on topology
        for op in self.all_requirements:
            if op not in self.outputs:
                continue
            try:
                viz_out = op.op_nb_viz(self.outputs[op])
            except NotImplementedError:
                if isinstance(self.outputs[op], pd.DataFrame):
                    from dsdag.op_library.templates.dataframe import FrameBrowseMaixin
                    #viz_out = FrameBrowseMaixin.op_nb_viz(self.outputs[op])
                    viz_out = FrameBrowseMaixin.build_df_browse_widget(self.outputs[op])
                else:
                    viz_out = widgets.Output()
                    viz_out.append_display_data("Op %s has no viz method" % op.unique_cls_name)
            tab_child_widgets.append(viz_out)
            tab_titles.append(op.unique_cls_name)

        tab = widgets.Tab(layout=widgets.Layout(width='900px'))
        #tab = widgets.Accordion(layout=widgets.Layout(width='1000px'))

        tab.children = tab_child_widgets
        for i, c in enumerate(tab_child_widgets):
            tab.set_title(i, tab_titles[i])

        if show_dag:
            from IPython.display import SVG
            dag_out = widgets.Output(layout=widgets.Layout(width='800px'))
            dag_out.append_display_data(SVG(self.viz()._repr_svg_()))
            #dag_out = SVG(self.viz()._repr_svg_())
            out_widget = widgets.VBox([dag_out, tab])
            out_widget.layout.height = '400px'
        else:
            out_widget = tab
        display(out_widget)


    def timings(self):
        """
        Returns a Pandas series of the Op latencies in seconds
        """
        op_lat_map = {op.get_name(): end_t - start_t
                      for op, (start_t, end_t) in self.start_and_finish_times.items()}
        op_latency_s = pd.Series(op_lat_map, name='op_latency').sort_values(ascending=False)
        return op_latency_s

    def plot(self):
        import matplotlib
        from matplotlib import pyplot as plt
        import matplotlib.dates as mdates
        sys_s = pd.Series(self.system_utilization)
        sys_s.index = pd.to_datetime(sys_s.index, unit='s')
        sys_s = sys_s / (1024. ** 2)

        height = 0.20 * len(self.start_and_finish_times.keys())
        fig, ax = plt.subplots(figsize=(16, height))
        # gd.configure_imports(matplotlib_style='default')

        ax2 = ax.twinx()
        ax2 = sys_s.rename("Memory").plot(ax=ax2, legend=True)
        ax2.set_ylabel("Memory Usage (MB)", fontsize=15)
        ax2.grid(False)

        names = list()
        # ylim = ax.get_ylim()

        sorted_iter = sorted(self.start_and_finish_times.keys(),
                             key=lambda v: self.start_and_finish_times[v][0])

        for i, _op in enumerate(sorted_iter):
            start_t, stop_t = self.start_and_finish_times[_op]

            start_t_s = mdates.date2num(pd.to_datetime(start_t, unit='s'))
            stop_t_s = mdates.date2num(pd.to_datetime(stop_t, unit='s'))
            ax.barh(i, stop_t_s - start_t_s, left=start_t_s,
                    height=.5, align='center')
            names.append(_op.get_name())

        ax.set(yticks=range(len(names)), yticklabels=names)
        ax.set_xlabel('Time', fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(True)
        fig.tight_layout()


class DAG2(DAG):
    def __init__(self, required_outputs,
                 read_from_cache=False,
                 write_to_cache=False,
                 cache=None,
                 cache_eviction=False,
                 force_downstream_rerun=True,
                 pbar=True,
                 live_browse=False,
                 logger=None):
        """
        Build a DAG that produces outputs from a set of Ops. After construction, calling the DAG will
        commence processing and return the required outputs from the relevant Ops.

        :param required_outputs: (OpVertex) Ops whose output is to be returned/materialized
        :param read_from_cache: (bool) True if Op outputs should be read from the provided cache
        :param write_to_cache: (bool) True if Op outputs should be written to the provided cache
        :param cache: (RepoTree) Caching location
        :param cache_eviction: (bool) If True, only the Op outputs required for the dag at each step are kept in the cache
        :param force_downstream_rerun: (bool) True if all dependent outputs should be rerun after an Op is run
        :param pbar: (bool) Use tqdm progress bar
        :param live_browse: (bool) Experimental
        :param logger: (logger or string level) Pass logger level (e.g. 'INFO', 'WARN') to set default level or pass
                        custom logger
        """
        #if isinstance(logger, basestring):
        if isinstance(logger, str):
            log_level = logger
            logger = None
        else:
            log_level = DAG._default_log_level

        if logger is None:
            logger = self._create_dag_logger('DAG', log_level)

        self.logger = logger
        self.log_level = log_level
        self.op_loggers = dict()

        ####
        self.live_browse = live_browse

        self.read_from_cache = read_from_cache
        self.write_to_cache = write_to_cache
        self.using_cache = (self.read_from_cache or self.write_to_cache)
        self.force_downstream_rerun = force_downstream_rerun

        self.cache_eviction = cache_eviction
        if self.using_cache and (cache is None):
            logger.info("Using dict cache")
            self.cache = DAG._CACHE
        elif cache is not None:
            self.cache = cache
        else:
            logger.info("Using dict cache")
            self.cache = DAG._CACHE


        if not isinstance(required_outputs, (list, tuple)):
            self.required_outputs = [required_outputs]
        elif isinstance(required_outputs, tuple):
            self.required_outputs = list(required_outputs)
        elif isinstance(required_outputs, list):
            self.required_outputs = required_outputs
        else:
            msg = "Expected required outputs to be a single object, list, or tuple.\n"
            msg += "Got %s" % str(type(required_outputs))
            raise ValueError(msg)

        self.output_ordering_map = {ro: i for i, ro in enumerate(self.required_outputs)}
        self.lazy_outputs = [ro for ro in self.required_outputs if isinstance(ro, str)]
        if len(self.lazy_outputs) > 0:
            self.logger.info("Lazy required outputs found: %s"
                             % ", ".join(self.lazy_outputs))
            self.required_outputs = list(set(self.required_outputs) - set(self.lazy_outputs))


        self.pbar = pbar

        self.op_name_counts = dict()# Counter(o.get_name() for o in self.all_ops.values())
        self.op_suffixes = dict()
        self.all_ops, self.dep_map, self.input_op_map = self.build(self.required_outputs)


        self.runtime_parameters = dict()
        for op in self.all_ops:
            for p_name, param in op.opk.runtime_parameters.items():
                self.runtime_parameters[p_name] = param

        self.name_to_op_map = {o.get_name():o for o in self.all_ops.values()}
        for lo in self.lazy_outputs:
            if lo not in self.name_to_op_map:
                self.logger.error("Lazy Op %s is not in resulting build Ops: %s"
                                    % (lo, "\n" + "\n".join(self.name_to_op_map.keys())))
                raise ValueError("Lazy Op %s could not be resolved after build" % lo)

            self.logger.info("Adding %s to outputs" % lo)
            self.required_outputs.append(self.name_to_op_map[lo])
            self.output_ordering_map[self.name_to_op_map[lo]] = self.output_ordering_map[lo]
            del self.output_ordering_map[lo]
            self.required_outputs = list(sorted(self.required_outputs,
                                                key=self.output_ordering_map.get))

        self.dep_sets = {p:set(d.values()) if isinstance(d, dict) else set(d)
                         for p, d in self.dep_map.items()}
        self.topology = list(toposort(self.dep_sets))

        self.outputs = dict()
        self.dag_start_time = None
        self.system_utilization = dict()
        self.start_and_finish_times = dict()
        self.all_requirements = [d for t in self.topology for d in t]
        self.completed_ops = dict()
        self._call_args = None
        self._call_kwargs = None


    def collect_op_inputs(self, op):
        dependencies = self.dep_map.get(op, dict())
        process_name = op.get_name()
        proc_args, proc_kwargs = list(), dict()

        if isinstance(dependencies, dict):
            for k, v in dependencies.items():
                if v not in self.outputs:
                    msg = "The process %s has a missing dependency:" % process_name
                    msg += "%s=%s" % (k, v.__class__.__name__)
                    self.logger.error(msg=msg)
                    return -1
                if isinstance(self.outputs[v], idt.RepoLeaf):
                    self.outputs[v] = self.outputs[v].load()

                if op.opk.is_unpack_required(v) or v.opk.unpack_output:
                    if isinstance(self.outputs[v], dict):
                        proc_kwargs.update(self.outputs[v])
                    else:
                        proc_args += list(self.outputs[v])
                else:
                    proc_kwargs[k] = self.outputs[v]

        elif isinstance(dependencies, list):
            for _i, v in enumerate(dependencies):
                if v not in self.outputs:
                    msg = "The process %s (%s) has a missing dependency:" % (process_name,
                                                                             op.__class__.__name__)
                    msg += "%s=%s" % ("*arg[%d]" % _i, v.__class__.__name__)
                    msg += "\n" + str(v)
                    msg += "\n" + str(self.get_dag_unique_op_name(v))
                    msg += "\n" + str(hash(v))
                    self.logger.error(msg=msg)
                    return -1
                if isinstance(self.outputs[v], idt.RepoLeaf):
                    self.outputs[v] = self.outputs[v].load()

                if op.opk.is_unpack_required(v) or v.opk.unpack_output:
                    if isinstance(self.outputs[v], dict):
                        proc_kwargs.update(self.outputs[v])
                    else:
                        proc_args += list(self.outputs[v])
                else:
                    proc_args.append(self.outputs[v])

        return proc_args, proc_kwargs


    def build(self, required_outputs):
        if not isinstance(required_outputs, list):
            msg = "DAG's build method only accepts a list of outputs"
            raise ValueError(msg)

        deps_to_resolve = required_outputs
        dep_map = dict()
        all_ops = dict()
        input_op_map = dict()
        # Iterate rather than recurse
        # deps_to_resolve treated like a FIFO queue
        while len(deps_to_resolve) > 0:
            # Pop a dep
            o = deps_to_resolve[0]
            deps_to_resolve = deps_to_resolve[1:]
            #####-----
            # If Op already registered, move on
            if any(o == _o for _o in all_ops):
                continue

            # Give each Op a reference to this DAG
            o._set_dag(self)

            # If name is duplicated, register a unique suffix for the op
            o_name = o.get_name()
            self.op_name_counts[o_name] = self.op_name_counts.get(o_name, 0) + 1
            if self.op_name_counts[o_name] > 1:
                self.op_suffixes[o] = '_%d' % (self.op_name_counts[o_name] - 1)

            # Ops stored in a dict for easy lookup
            all_ops[o] = o

            try:
                # Try producing the input Ops that need to be given to this Op
                #dep_map[o] = o.opk.requires_callable() if o.opk.requires_callable is not None else dict()
                req = o.opk.get_requires(o)
                dep_map[o] = req() if req is not None else dict()
            except:
                print("Error producing requires for %s" % o.get_name())
                raise

            # requires can return a map (for kwargs) or a list (for args)
            if isinstance(dep_map[o], (list, tuple)):
                if isinstance(dep_map[o], tuple):
                    dep_map[o] = list(dep_map[o])
                deps_to_resolve += dep_map[o]
            elif isinstance(dep_map[o], dict):
                deps_to_resolve += list(dep_map[o].values())
            else:
                from dsdag.core.op import OpVertexAttr
                t = (OpVertex, OpVertexAttr)
                if not isinstance(dep_map[o], t) and not issubclass(type(dep_map[o]), t):

                    msg = "%s requires returned %s - Op requires must return a list or dict of OpVertices or a single OpVertex"
                    raise ValueError(msg % (str(o), str(dep_map[o])))
                # Treat single op returns like a list with only one element
                dep_map[o] = [dep_map[o]]
                deps_to_resolve += dep_map[o]

            # With every new Op, we want to check that this Op isn't
            # already being satisfied.

            # Go through all ops that have been processed at this point
            for o in all_ops.keys():
                if isinstance(dep_map[o], dict):
                    #For this operations dependencies (dict)
                    for req_k in dep_map[o].keys():
                        # If this is already satisified
                        if dep_map[o][req_k] in all_ops:
                            # Take the existing (resolved) op and overwrite
                            # this ops dependency to it # However, the key object is not overwritten - so explicitly delete and update
                            # TODO: Is this still necessary?
                            _o = all_ops[dep_map[o][req_k]]
                            del(all_ops[_o])
                            dep_map[o][req_k] = _o
                            all_ops[_o] = _o
                elif isinstance(dep_map[o], list):
                    #For this operations dependencies (list)
                    for i, req_k in enumerate(dep_map[o]):
                        if req_k in all_ops:
                            _o = all_ops[dep_map[o][i]]
                            dep_map[o][i] = _o
                            del all_ops[_o]
                            all_ops[_o] = _o
                else:
                    msg = "Found unsupported Op dependency object %s" % type(o)
                    raise ValueError(msg)

        import dsdag
        input_op_map = {k: o for k, o in all_ops.items()
                        if isinstance(o, dsdag.ext.misc.InputOp)}
        return all_ops, dep_map, input_op_map


    def run_dag(self, *args, **kwargs):
        self.dag_start_time = time.time()
        computed = list()

        topology = self.find_min_topology()
        run_reqs = [o for l in topology for o in l]

        if self.pbar:
            from tqdm.auto import tqdm
            self.tqdm_bar = tqdm(total=len(run_reqs))

        # For each ter
        for i, ind_processes in enumerate(topology):
            ind_processes = list(ind_processes)
            # For each dependency in the iter
            for j, process in enumerate(ind_processes):
                if self.live_browse and len(self.outputs) > 0:
                    self.browse()

                process_name = self.get_dag_unique_op_name(process)

                if self.pbar:
                    self.tqdm_bar.set_description(process_name)
                else:
                    self.logger.info(process_name)

                # May be a list of dependencies (*args) or a dict (**kwargs)
                dependencies = self.dep_map.get(process, dict())
                dep_values = dependencies.values() if isinstance(dependencies, dict) else dependencies

                tpid = "[%d.%d]" % (i, j)
                #process_in_cache = ((isinstance(self.cache, dict) and process in self.cache)
                #                    or
                #                    (isinstance(self.cache, idt.RepoTree)
                #                     and process_name in self.cache
                #                     and hash(process) == self.cache[process_name].md['op_hash']))
                process_in_cache = self.dependencies_in_cache(process)
                load_from_cache = (process not in self.required_outputs  # always run specified vertices
                                   # Op is set to cacheable (default)
                                   and getattr(process, '_cacheable', False)
                                   # If None of an Ops depends has been (re)computed
                                   # and we're not forcing all downstream to run
                                   and (not any(p in computed for p in dep_values)
                                        or not self.force_downstream_rerun)
                                    # DAG is set to read from cache
                                   and self.read_from_cache
                                    # Proces is in the cache (check dict or IDT)
                                   and process_in_cache
                                   #and (process_name in self.cache or process in self.cache)
                                   )
                if process in self.input_op_map:
                    pass

                if load_from_cache:
                    if not self.pbar:
                        self.logger.info("%s Will use cached output of %s"
                                         % (tpid, process_name))
                    if isinstance(self.cache, idt.RepoTree):
                        self.outputs[process] = self.cache[process_name]
                    else:
                        self.outputs[process] = self.cache[process]
                else:
                    proc_args, proc_kwargs = self.collect_op_inputs(process)

                    self.exec_process(process=process,
                                      proc_args=proc_args,
                                      proc_exec_kwargs=proc_kwargs,
                                      tpid=tpid)
                    computed.append(process)

                if self.pbar:
                    self.tqdm_bar.update(1)

                future_deps = list()
                #####
                # Go through future Ops to determine dependencies

                # Go through remaining processes at this level
                for p in ind_processes[j+1:]:
                    deps = self.dep_map.get(p, dict())
                    dep_iter = deps if isinstance(deps, list) else deps.values()
                    future_deps += list(d for d in dep_iter)

                # Go through future levels
                ### Include this level - so really we are removing leftovers from the previous level
                for topo in topology[i:]:
                    for p in topo:
                        deps = self.dep_map.get(p, dict())
                        vals = deps.values() if isinstance(deps, dict) else deps
                        future_deps += list(d for d in vals)
                future_deps = set(future_deps)
                self.logger.debug("%d Future deps" % len(future_deps))

                 # No reason to save the cached output back
                if self.write_to_cache and not load_from_cache and process._cacheable:
                    if not self.pbar:
                        self.logger.info("Persisting output of %s" % process_name)
                    if isinstance(self.cache, idt.RepoTree):
                        import hashlib
                        m = hashlib.sha256()
                        m.update(str(process).encode('utf-8'))
                        #m.update(str(process._req_hashable))
                        h = str(m.digest())
                        self.cache.save(self.outputs[process],
                                            name=process_name,
                                            author='dag',
                                        op_hash=h,
                                            #param_names=list(proc_kwargs.keys()),
                                        #op_params=process._param_hashable,
                                        #op_class_src_hash=hash(inspect.getsource(process.__class__)),
                                            auto_overwrite=True)
                    elif isinstance(self.cache, dict):
                        self.cache[process] = self.outputs[process]

                for k in list(self.outputs.keys()):
                    if k not in future_deps and k not in self.required_outputs:
                        self.logger.debug("Deleting future dep %s" % k.__class__.__name__)
                        del self.outputs[k]

                        in_cache = (k.__class__.__name__ in self.cache or k in self.cache)
                        if k.opk.cacheable and self.cache_eviction and in_cache:
                            self.logger.debug("Removing unnecessary op from cache %s"
                                              % k.__class__.__name__)
                            if isinstance(self.cache, idt.RepoTree):
                                self.cache[k.__class__.__name__].delete('dag')
                            else:
                                del self.cache[k]

        if self.pbar:
            self.tqdm_bar.close()

        if self.live_browse and len(self.outputs) > 0:
            self.browse()

        return self.get_op_output(self.required_outputs)