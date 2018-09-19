import logging
import interactive_data_tree as idt
import time
from toposort import toposort
import pandas as pd #TODO: this dep should not be in core dag

# TODO: Must dedup the Operations so that they are run once in the depends
#       keep track of different ops and their params - don't instantiate the same
class DAG(object):
    def __init__(self, required_outputs,
                 read_from_cache=False,
                 write_to_cache=False,
                 cache=None,
                 force_downstream_rerun=True,
                 pbar=True,
                 live_browse=False,
                 logger=None):
        if isinstance(logger, basestring):
            #if logger.lower() in ('WARN', 'INFO', 'ERROR', 'DEBUG')
            log_level = logger
            logger = None
        else:
            log_level = 'WARN'

        if logger is None:
            logger = logging.getLogger()
            logger.setLevel(log_level)
        self.logger = logger
        ####
        self.live_browse = live_browse

        self.read_from_cache = read_from_cache
        self.write_to_cache = write_to_cache
        self.using_cache = (self.read_from_cache or self.write_to_cache)
        self.force_downstream_rerun = force_downstream_rerun
        if self.using_cache and (cache is None):
            self.logger("Use cache set, but None provided, creating default")
            self.cache = idt.RepoTree()
        elif cache is not None:
            self.cache = cache
        else:
            self.cache = dict()

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

        self.pbar = pbar

        self.all_ops, self.dep_map = self.build(self.required_outputs)
        self.name_to_op_map = {o.get_name():o for o in self.all_ops.values()}

        self.dep_sets = {p:set(d.values()) if isinstance(d, dict) else set(d)
                         for p, d in self.dep_map.items()}
        self.topology = list(toposort(self.dep_sets))

        self.outputs = dict()
        self.start_and_finish_times = dict()
        self.all_requirements = [d for t in self.topology for d in t]
        self.completed_ops = dict()

    def __call__(self, *args, **kwargs):
        return self.run_dag()

    def __getitem__(self, item):
        return self.completed_ops.get(item, self.name_to_op_map[item])

    def build(self, required_outputs):
        if not isinstance(required_outputs, list):
            msg = "DAG's build method only accepts a list of outputs"
            raise ValueError(msg)

        deps_to_resolve = required_outputs
        dep_map = dict()
        all_ops = dict()
        # Iterate rather than recurse
        # deps_to_resolve treated like a FIFO queue
        while len(deps_to_resolve) > 0:
            # Pop a dep
            o = deps_to_resolve[0]
            deps_to_resolve = deps_to_resolve[1:]
            #####-----
            if any(o == _o for _o in all_ops):
                continue

            o._set_dag(self)
            all_ops[o] = o
            dep_map[o] = o.requires()

            # requires can return a map (for kwargs) or a list (for args)
            if isinstance(dep_map[o], list):
                deps_to_resolve += list(dep_map[o])
            elif isinstance(dep_map[o], dict):
                deps_to_resolve += list(dep_map[o].values())
            else:
                raise ValueError("requires must return a list or dict")

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
                            # this ops dependency to it
                            # However, the key object is not overwritten - so explicitly delete and update
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

        return all_ops, dep_map

    def exec_process(self, process,
                     tpid,
                     proc_exec_kwargs=None,
                     proc_args=None):
        p = process

        if not self.pbar:
            self.logger.info("%s Executing %s" % (tpid, process.__class__.__name__))

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
        self.start_and_finish_times[process] = (start_t, finish_t)

        if not self.pbar:
            self.logger.info("%s %s completed in %.2fs" % (tpid,
                                                           process.__class__.__name__,
                                                           finish_t - start_t))
        self.completed_ops[p.get_name()] = p

    def run_dag(self):
        computed = set()
        if self.pbar:
            from tqdm import tqdm
            self.tqdm_bar = tqdm(total=len(self.all_requirements))

        # For each ter
        for i, ind_processes in enumerate(self.topology):
            ind_processes = list(ind_processes)
            # For each dependency in the iter
            for j, process in enumerate(ind_processes):
                if self.live_browse and len(self.outputs) > 0:
                    self.browse()
                process_cls_name = process.__class__.__name__
                process_friendly_name = process.get_name()
                process_repr = repr(process)
                if self.pbar:
                    self.tqdm_bar.set_description(process_friendly_name)
                else:
                    self.logger.info(process_friendly_name)

                dependencies = self.dep_map.get(process, dict())
                dep_values = dependencies.values() if isinstance(dependencies, dict) else dependencies

                tpid = "[%d.%d]" % (i, j)

                load_from_cache = (process not in self.required_outputs  # always run specified vertices
                                   and process._cacheable
                                   and (not any(p.__class__.__name__ in computed for p in dep_values)
                                        or not self.force_downstream_rerun)
                                   and self.read_from_cache
                                   and process_cls_name in self.cache)
                if load_from_cache:
                    if not self.pbar:
                        self.logger.info("%s Will use cached output of %s"
                                         % (tpid, process_cls_name))
                    self.outputs[process] = self.cache[process_cls_name]
                else:
                    kwargs = dict()
                    if isinstance(dependencies, dict):
                        for k, v in dependencies.items():
                            if v not in self.outputs:
                                msg = "The process %s has a missing dependency:" % process_cls_name
                                msg += "%s=%s" % (k, v.__class__.__name__)
                                self.logger.error(msg=msg)
                                return -1
                            if isinstance(self.outputs[v], idt.RepoLeaf):
                                self.outputs[v] = self.outputs[v].load()
                            kwargs[k] = self.outputs[v]

                        computed.add(process_cls_name)
                        self.exec_process(process=process,
                                          proc_exec_kwargs=kwargs,
                                          tpid=tpid)
                    elif isinstance(dependencies, list):
                        proc_args = list()
                        for i, v in enumerate(dependencies):
                            if v not in self.outputs:
                                msg = "The process %s has a missing dependency:" % process_cls_name
                                msg += "%s=%s" % (k, v.__class__.__name__)
                                self.logger.error(msg=msg)
                                return -1
                            if isinstance(self.outputs[v], idt.RepoLeaf):
                                self.outputs[v] = self.outputs[v].load()
                            #kwargs[k] = self.outputs[v]
                            proc_args.append(self.outputs[v])

                        computed.add(process_cls_name)
                        self.exec_process(process=process,
                                          proc_args=proc_args,
                                          #proc_exec_kwargs=kwargs,
                                          tpid=tpid)
                if self.pbar:
                    self.tqdm_bar.update(1)

                future_deps = list()
                # Go through remaining processes at this level
                for p in ind_processes[j+1:]:
                    deps = self.dep_map.get(p, dict())
                    dep_iter = deps if isinstance(deps, list) else deps.values()
                    future_deps += list(d for d in dep_iter)

                # Go through future levels
                for topo in self.topology[i:]:
                    for p in topo:
                        deps = self.dep_map.get(p, dict())
                        vals = deps.values() if isinstance(deps, dict) else deps
                        future_deps += list(d for d in vals)
                future_deps = set(future_deps)
                self.logger.debug("%d Future deps" % len(future_deps))

                for k in self.outputs.keys():
                    if k not in future_deps and k not in self.required_outputs:
                        self.logger.debug("Deleting future dep %s" % k.__class__.__name__)
                        del self.outputs[k]

                # No reason to save the cached output back
                if self.write_to_cache and not load_from_cache and process._cacheable:
                    if not self.pbar:
                        self.logger.info("Persisting output of %s" % process_cls_name)
                    self.cache.save(self.outputs[process],
                                        name=process_cls_name,
                                        author='dag',
                                        param_names=list(kwargs.keys()),
                                        auto_overwrite=True)

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


    def viz(self, fontsize='10',
            color_mapping=None,
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
            if friendly_names:
                return m.get_name()
            else:
                return m.unique_cls_name

        for m, deps in merged.items():
            n = _get_name(m)

            node_attrs = m._get_viz_attrs()
            if color_mapping is not None and n in color_mapping:
                node_attrs['color'] = color_mapping[n]

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
            img_name = dot.render(cleanup=True,
                                  directory=tempfile.gettempdir())

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