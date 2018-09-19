#import luigi
from dsdag.core.op import OpVertex
from dsdag.core.parameter import BaseParameter, RepoTreeParameter

import gnwdata as gd
class Passthrough(OpVertex):
    _never_cache = True
    def _node_color(self):
        return '#eded5c'

    def _node_shape(self):
        return 'cylinder'

    def run(self, o):
        return o


class BaseDataProvider(OpVertex):
    #param = luigi.Parameter()
    #system = 'greenplum'
    system = BaseParameter('greenplum')

    system_kwargs = dict()
    existing_connections = dict()
    system_map = dict(greenplum=gd.greenplum,
                      oracle=gd.oracle)

    def _node_color(self):
        return '#eded5c'

    def _node_shape(self):
        return 'cylinder'

    def requires(self):
        raise NotImplementedError("Incomplete Base DataProvider class - implement requires")

    def set_provider(self):
        kw_str = "".join("%s=%s" % (str(k), str(self.system_kwargs[k]))
                         for k in sorted(self.system_kwargs.keys()))

        # Some systems (oracle) need a schema parameter, so not
        # all connections are going to be the same
        cxn_exists = ((self.system in self.existing_connections)
                       and kw_str in self.existing_connections[self.system])
        if not cxn_exists:
            cxn = self.system_map[self.system](**self.system_kwargs)
            self.existing_connections[self.system] = {kw_str:cxn}

        self.provider = self.existing_connections[self.system][kw_str]
        return self.provider

    def run(self, q):
        self.set_provider()
        return self.provider.query(q)

    def output(self):
        pass
        #return self.


class DataTreeLoader(OpVertex):
    rt_leaf = RepoTreeParameter()
    _never_cache = True


    def _node_color(self):
        return '#ed9d5c'

    def _node_shape(self):
        return 'cylinder'

    def run(self):
        return self.rt_leaf.load()
        #import interactive_data_tree as idt
        #if isinstance(self.rt_leaf, (list, tuple)):
        #    if any(not isinstance(rl, (idt.RepoLeaf)) for rl in self.rt_leaf):
        #        raise ValueError("One of the objects provided in the set is not an idt.RepoLeaf")
        #    return [l.load() for l in self.rt_leaf]
        #elif isinstance(self.rt_leaf, (idt.RepoLeaf)):
        #    return self.rt_leaf.load()
        #else:
        #    msg = ("%s can't handle rt_leaf parameter of type %s"
        #           % (self.__class__.__name__, type(self.rt_leaf)))
        #    raise ValueError(msg)
