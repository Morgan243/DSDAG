from dsdag.core.op import OpVertex
from dsdag.core.parameter import BaseParameter, RepoTreeParameter

class Passthrough(OpVertex):
    _never_cache = True
    def _node_color(self):
        return '#eded5c'

    def _node_shape(self):
        return 'cylinder'

    def run(self, o):
        return o

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
