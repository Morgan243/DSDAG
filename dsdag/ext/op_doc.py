#from dsdag.core.op import OpVertex
from dsdag.core.op import OpParent
from types import ModuleType
import inspect

def find_ops(_o, max_depth=3, depth=0,
             types=(OpParent),
             filter_func=None,
             keys_as_string=False,
             module_white_list=None):
    if inspect.ismodule(_o) and depth < max_depth:
        # Get Classes that Originate in this Module
        _names = [n for n, cls_o in inspect.getmembers(_o, inspect.isclass)
                  if cls_o.__module__ == _o.__name__]
        # Get Modules that are imported in this module to search next
        _names += [n for n, cls_o in inspect.getmembers(_o, inspect.ismodule)
                    if module_white_list is None or (module_white_list is not None and cls_o in module_white_list)]

        to_ret = dict()
        for _n in _names:
            rec_o = _o.__dict__[_n]
            t = find_ops(rec_o, depth=depth + 1, max_depth=max_depth,
                         types=types, filter_func=filter_func,
                         keys_as_string=keys_as_string,
                         module_white_list=module_white_list)
            # Only keep if there was something relevant down this path
            if t is not None:
                to_ret[rec_o if not keys_as_string else _n] = t

        return to_ret if len(to_ret) > 0 else None

    # If we find an Op - return it!
    elif inspect.isclass(_o) and issubclass(_o, types):
        if filter_func is not None and filter_func(_o):
            return _o
        elif filter_func is None:
            return _o
        else:
            return None
    else:
        return None


class auto_doc(object):
    jinja_template_html = """
<h3>{{ op_name }}</h3>
<i>{{ op_docs }}</i>

<table border="1">
{% for row in list_items %}
<tr>{% for n in row %}
<td>{{ n }}</td>
{% endfor %}
</tr>
{% endfor %}
</table>
"""

    jinja_template_markdown = """
<h3>{{ op_name }}</h3>

<p>{{ op_docs }}</p>

{% if list_items|length > 1 %}
{% for row in list_items %}
| {{ row|join(" | ")}} |{% if loop.index == 1 %}
| - | - |{% endif %}{% endfor %}
{% else %}
<i>No Features provided and no parameters exposed</i>
{% endif %}

{{ append_txt }}
"""

    @staticmethod
    def combine_items(*args):
        max_len = max(map(len, args))
        _args = [list(a) + ([''] * (max_len - len(a))) for a in args]
        return zip(*_args)

    @staticmethod
    def make_jinja_op_doc(op, template):
        f = getattr(op, 'features_provided', [])
        f = f if isinstance(f, list) else []

        p = op.__ordered_params__
        list_items = auto_doc.combine_items(
            ['<b>Features Provided</b>'] + sorted(f),
            ['<b>Op Parameters</b>'] + sorted(p)
        )
        ######
        op_docs = getattr(op, '__original_doc__', op.__doc__)
        op_docs = 'No Doc String!' if op_docs is None else op_docs
        ######
        import inspect
        requires_code = inspect.getsource(op.requires)
        requires_code = """
```python
%s
```""" % requires_code

        doc = template.render(op_name=op.__module__ + '.' + op.__name__,
                              op_docs=op_docs,
                              list_items=list_items,
                              append_txt=requires_code)
        return doc

    @staticmethod
    def make_jinja_docs(_o,
                        template,
                        top_level_order=None,
                        prefix_html=None,
                        postfix_html=None,
                        depth=0):
        if isinstance(_o, dict):
            sub_docs = list()
            if top_level_order is None or depth != 0:
                top_level_order = sorted(_o.keys())

            for k in top_level_order:
                new_o = _o[k]
                if new_o is None:
                    continue

                d = auto_doc.make_jinja_docs(new_o,
                                             template=template,
                                             top_level_order=top_level_order,
                                             depth=depth + 1)

                #if depth == 0:
                if inspect.ismodule(k):
                    op_docs = k.__doc__ if k.__doc__ is not None else 'No Doc String'
                    d = ("<h2><u>%s</u></h2>" % k.__name__) + ("<i><b>%s</b></i>" % op_docs) + d + '\n<hr>'

                sub_docs.append(d)

            sub_docs = ([prefix_html if prefix_html is not None else '']
                        + sub_docs +
                        [postfix_html if postfix_html is not None else ''])
            return "\n".join(sub_docs)

        elif isinstance(_o, OpParent) or (inspect.isclass(_o) and issubclass(_o, OpVertex)):
            doc = auto_doc.make_jinja_op_doc(_o, template=template)
            return doc
        else:
            print("dont know %s" % str(_o))

from IPython.display import Markdown, display
from jinja2 import Template
def document_module(module_o, prefix_html=None, postfix_html=None,
                    jinja_template=auto_doc.jinja_template_markdown,
                    top_level_order=None, module_white_list=None,
                    display_wrapper=Markdown, return_raw_str=False):
    if not isinstance(jinja_template, Template):
        jinja_template = Template(jinja_template)

    op_map = find_ops(module_o, module_white_list=module_white_list)

    doc_str = auto_doc.make_jinja_docs(op_map, template=jinja_template,
                                       prefix_html=prefix_html, postfix_html=postfix_html,
                                       top_level_order=top_level_order)
    if display_wrapper is not None:
        display(display_wrapper(doc_str))

    if return_raw_str:
        return doc_str



def flatten_op_map(d):
    if isinstance(d, dict):
        return sum((flatten_op_map(v) for v in d.values()), list())
    return [(d.__module__ + '.' + d.__name__, d)]


def extract_op_text_map(op):
    fp = getattr(op, 'features_provided', [''])
    fp = fp if isinstance(fp, list) else [fp]
    fp = map(str, fp)

    return dict(name=op.__name__,
                doc=getattr(op, '__original_doc__', '__doc__'),
                features=" ".join(fp))

def search_ops(ops, q_str, case_sensitive=False, return_all=False):
    import pandas as pd

    if inspect.ismodule(ops):
        ops = dict(flatten_op_map(find_ops(ops))).values()
    elif not isinstance(ops, list):
        raise ValueError("Ops parameter should be alist of ops or a module")

    q_str = q_str if case_sensitive else q_str.lower()
    results = dict()

    for _o in ops:
        txt_map = extract_op_text_map(_o)
        r = {k: (q_str in (v if case_sensitive else v.lower())) if v is not None else False
             for k, v in txt_map.items()}
        # results[_o.__module__ + '.' + _o.__name__] = r
        results[_o] = r

    r_df = pd.DataFrame(results).T
    # Higher is better
    r_df['match_score'] = r_df.sum(axis=1).sort_values(ascending=False)
    ret = r_df.sort_values('match_score', ascending=False)

    if not return_all:
        ret = ret[ret['match_score'] > 0]

    return ret