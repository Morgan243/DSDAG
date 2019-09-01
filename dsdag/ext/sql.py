import string
from string import Formatter
#from dsdag.core.op import OpVertex, OpMeta
from dsdag.core.op import opvertex2 as opvertex
from dsdag.core.op import parameter, OpParent

def highlight_sql(q):
    from pygments import highlight
    from pygments.lexers import SqlLexer
    from pygments.formatters import HtmlFormatter
    formatter = HtmlFormatter(style='colorful')
    pygment_html = highlight(q, SqlLexer(), formatter)
    style_html = """
    <style>
    {pygments_css}
    </style>
    """.format(pygments_css=formatter.get_style_defs())

    html = style_html + pygment_html
    return html

@opvertex
class SQL:
    """
    Provides a 'q' parameter and a highlighted representation for notebook views.

    This Op by itself will attemp to populate string paramteres with arguments passed
    to the op during runtime. If wrap as is set, then the query will be wrapped in
    a sub query. Regardless, the resulting query string is returned at runtime.

    """
    q = parameter(default="""select * from something limit 100""")
    wrap_as = parameter(None, help_msg="Wrap the sql query in () as ")

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        if len(kwargs) > 0:
            q = self.q.format(**kwargs)
        else:
            q = self.q

        if self.wrap_as is not None:
            return "(%s) AS %s" % (q, self.wrap_as)
        else:
            return q

    def _repr_html_(self):
        q = self.q
        parse_args = string.Formatter().parse(q)
        q_params = [pa[1] for pa in parse_args
                    if pa[1] is not None]
        op_params = self.get_parameters()

        format_kwargs = {p: op_params[p]
                        if p in op_params and not isinstance(op_params[p], BaseParameter)
                            else '{%s}' % p
                        for p in q_params }

        if hasattr(self, 'run_kwargs'):
            format_kwargs.update(self.run_kwargs)

        q = q.format(**format_kwargs)

        return highlight_sql(q)

    def _node_color(self):
        return '#59ba5e'

    def op_nb_viz(self, op_out, viz_out=None):
        from pygments import highlight
        from pygments.lexers import SqlLexer
        from pygments.formatters import HtmlFormatter
        from IPython.display import HTML
        import ipywidgets as widgets

        if viz_out is None:
            viz_out = widgets.Output(layout=widgets.Layout(width = '45%'))


        formatter = HtmlFormatter(style='colorful')

        pygment_html = highlight(op_out, SqlLexer(), formatter)
        style_html = """
        <style>
        {pygments_css}
        </style>
        """.format(pygments_css=formatter.get_style_defs())

        html = style_html + pygment_html
        viz_out.append_display_data(HTML(html))
        return viz_out

@opvertex
class SQL_WrapAs:
    """
    Wraps a query string as a sub-query
    """
    alias = parameter(None, help_msg="Wrap the sql query in () as <alias>")

    def run(self, q):
        return "(%s) AS %s" % (q, self.alias)

    def op_nb_viz(self, op_out, viz_out=None):
        from IPython.display import HTML
        import ipywidgets as widgets
        if viz_out is None:
            viz_out = widgets.Output(layout=widgets.Layout(width = '45%'))

        html = highlight_sql(op_out)
        viz_out.append_display_data(HTML(html))
        return viz_out

    def _node_color(self):
        return '#59ba5e'

@opvertex
class SQL_ParamMixin:
    """
    Contains many baseline parameters useful for query Ops, including clauses, key_map, and join_keys.

    Other than parameters, this Op provides a helper method to collect all parameters into a dictionary,
    useful when all parameters need to be passed to a required SQL param Op from a non-SQL param op.
    """
    key_map = parameter(None,
                            help_msg="Map select alias to select source (clnt_ref_id->'CL.clnt_ref_id/10')")

    ### CLAUSES
    # Raw SQL text to inserted into the query
    select_clause = parameter(None,
                                  help_msg="Select statement to include after SELECT, but before FROM ")
    join_clause = parameter(None,
                                help_msg="Join statement to include after FROM but before WHERE")
    where_clause = parameter(None,
                                 help_msg="SQL filter to be appended to the WHERE clause")
    groupby_clause = parameter(None,
                                 help_msg="SQL filter to be appended to the WHERE clause")
    ###

    join_ops = parameter(None,
                             "Pass a SQL Op or mapping of SQL Ops to be joined. These will"
                             "specified as requirements by this OP and will be passed as named_joined_queries"
                             "at runtime.")
    join_keys = parameter(None)
    _join_template = """
    JOIN
        {sample_filter_query}
    ON
        {join_condition}
    """
    extra_q_kwargs = parameter(None, "Mapping of additional format parameters that need to "
                                         "be specified in the query (e.g. snapshot date, code)")

    def make_sql_param_kwargs(self):
        return dict(select_clause=self.select_clause,
                    key_map=self.key_map,
                    join_clause=self.join_clause,
                    join_keys=self.join_keys,
                    where_filter=self.where_clause,
                    groupby_add=self.groupby_clause,
                    join_ops=self.join_ops)


@opvertex
class SQL_Param(SQL, SQL_ParamMixin):
    # not valid by default - but gets the point across
    #q = BaseParameter(default="""select * {select_clause} from something limit
#{join_clause} 100 WHERE {where_filt}""")

    def requires(self):
        if self.join_ops is None:
            return dict()
        elif isinstance(self.join_ops, dict):
            # A dictionary mapping join names to ops - just return it as requirements
            if not all(isinstance(o, (SQL, SQL_WrapAs)) for o in self.join_ops.values()):
                raise ValueError("All join ops must be SQL derived")
            return self.join_ops
        elif issubclass(type(self.join_ops), (OpParent, SQL)):
            # No name given, so just make one up ('param_subq')
            return dict(param_subq = self.join_ops)
        else:
            raise ValueError("Don't understand join ops parameter: %s" % str(type(self.join_ops)))

    def _join_statement_from_param_clause(self, q_name, q):
        jc = self._join_clause.format(sample_filter_name=q_name,
                                      sample_filter_query="%s" % q)
        if any(fname == 'sample_filter_name' for _, fname, _, _ in Formatter().parse(jc)):
            jc = jc.format(sample_filter_name=q_name)
        return jc

    def _join_statement_from_template_and_map(self, q_name, q, join_keys=None):
        if join_keys is None:
            join_keys = list(map(lambda s: s.lower(), self._join_key_map.keys()))

        join_conds = "\nAND ".join((v.format(sample_filter_name=q_name)
                                    for k, v in self._join_key_map.items()
                                                if k.lower() in join_keys))
        jc = self._join_template.format(sample_filter_query=q,
                                        join_condition=join_conds)
        if any(fname == 'sample_filter_name' for _, fname, _, _ in Formatter().parse(jc)):
            jc = jc.format(sample_filter_name=q_name)

        return jc

    def run(self, *join_queries, **named_join_queries):
        # Default to empty string for clause additions
        frmt_params = dict(select_clause = "" if self.select_clause is None else self.select_clause,
                            where_clause = "" if self.where_clause is None else self.where_clause,
                            join_clause = "" if self.join_clause is None else self.join_clause,
                            groupby_clause = "" if self.groupby_clause is None else self.groupby_clause)

        # New way of doing things
        use_join_map = hasattr(self, '_join_template') and hasattr(self, '_join_key_map')
        # Old way of doing things
        use_join_clause = hasattr(self, '_join_clause')

        #join_keys = named_join_queries.pop('join_keys', None)

        if self.extra_q_kwargs is not None:
            if not isinstance(self.extra_q_kwargs, dict):
                raise ValueError("Parameter extra_q_kwargs for %s must be a dictionary" % self.__class__.__name__)
            frmt_params.update(self.extra_q_kwargs)

        ####
        if len(named_join_queries) > 0:
            if use_join_map:
                join_statements = [self._join_statement_from_template_and_map(q_name, q, join_keys=self.join_keys)
                                   for q_name, q in named_join_queries.items()]
            elif use_join_clause:
                join_statements = [self._join_statement_from_param_clause(q_name, q)
                                    for q_name, q in named_join_queries.items()]
            else:
                msg = "Passed requirements to %s requires a _join_clause or _join_key_map static attr" % str(self)
                raise ValueError(msg)

            frmt_params['join_clause'] += "\n\n" + "\n\n".join(join_statements)

        if len(join_queries) > 0:
            if use_join_map:
                raise ValueError("Join map for unamed joins?")
            elif use_join_clause:
                join_statements = [self._join_clause.format(sample_filter_name="",
                                                            sample_filter_query=q)
                                    for q in join_queries]
            else:
                raise ValueError("Pass requirements to %s requires a _join_clause static attr" % str(self))

            frmt_params['join_clause'] += "\n\n" + "\n\n".join(join_statements)

        fieldnames = [fname for _, fname, _, _ in Formatter().parse(self.q) if fname]
        for f in fieldnames:
            if f not in frmt_params:
                if not hasattr(self, f):
                    raise ValueError("Cannot find field %s" % f)
                else:
                    frmt_params[f] = getattr(self, f)

        try:
            frmt_kwargs = {fname: frmt_params[fname] for fname in fieldnames}
        except KeyError as e:
            print(fieldnames)
            print(self.q)
            raise


        q = self.q.format(**frmt_kwargs)

        if self.wrap_as is not None:
            q = "(%s) AS %s" % (q, self.wrap_as)

        return q

#############

class SQL_CountSamples(SQL):
    q = """select count(*) as n_samples from ({subq}) as sq"""

    def run(self, subq):
        return self.q.format(subq=subq)


class SQL_RandomSample(SQL):
    q = """select * from ({subq}) as {alias} where random() < {rate} {where_addon}"""
    alias = parameter("sq", help_msg="Alias for sub query")
    rate = parameter(.1)
    where_addon = parameter(None)
    limit = parameter(None)

    def run(self, subq):

        if self.where_addon is None:
            self.where_addon = ""

        if self.limit is not None:
            self.where_addon = " limit %s" % str(self.limit)

        return self.q.format(subq=subq, rate=self.rate,
                             where_addon=self.where_addon,
                             alias=self.alias)
                             #name=self.get_name())

class SQL_WrapQuery(SQL):
    q = """select {select_clause} from ({subq}) as {subq_alias} {where_addon}"""
    select_clause = parameter('*', help_msg='Select string')
    where_addon = parameter(None, help_msg='Where clause (inlucde WHERE keyword)')
    subq_alias = parameter('sq', help_msg='What to name the sub query (as <name>)')

    def run(self, subq):
        where_add = self.where_addon if self.where_addon is not None else ''

        q = self.q.format(select_clause=self.select_clause,
                          subq=subq, subq_alias=self.subq_alias,
                          where_addon=where_add)
        return q


class SQL_SelectLiterals(SQL):
    q = """VALUES {value_tuples} as {subq_name} ({field_names})"""
    value_tuples = parameter(help_msg="List of tuples of values, each tuple should have the same size and a column types")
    field_names = parameter(help_msg="List of field names, should be same length as tuple in value_tuples parameter")
    subq_name = parameter('subq', help_msg="(Optional) Alias for the resulting table selection")
    fields_to_quote = parameter(None, help_msg="(Optional) Field names that are strings that should be quoted (e.g. plcy_nbr_id)")

    def run(self):
        fn_map = {i:fn for i, fn in enumerate(self.field_names)}

        vt = ["(%s)" % ",".join("'%s'" % str(_v) if fn_map[i] in self.fields_to_quote else str(_v)
                                for i, _v in enumerate(v))
              for v in self.value_tuples]
        vt = ", ".join(vt)
        fn = ", ".join(str(f) for f in self.field_names)

        return self.q.format(value_tuples=vt, field_names=fn, subq_name=self.subq_name)

class SQL_SelectFromDataFrame(SQL):
    q = """(VALUES {value_tuples}) as {subq_name} ({field_names})"""
    #df = BaseParameter(help_msg="DataFrame to be selecte from - column names will be the field name")
    #df = UnhashableParameter(help_msg="DataFrame to be selecte from - column names will be the field name")
    subq_name = parameter(None, help_msg="(Optional) Alias for the resulting table selection")

    def run(self, *args, **kwargs):
        all_params = list(args) + list(kwargs.values())
        if len(all_params) != 1:
            msg = ("%s expects only one input, but %d given (reminder: dataframe is not a c-tor param)"
                   % (str(self), len(all_params)))
            raise ValueError(msg)
        else:
            self.df = all_params[0]

        fn_map = {i:fn for i, fn in enumerate(self.df.columns)}

        quote_fields = self.df.columns[self.df.dtypes == 'object'].values

        # Take each row and turn it into a comma seperated literal, quote those that need it
        func = lambda r: "(%s)" % ", ".join((("'%s'" % str(v)) if fn_map[i] in quote_fields else str(v))
                                            for i, v in enumerate(r))
        # Tuplify using func
        vt = self.df.apply(func, axis=1)

        # Tuples together our tuples...
        vt = ", ".join(vt)
        fn = ", ".join(str(f) for f in self.df.columns)

        return self.q.format(value_tuples=vt, field_names=fn,
                             # if a name wasn't provided, it will need to be
                             # This param is left for use later so that the same name can be used
                             # elsewhere (e.g. in a join statement)
                             subq_name=self.subq_name if self.subq_name is not None else "{sample_filter_name}")


