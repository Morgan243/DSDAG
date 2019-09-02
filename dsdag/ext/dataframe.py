import pandas as pd
from dsdag.core.op import opvertex as opvertex
from dsdag.core.op import parameter as parameter

@opvertex
class DFOp:
    def _node_color(self):
        return '#615eff'

@opvertex
class Read_CSV:
    filepath_or_buffer = parameter()
    sep = parameter(',')
    delimiter = parameter(None)
    header = parameter('infer')
    def run(self, path=None):
        if path is None:
            path = self.filepath_or_buffer
        return pd.read_csv(filepath_or_buffer=path,
                           sep=self.sep, delimiter=self.delimiter,
                           header=self.header)

@opvertex
class Merge:
    #key = parameter()
    how = parameter('inner')

    on = parameter(None)
    left_on = parameter(None)
    right_on = parameter(None)
    left_index = parameter(False)
    right_index = parameter(False)
    sort = parameter(False)
    suffixes = parameter(('_x', '_y'))
    copy = parameter(True)
    indicator = parameter(False)
    validate = parameter(None)

    def requires(self):
        raise NotImplementedError()

    def run(self, *args):
        #frames = list(kwargs.values())
        frames = list(args)
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on=self.on, how=self.how,
                                  left_on=self.left_on, right_on=self.right_on,
                                  left_index=self.left_index, right_index=self.right_index,
                                  sort=self.sort, suffixes=self.suffixes, copy=self.copy,
                                  indicator=self.indicator, validate=self.validate)

        return merged

@opvertex
class Concat:
    axis = parameter(0)
    join = parameter('outer')
    ignore_index = parameter(False)
    keys = parameter(None)
    levels = parameter(None)
    verify_integrity = parameter(False)
    sort = parameter(None)
    copy = parameter(True)

    def requires(self):
        raise NotImplementedError()

    def run(self, *args):
        return pd.concat(args, axis=self.axis, join=self.join,
                         ignore_index=self.ignore_index,
                         keys=self.keys, levels=self.levels,
                         verify_integrity=self.verify_integrity,
                         sort=self.sort, copy=self.copy)

class Join(DFOp):
    how = parameter('left')
    lsuffix = parameter('')
    rsuffix = parameter('')
    sort = parameter(False)

    def requires(self):
        raise NotImplementedError()

    def run(self, *args):
        assert all(isinstance(o, pd.DataFrame) for o in args)
        ret = args[0]
        for o in args[1:]:
            ret = ret.join(o, how=self.how, lsuffix=self.lsuffix,
                           rsuffix=self.rsuffix, sort=self.sort)
        return ret

class Drop(DFOp):
    labels = parameter()
    axis = parameter(0)

    def run(self, df):
        return df.drop(labels=self.labels, axis=self.axis)

class Query(DFOp):
    q = parameter()
    def run(self, df):
        return df.query(self.q)

class AssignColumn(DFOp):
    column = parameter(None)
    value = parameter(None)
    assignments = parameter(None)
    def run(self, df):
        if self.assignments is not None:
            for col, val in self.assignments.items():
                df[col] = val
        else:
            df[self.column] = self.value
        return df

class RenameColumns(DFOp):
    columns = parameter(None)
    copy = parameter(True)
    inplace = parameter(False)
    level = parameter(None)

    def run(self, df):
        return  df.rename(columns=self.columns, copy=self.copy,
                          inplace=self.inplace, level=self.level)

class ApplyMap(DFOp):
    func = parameter()

    def run(self, df):
        return df.applymap(func=self.func)

class DropDuplicates(DFOp):
    subset = parameter(None)
    keep = parameter('first')
    inplace = parameter(False)

    def run(self, df):
        return df.drop_duplicates(subset=self.subset, keep=self.keep,
                                  inplace=self.inplace)

class SelectColumns(DFOp):
    columns = parameter(None)
    def run(self, df):
        return df[self.columns]

class DropNa(DFOp):
    axis=parameter(0)
    how=parameter('any')
    thresh=parameter(None)
    subset=parameter(None)

    def run(self, df):
        return df.dropna(axis=self.axis, how=self.how, thresh=self.thresh,
                         subset=self.subset, inplace=False)

######
import threading
from IPython.display import display, clear_output
import time
import matplotlib

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


class FrameBrowseMixin():

    @staticmethod
    def build_df_browse_widget(df, orientation='horizontal'):
        def closure(ix, columns=None):
            display(df.iloc[ix:ix + 5])

        ix_slider = widgets.IntSlider(min=0, max=len(df),
                                      step=1, value=0,
                                      orientation=orientation,
                                      description='Start row')
        sub_widgets = dict(ix=ix_slider)

        df_browse_out = widgets.interactive_output(closure, sub_widgets);
        df_browse_widget = widgets.VBox([ix_slider, df_browse_out])
        return df_browse_widget

    @staticmethod
    def build_df_rdt_widget(df):
        out = widgets.Output()

        def btn_click(b):
            out.clear_output()
            desc = df.describe().T
            out.append_display_data(desc)

        btn = widgets.Button(description="Compute RDT")
        btn.on_click(btn_click)

        box = widgets.VBox([btn, out])

        return box

    @staticmethod
    def build_df_sample_widget(df, orientation='horizontal'):

        int_txt = widgets.BoundedIntText(
            value=5,
            min=1,
            max=50,
            step=1,
            description='Sample size:',
            disabled=False
        )
        sample_out = widgets.Output()

        def closure(n, columns=None):
            sample_out.clear_output()
            sample_out.append_display_data(df.sample(n))

        def btn_click(b):
            sample_out.clear_output()
            sample_out.append_display_data(df.sample(int_txt.value))

        button = widgets.Button(description='Sample')
        button.on_click(btn_click)

        df_sample_widget = widgets.VBox([widgets.HBox([int_txt, button]),
                                         sample_out])
        return df_sample_widget

    @staticmethod
    def build_df_query_widget(df):
        out = widgets.Output()
        txt_q = widgets.Text(placeholder="Write a valid query for df.query()")

        def btn_click(b):
            out.clear_output()
            res = df.query(txt_q.value).head()
            out.append_display_data(res)

        btn = widgets.Button(description="Query")
        btn.on_click(btn_click)

        box = widgets.VBox([widgets.HBox([btn, txt_q]), out])

        return box

    @staticmethod
    def build_df_univariate_widget(df):
        out = widgets.Output()
        # txt_q = widgets.Text(placeholder="Write a valid query for df.query()")

        features = list(df.columns)
        dropdown = widgets.Dropdown(options=features,
                                    value=features[0],
                                    description='Feature:',
                                    disabled=False)

        def btn_click(b):
            out.clear_output()
            s = df[dropdown.value]
            entropy = 0
            out.append_display_data(entropy)

            matplotlib.pyplot.ioff()
            fig, ax = matplotlib.pyplot.subplots()
            ax.clear()
            ax = s.hist(bins=40, ax=ax)
            out.append_display_data(ax.figure)

            matplotlib.pyplot.ion()

        btn = widgets.Button(description="GO")
        btn.on_click(btn_click)

        box = widgets.VBox([widgets.HBox([dropdown, btn]), out])

        return box

    @staticmethod
    def op_nb_viz(op_out, viz_out=None):
        df = op_out
        if viz_out is None:
            viz_out = widgets.Output(layout=widgets.Layout(width = '45%'))

        tab_child_widgets = list()
        tab_titles = list()

        # DF Browse
        tab_child_widgets.append(FrameBrowseMixin.build_df_browse_widget(df))
        tab_titles.append('Browse')

        # RDT
        tab_child_widgets.append(FrameBrowseMixin.build_df_rdt_widget(df))
        tab_titles.append('RDT')

        # Sample
        tab_child_widgets.append(FrameBrowseMixin.build_df_sample_widget(df))
        tab_titles.append('Sample')

        # Query
        tab_child_widgets.append(FrameBrowseMixin.build_df_query_widget(df))
        tab_titles.append('Query')

        # Univariate Analysis
        tab_child_widgets.append(FrameBrowseMixin.build_df_univariate_widget(df))
        tab_titles.append('Univariate')

        tab = widgets.Tab()

        tab.children = tab_child_widgets
        for i, c in enumerate(tab_child_widgets):
            tab.set_title(i, tab_titles[i])

        viz_out.append_display_data(tab)
        return viz_out

        #display(tab)

        #if self.passthrough:
         #   return df


class FrameBrowse(DFOp, FrameBrowseMixin):
    passthrough = parameter(True)

    def op_nb_viz(self, op_out, viz_out=None):
        return FrameBrowseMixin.op_nb_viz(op_out, viz_out)

    def run(self, df):
        output = self.op_nb_viz(df)
        display(output)
        if self.passthrough:
            return df
