import uuid
from dsdag.core.op import OpVertex
from dsdag.core.parameter import BaseParameter, DatetimeParameter, UnhashableParameter, RepoTreeParameter
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn import pipeline as pl
from imblearn import over_sampling as os

from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV

from multiprocessing import Manager
from multiprocessing import Pool

from tqdm import tqdm

class BaseWrangler(OpVertex):
    def requires(self):
        return dict()

    def get_input_features(self):
        return list(sum(self.get_input_feature_mapping().values(), list()))

    def get_input_feature_mapping(self):
        inputs = self.get_input_ops()
        feature_map = {in_name: in_o.features_provided
                       for in_name, in_o in inputs.items()
                        if hasattr(in_o, 'features_provided')}
        return feature_map


from sklearn import tree
import pydot
import collections
class DT_Explain():
    @staticmethod
    def visualize_tree(dt, features, path=None,
                       image_output_path=None):
        from IPython.display import Image
        # Visualize data
        dot_data = tree.export_graphviz(dt,
                                        feature_names=features,
                                        out_file=None,
                                        filled=True,
                                        rounded=True, node_ids=True)
        graph = pydot.graph_from_dot_data(dot_data)[0]

        colors = ('blue', 'orange')
        edges = collections.defaultdict(list)

        for edge in graph.get_edge_list():
            edges[edge.get_source()].append(int(edge.get_destination()))

        for edge in edges:
            # coloring parent nodes
            if path is not None:
                if int(edge) in path:
                    graph.get_node(edge)[0].set_fillcolor('red')
                else:
                    graph.get_node(edge)[0].set_fillcolor('white')

            # finding all children, step needed to color final nodes in the tree
            for i in range(2):
                child = str(edges[edge][i])
                if path is not None:
                    if int(child) in path:
                        graph.get_node(child)[0].set_fillcolor('red')
                    else:
                        graph.get_node(child)[0].set_fillcolor('white')

        output_path = 'tree.png' if image_output_path is None else image_output_path
        graph.write_png(output_path)
        return Image(output_path)

    @staticmethod
    def rev_path(tr, node_id):
        matching = [(i, 'left') if l == node_id else (i, 'right')
                    for i, (l, r) in enumerate(zip(tr.children_left, tr.children_right))
                    if l == node_id or r == node_id]
        if node_id == 0:
            return []

        assert len(matching) == 1
        parent_pair = matching[0]

        return DT_Explain.rev_path(tr, parent_pair[0]) + [parent_pair]

    @staticmethod
    def print_node_path(tr, node_id, features,
                        rescale=None, transform_map=None):
        if transform_map is None:
            transform_map = dict()

        t_p = DT_Explain.rev_path(tr, node_id)
        t_p.append((node_id, None))

        path_data = list()
        for nid, direction in t_p:
            f = tr.feature[nid]
            ineq_str = dict(left='<=', right='>').get(direction, None)

            if tr._tree.TREE_UNDEFINED == f:
                fname = None
                thresh = None
                mtype = None
                orig_thresh = None
            else:
                fname = features[f]
                thresh = tr.threshold[nid]

                mtype = transform_map.get(fname, 'unknown')

                orig_thresh = thresh
                if rescale is not None:
                    _t = rescale(fname, thresh, transform_type=mtype)
                    thresh = _t if _t is not None else thresh

            _d = dict(node_id=nid, treatment=mtype, feature_name=fname, inequality=ineq_str, threshold=thresh,
                      orig_thresh=orig_thresh)
            # Left means '<='
            if ineq_str is not None:
                split_str = "[%d] (%s) %s %s %.3f" % (nid, mtype, fname, ineq_str, thresh)
                if orig_thresh != thresh:
                    split_str += ' (with scaling = %.3f)' % orig_thresh
                vals = tr.value[nid][0]
                _d['class_0_n'] = vals[0]
                _d['class_1_n'] = vals[1]
            else:
                vals = tr.value[node_id][0]
                _d['class_0_n'] = vals[0]
                _d['class_1_n'] = vals[1]

                _d['inequality'] = '<=' if vals[1] > vals[0] else '>'

                class_counts = "%d, %d" % (vals[0], vals[1])
                samples = sum(vals)
                imp = tr.impurity[node_id]
                str_params = dict(n_samples=samples, class_counts=class_counts, impurity=imp)
                scoring_str = "- N Samples: {n_samples}\n- Class Counts: {class_counts}\n- Impurity: {impurity}"

                split_str = "[%d]" % nid
                if fname is None:
                    split_str += ' Leaf'
                split_str += "\n" + scoring_str.format(**str_params)
            path_data.append(_d)
            print(split_str)
        path_df = pd.DataFrame(path_data)
        path_df.index.name = 'path_step'

        path_df['n_samples'] = path_df['class_0_n'] + path_df['class_1_n']
        return path_df

    @staticmethod
    def get_node_mask(df, tr, node_id,
                      features=None,
                      rescale=None, transform_map=None,
                      **ufuncs):
        if transform_map is None:
            transform_map = dict()

        if features is None:
            features = df.columns.tolist()

        t_p = DT_Explain.rev_path(tr, node_id)
        t_p.append((node_id, None))

        # Default mask is all True
        mask = pd.Series(True, df.index)

        mask_iter_res = list()
        for nid, direction in t_p:
            f = tr.feature[nid]
            fname = features[f]
            thresh = tr.threshold[nid]
            mtype = transform_map.get(fname, 'extracted')

            vals = tr.value[nid][0]
            is_leaf = True if (tr.children_left[nid] < 0) and (tr.children_right[nid] < 0) else False

            if rescale is not None and not is_leaf:
                rescaled_thresh = rescale(fname, thresh, transform_type=mtype)
            else:
                rescaled_thresh = None

            if direction is None and not is_leaf:
                # If here, it means we are on a selected node (direction == None)
                #  so we assume the node's True filter is applied (True is always left in Sklearn DT)
                direction = 'left'
            elif is_leaf:
                direction = None

            if direction == 'left':
                mask &= df[fname] <= thresh
                ineq_str = '<='
            elif direction == 'right':
                mask &= df[fname] > thresh
                ineq_str = '>'
            else:
                ineq_str = 'leaf'

            _d = dict(node_id=nid,
                      feature_name=fname if not is_leaf else None,
                      threshold=thresh if not is_leaf else None,
                      threshold_before_treatment=rescaled_thresh,
                      inequality=ineq_str,
                      treatment_type=mtype if not is_leaf else None,
                      n_selected=mask.sum(),
                      portion_selected=mask.mean())
            _d.update({k: func(df[mask]) for k, func in ufuncs.items()})
            mask_iter_res.append(_d)

        return mask, pd.DataFrame(mask_iter_res).set_index('node_id')

    ###-----
    @staticmethod
    def target_sample_ratios_from_tree(mtree):
        node_proportion_s = pd.Series(mtree.tree_.value[:, 0, 1] / mtree.tree_.value[:, 0, 0]).sort_values(ascending=False)
        node_proportion_s.index.name = 'node_id'
        return node_proportion_s

    @staticmethod
    def inspect_dt_model(feat_df, model_leaf, test_top_n=7):
        import ipywidgets as widgets
        from IPython.display import display
        tree_feats = model_leaf.md['features']
        mtree = model_leaf.load()

        node_proportion_s = DT_Explain.target_sample_ratios_from_tree(mtree)

        nid_to_mask_map = {_nid: DT_Explain.get_node_mask(feat_df, mtree.tree_, _nid, features=tree_feats)[0]
                           for _nid in node_proportion_s.head(test_top_n).index}

        test_rate_df = pd.DataFrame([dict(node_id=_nid, n_samples=m.sum(), sample_proportion=m.mean(),
                                          target_rate=feat_df[m]['is_opportunity'].mean())
                                     for _nid, m in nid_to_mask_map.items()]).set_index('node_id')
        test_rate_df.sort_values(['target_rate', 'n_samples'], inplace=True, ascending=False)

        tree_viz = DT_Explain.visualize_tree(mtree, tree_feats)

        # return widgets.HBox([best_node_out, tree_viz_out, df_out])
        # return best_node_out, tree_viz_out, df_out
        # return tree_viz

        df_out = widgets.Output(layout=widgets.Layout(width='1500px'))
        tree_out = widgets.Output(layout=widgets.Layout(width='3700px'))

        df_out.append_display_data(test_rate_df.round(4))
        param_str = "\n".join("%s=%s" % (k, v) for k, v in mtree.get_params().items()
                              if v is not None and k not in ('presort', 'splitter',
                                                             'min_imprutity_decrease',
                                                             'min_weight_fraction_leaf',
                                                             'min_impurity_decrease'))
        with df_out:
            # display(param_str)
            print(param_str)
        # df_out.append_display_data(param_str)

        tree_out.append_display_data(tree_viz)
        display(widgets.HBox([df_out, tree_out]))
        # display(tree_viz)
        # display(test_rate_df)

    @staticmethod
    def inspect_dt_node(feat_df, model_leaf, node_id, rescale=None, transform_map=None,
                        **ufuncs):
        tree_feats = model_leaf.md['features']
        mtree = model_leaf.load()

        mask, mask_df = DT_Explain.get_node_mask(feat_df, mtree.tree_, node_id,
                                                 features=tree_feats, rescale=rescale,
                                                 transform_map=transform_map,
                                                 **ufuncs)
                                                 #target_rate=lambda _df: _df['is_opportunity'].mean())
        # print("Computed mask DF %d" % node_sel_w.value)
        col_ordering = ['feature_name',
                        'inequality',
                        'threshold_before_treatment',
                        'threshold',
                        'n_selected',
                        #'opportunity_rate',
                        'portion_selected',
                        'treatment_type'
                        ] + list(ufuncs.keys())
        return mask_df[col_ordering]

class BoostedBinaryClassifier(BaseWrangler):
    model_class = BaseParameter(DecisionTreeClassifier)
    model_param_grid = BaseParameter(None)
    #model_param_grid = BaseParameter(dict(max_depth=range(2, 23, 2), criterion=['gini', 'entropy'],
    #                                  min_samples_split=range(2, 20, 1)))
    target = BaseParameter(None, help_msg='String column name of the target df column')
    features = BaseParameter(None, help_msg='List of string column names of the feature columns'
                                            'If None, all columns minus target are used')
    n_jobs = BaseParameter(8, help_msg='Number of processes to use with Joblib in sklearn GridSearch')
    resample = BaseParameter(True, help_msg="Whether to use imblearn's random rebalancing")
    train_mode = BaseParameter(True, help_msg="Set False to use what model???")
    scorer_map = dict(
        f1=f1_score,
        accuracy=accuracy_score,
        precision=precision_score,
        recall=recall_score,
        mathews=matthews_corrcoef,
        roc=roc_auc_score
    )
    model_name = BaseParameter("binary_classifier_%s" % str(uuid.uuid4()).replace('-', '_'))
    #model_rt = RepoTreeParameter(None)#UnhashableParameter(None)
    model_rt = UnhashableParameter(None)
    performance_metric = BaseParameter('f1',
                                       help_msg="One of {"  + ", ".join(sorted(scorer_map.keys())) + "}")
                                       #help_msg="One of {'f1', 'accuracy', 'precision', 'recall', 'mathews'}")
    performance_metric_kwargs = BaseParameter(dict(average='binary'),
                                               help_msg="performance metric kwargs")
    comp_key = BaseParameter(None, "Set the key/index for output scores and other metrics")
    scoring_model_name = BaseParameter(None)
    score_series_name = BaseParameter('proba')
    return_model = BaseParameter(False, help_msg="Return the model rather than results")

    def _node_color(self):
        return '#fc6220'

    def run(self, df, features=None, target=None):
        if features is None and self.features is None:
            msg = "Either the Op parameter 'features' must be set or it must be passed to Op's run"
            raise ValueError(msg)
        if target is None and self.target is None:
            msg = "Either the Op parameter 'target' must be set or it must be passed to Op's run"
            raise ValueError(msg)

        logger = self.get_logger()

        self.features = features if features is not None else self.features
        self.target = target if target is not None else self.target

        self.predictions = None
        self.test_predictions = dict()
        self.train_ixes = None
        self.test_ixes = None

        if self.features is None:
            #logger.info("No features provided - searching input ops for 'features_provided' attribute")
            #self.features = self.get_input_features()
            logger.info("No features provided - using all but target as features")
            self.features = df.columns.drop(self.target).tolist()

        logger.info("Number of features: %d" % len(self.features))

        if self.train_mode:
            logger.info("Training")
            self.model = None
            ret = self.train(df)
        else:
            logger.info("Scoring")
            ret = self.score(df)

        if self.return_model:
            ret = self.model

        return ret

    def train(self, df):
        logger = self.get_logger()
        if self.model_rt is None:
            logger.warn("models are not being saved")

        np.random.seed(42)
        self.train_ixes, self.test_ixes = train_test_split(df.index,
                                                           stratify=df[self.target])
        train_df = df.loc[self.train_ixes]
        test_df = df.loc[self.test_ixes]

        logger.info("Train size: %d, Test size: %d" % (len(train_df), len(test_df)))
        logger.info("Scorer: %s" % self.performance_metric)
        if self.resample:
            # Log here so it doesn't log each iteration of the loop
            logger.info("Using pipelined random over sampler")

        m_name = self.model_class.__name__

        if self.resample:
            resampler = os.RandomOverSampler
            m = pl.Pipeline([('resampler', resampler()),
                             (m_name, self.model_class())])
            if self.model_param_grid is not None:
                tmp_grid = {"%s__%s" % (m_name, param): vals
                            for param, vals in self.model_param_grid.items()}
        else:
            m = self.model_class()
            tmp_grid = self.model_param_grid

        if self.model_param_grid is not None:
            m = self.cv_param_search(m,
                                     train_df[self.features],
                                     train_df[self.target],
                                     tmp_grid,
                                     scorer=self.performance_metric,
                                     n_jobs=self.n_jobs)
        else:
            m = m.fit(train_df[self.features], train_df[self.target])

        y_pred = m.predict(test_df[self.features])
        self.test_predictions[m_name] = pd.Series(y_pred,  # index=self.test_ixes,
                                                  index=(df.loc[self.test_ixes].set_index(self.comp_key).index
                                                         if self.comp_key is not None
                                                         else self.test_ixes),
                                                  name="%s_test_predictions" % m_name)
        print(m_name)
        self.print_classification_report(test_df[self.target],
                                         y_pred)
        test_performance = self.performance(test_df[self.target],
                                            y_pred)

        if self.model_rt is not None:
            save_name = self.model_name + "_" + m_name
            # TODO: Don't overwrite with worse performing model?
            self.model_rt.save(m, save_name,
                               author='pipeline',
                               auto_overwrite=True,
                               **test_performance)

        # print("Testing best model (%s)" % str(best_model))
        logger.info("Producing predictions for all models trained")
        scores = self.score(df, m)

        return m, test_performance, scores

    def score(self, df, model=None):
        # TODO: select best
        #model_df = self.asr_model_features(df)
        logger = self.get_logger()
        if model is None:
            if self.scoring_model_name is None:
                g = (l for l in self.model_rt.iterleaves(progress_bar=False)
                            if self.name in l.name and self.performance_metric in l.read_metadata())
                m_leaf = max(g, key=lambda l: l.md[self.performance_metric])
            elif self.scoring_model_name in self.model_rt:
                m_leaf = self.model_rt[self.scoring_model_name]
            else:
                msg = "Value of parameter scoring_model_name is not in tree %s" % str(self.model_rt)
                raise ValueError(msg)

            logger.info("Using %s " % m_leaf.name)
            model = m_leaf.load()
        probas = model.predict_proba(df[self.features])

        s = pd.Series(probas[:, 1], name=self.score_series_name,
                      index=df.set_index(self.comp_key).index if self.comp_key is not None else df.index)
        self.predictions = s
        return s

    @staticmethod
    def print_classification_report(y_true, y_pred,
                                    **kwargs):
        print classification_report(y_true, y_pred,
                                    **kwargs)

    @staticmethod
    def performance(y_true, y_pred):
       return dict(
                accuracy=accuracy_score(y_true, y_pred),
                f1=f1_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred),
                recall=recall_score(y_true, y_pred),
            )

    @staticmethod
    def cv_param_search(model, X, y,
                        param_grid,
                        scorer='f1', verbose=1,
                        n_jobs=4):
        try:
            _scorer = BaseBinaryClassifierModel.scorer_map[scorer]
        except KeyError:
            msg = ("Scorer %s is not supported, use one of {%s}"
                  % (scorer, ", ".join(BaseBinaryClassifierModel.scorer_map.keys())))
            print(msg)
            raise

        cv = GridSearchCV(estimator=model, param_grid=param_grid,
                          scoring=make_scorer(_scorer),
                          verbose=verbose, n_jobs=n_jobs,
                          pre_dispatch=n_jobs)
        cv_res = cv.fit(X, y)
        return cv_res

class BaseBinaryClassifierModel(BaseWrangler):
    baseline_models = [(DecisionTreeClassifier, dict()),
                       (RandomForestClassifier,dict()),
                       (GradientBoostingClassifier, dict())]
    param_search_models =[
        #(LogisticRegression, dict(penalty=['l1', 'l2'],
        #                          C=[.75, .9, 1., 1.1])),
        (DecisionTreeClassifier, dict(max_depth=range(2, 23, 2), criterion=['gini', 'entropy'],
                                      min_samples_split=range(2, 20, 1))),

        (RandomForestClassifier, dict(n_estimators=[5,  15, 30, 60, 100, 150],
                                      max_depth=[2, 3, 5, 7, 13],
                                      min_samples_split=range(2, 20, 3))),

        (GradientBoostingClassifier, dict(learning_rate=[0.085, 0.1, 0.2, ],
                                          n_estimators=[10, 30, 60, 100, 150],
                                          max_depth=[2, 3, 5, 7]))  # ,
        # min_samples_split=range(2, 10))
    ]

    # Improve these to also support integer indexing
    target = BaseParameter(help_msg='String column name of the target df column')
    features = BaseParameter(None, help_msg='List of string column names of the feature columns'
                                            'If None, all columns minus target are used')
    n_jobs = BaseParameter(8, help_msg='Number of processes to use with Joblib in sklearn GridSearch')
    resample = BaseParameter(True, help_msg="Whether to use imblearn's random rebalancing")
    train_mode = BaseParameter(True, help_msg="Set False to use what model???")
    test_size = BaseParameter(0.3, help_msg="Ratio of samples to split off for final testing")
    return_model = BaseParameter(False, help_msg="Return the model rather than results")
    model_name = BaseParameter("binary_classifier_%s" % str(uuid.uuid4()).replace('-', '_'))
    #model_rt = RepoTreeParameter(None)
    model_rt = UnhashableParameter(None)
    scoring_model_name = BaseParameter(None)
    comp_key = BaseParameter(None, "Set the key/index for output scores and other metrics")
    score_series_name = BaseParameter('proba')

    # WARN: Process assumes that higher scores are better
    scorer_map = dict(
        f1=f1_score,
        accuracy=accuracy_score,
        precision=precision_score,
        recall=recall_score,
        mathews=matthews_corrcoef,
        roc=roc_auc_score
    )
    performance_metric = BaseParameter('f1',
                                       help_msg="One of {"  + ", ".join(sorted(scorer_map.keys())) + "}")
                                       #help_msg="One of {'f1', 'accuracy', 'precision', 'recall', 'mathews'}")
    performance_metric_kwargs = BaseParameter(dict(average='binary'),
                                               help_msg="performance metric kwargs")

#    def __init__(self):
#        self.predictions = None
#        self.test_predictions = None
#        self.train_ixes = None
#        self.test_ixes = None

    def run(self, df):
        logger = self.get_logger()

        self.predictions = None
        self.test_predictions = dict()
        self.train_ixes = None
        self.test_ixes = None

        if self.features is None:
            #logger.info("No features provided - searching input ops for 'features_provided' attribute")
            #self.features = self.get_input_features()
            logger.info("No features provided - using all but target as features")
            self.features = df.columns.drop(self.target).tolist()

        logger.info("Number of features: %d" % len(self.features))

        if self.train_mode:
            logger.info("Training")
            self.model = None
            ret = self.train(df)
        else:
            logger.info("Scoring")
            ret = self.score(df)

        if self.return_model:
            ret = self.model

        return ret

    def score(self, df, model=None):
        # TODO: select best
        #model_df = self.asr_model_features(df)
        logger = self.get_logger()
        if model is None:
            if self.scoring_model_name is None:
                g = (l for l in self.model_rt.iterleaves(progress_bar=False)
                            if self.name in l.name and self.performance_metric in l.read_metadata())
                m_leaf = max(g, key=lambda l: l.md[self.performance_metric])
            elif self.scoring_model_name in self.model_rt:
                m_leaf = self.model_rt[self.scoring_model_name]
            else:
                msg = "Value of parameter scoring_model_name is not in tree %s" % str(self.model_rt)
                raise ValueError(msg)

            logger.info("Using %s " % m_leaf.name)
            model = m_leaf.load()
        probas = model.predict_proba(df[self.features])

        s = pd.Series(probas[:, 1], name=self.score_series_name,
                      index=df.set_index(self.comp_key).index if self.comp_key is not None else df.index)
        self.predictions = s
        return s

    def evaluate(self):
        logger = self.get_logger()

    def train(self, df):
        logger = self.get_logger()
        if self.model_rt is None:
            logger.warn("models are not being saved")

        #train_df, test_df = train_test_split(df,
        #                                     stratify=df[self.target],
        #                                     test_size=.25)

        np.random.seed(42)
        self.train_ixes, self.test_ixes = train_test_split(df.index,
                                                           stratify=df[self.target])
        train_df = df.loc[self.train_ixes]
        test_df = df.loc[self.test_ixes]

        logger.info("Train size: %d, Test size: %d" % (len(train_df), len(test_df)))
        logger.info("Scorer: %s" % self.performance_metric)
        if self.resample:
            # Log here so it doesn't log each iteration of the loop
            logger.info("Using pipelined random over sampler")

        models = dict()
        perf_res = dict()
        best_model, best_model_metric = None, -np.inf
        for m, mgrid in self.param_search_models:
            m_name = m.__name__
            if self.resample:
                resampler = os.RandomOverSampler
                m = pl.Pipeline([('resampler', resampler()),
                                 (m_name, m())])
                tmp_grid = {"%s__%s" %(m_name, param) : vals
                            for param, vals in mgrid.items()}
            else:
                m = m()
                tmp_grid = mgrid

            m = self.cv_param_search(m,
                                     train_df[self.features],
                                     train_df[self.target],
                                     tmp_grid,
                                     scorer=self.performance_metric,
                                     n_jobs=self.n_jobs)

            y_pred = m.predict(test_df[self.features])
            #df.set_index(self.comp_key).index if self.comp_key is not None else df.index
            self.test_predictions[m_name] = pd.Series(y_pred, #index=self.test_ixes,
                                                      index=(df.loc[self.test_ixes].set_index(self.comp_key).index
                                                             if self.comp_key is not None
                                                             else self.test_ixes),
                                                      name="%s_test_predictions" % m_name)
            print(m_name)
            self.print_classification_report(test_df[self.target],
                                             y_pred)
            perf_res[m_name] = self.performance(test_df[self.target],
                                                y_pred)
            models[m_name] = m

            if perf_res[m_name][self.performance_metric] > best_model_metric:
                best_model_metric = perf_res[m_name][self.performance_metric]
                best_model = m


        if self.model_rt is not None:
            for model_type_name, m in models.items():
                save_name = self.model_name + "_" + model_type_name
                #TODO: Don't overwrite with worse performing model?
                self.model_rt.save(m, save_name,
                                author='pipeline',
                                auto_overwrite=True,
                                **perf_res[model_type_name])


        #print("Testing best model (%s)" % str(best_model))
        #scores = self.score(df, best_model)
        logger.info("Producing predictions for all models trained")
        scores = pd.DataFrame({mname: self.score(df, m)
                               for mname, m in models.items()},
                              index=df.index)

        return models, perf_res, scores

    @staticmethod
    def print_classification_report(y_true, y_pred,
                                    **kwargs):
        print classification_report(y_true, y_pred,
                                    **kwargs)

    @staticmethod
    def performance(y_true, y_pred):
       return dict(
                accuracy=accuracy_score(y_true, y_pred),
                f1=f1_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred),
                recall=recall_score(y_true, y_pred),
            )

    @staticmethod
    def cv_param_search(model, X, y,
                        param_grid,
                        scorer='f1', verbose=1,
                        n_jobs=4):
        try:
            _scorer = BaseBinaryClassifierModel.scorer_map[scorer]
        except KeyError:
            msg = ("Scorer %s is not supported, use one of {%s}"
                  % (scorer, ", ".join(BaseBinaryClassifierModel.scorer_map.keys())))
            print(msg)
            raise

        cv = GridSearchCV(estimator=model, param_grid=param_grid,
                          scoring=make_scorer(_scorer),
                          verbose=verbose, n_jobs=n_jobs,
                          pre_dispatch=n_jobs)
        cv_res = cv.fit(X, y)
        return cv_res

    #@classmethod
    def param_search_on_features_and_test(self, model_df, features, target,
                                          models_and_grids=None, resample=None,
                                          test_size=None, n_jobs=None):
        models_and_grids = (self.param_search_models
                            if models_and_grids is None else models_and_grids)
        test_size = test_size if test_size is not None else self.test_size
        resample = resample if resample is not None else self.resample
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        logger = self.get_logger()

        ####
        target_s = model_df[target]

        train_df, test_df = train_test_split(model_df,
                                             stratify=target_s,
                                             test_size=test_size)
        train_target_s, test_target_s = train_df[target], test_df[target]

        res_models = dict()
        model_perf = dict()
        for m, pgrid in models_and_grids:
            m_name = m.__name__
            if resample:
                resampler = os.RandomOverSampler
                m = pl.Pipeline([('resampler', resampler()),
                                 (m_name, m())])
                tmp_pgrid = {"%s__%s" %(m_name, param) : vals
                            for param, vals in pgrid.items()}
            else:
                tmp_pgrid = pgrid

            logger.info("Grid searching %s" % m_name)
            res_models[m_name] = self.cv_param_search(model=m,
                                                     X=train_df[features],
                                                     y=train_target_s,
                                                     param_grid=tmp_pgrid,
                                                     n_jobs=n_jobs)

            preds = res_models[m_name].predict(test_df[features])
            self.print_classification_report(test_target_s, preds)

            model_perf[m_name] = dict(
                accuracy=accuracy_score(test_target_s, preds),
                f1=f1_score(test_target_s, preds),
                precision=precision_score(test_target_s, preds),
                recall=recall_score(test_target_s, preds),
            )

        return res_models, model_perf

    #@classmethod
    def train_on_features_and_test(self, model_df, features, target,
                                   models=None, resample=True):
        """

        :param model_df:
        :param features:
        :param target:
        :param models:
        :param resample:
        :return: tuple of dictionaries
            (model name -> model object, model name -> model performance dict)
        """
        models = self.baseline_models if models is None else models

        train_df, test_df = train_test_split(model_df,
                                             stratify=model_df[target],
                                             test_size=.25)
        logger = self.get_logger()

        model_res = dict()
        perf_res = dict()
        for m, kwargs in models:
            m_name = m.__name__

            if resample:
                resampler = os.RandomOverSampler
                m = pl.Pipeline([('resampler', resampler()),
                                          (m_name, m(**kwargs))])

            m = m.fit(train_df[features], train_df[target])
            y_pred = m.predict(test_df[features])
            self.print_classification_report(test_df[target],
                                             y_pred)
            perf_res[m_name] = self.performance(test_df[target],
                                                y_pred)
            logger.info(perf_res)

            model_res[m_name] = m

        return model_res, perf_res


class DecisionTreeModel(BaseBinaryClassifierModel):
        param_search_models =[
        (DecisionTreeClassifier, dict(max_depth=range(2, 23, 2), criterion=['gini', 'entropy'],
                                      min_samples_split=range(2, 20, 1)))]


class BootstrapFeatureImportances(OpVertex):

    model = BaseParameter(None)
    target = BaseParameter(None)
    features = BaseParameter(None)
    n_jobs = BaseParameter(1)
    test_size = BaseParameter(0.25)
    train_resample_n = BaseParameter(None)

    def run(self, df):
        logger = self.get_logger()
        if self.features is None:
            #logger.info("No features provided - searching input ops for 'features_provided' attribute")
            #self.features = self.get_input_features()
            logger.info("No features provided - using all but target as features")
            self.features = df.columns.drop(self.target).tolist()


        assert self.model is not None
        assert self.target is not None

        results = BootstrapFeatureImportances.balanced_sample_bootstrap_model(self.model, df,
                                                                              features=self.features,
                                                                              target=self.target,
                                                                              n_jobs=self.n_jobs,
                                                                              test_size=self.test_size,
                                                                              train_resample_n=self.train_resample_n
                                                                              )
        return results

    @staticmethod
    def balanced_sample_bootstrap_model(model, in_df, features, target, n_jobs=1,
                                        test_size=.25,
                                        train_resample_frac=None, train_resample_n=None,
                                        test_resample_frac=None, test_resample_n=None,
                                        iters=100):

        if isinstance(target, pd.Series):
            target.name = target.name if target.name is not None else 'target'
            # in_df = pd.concat([in_df, target], axis=1)
            in_df = in_df.join(target)
            target = target.name

        # if resample_frac is not None:
        #    resample_n = int(len(in_df) * resample_frac)
        # elif resample_n is not None:
        #    resample_n = int(resample_n)
        # else:
        #    resample_n = None

        train_df, test_df = train_test_split(in_df,
                                             stratify=in_df[target],
                                             test_size=test_size)
        if n_jobs > 1:
            print("N jobs: %d" % n_jobs)
            mgr = Manager()
            ns = mgr.Namespace()
            ns.train_df = train_df
            ns.test_df = test_df
            p = Pool(n_jobs, maxtasksperchild=1)

            args_gen = ((ns, model, features, target,
                         train_resample_frac, train_resample_n,
                         test_resample_frac, test_resample_n)

                        for i in tqdm(range(iters), desc='Bootstrapping (n_jobs=%d)' % n_jobs))
            results = [r for r in p.imap(BootstrapFeatureImportances.parallel_from_ns, args_gen, chunksize=1)]
            p.close()
            p.join()
        else:
            results = list()
            for it in tqdm(range(iters)):
                d = BootstrapFeatureImportances.run_bs_iter(model, train_df, test_df, features, target,
                                                            train_resample_frac, train_resample_n,
                                                            test_resample_frac, test_resample_n)
                results.append(d)
        return results

    @staticmethod
    def run_bs_iter(model, train_df, test_df, features, target,
                    train_resample_frac, train_resample_n,
                    test_resample_frac, test_resample_n):
        if train_resample_frac is not None and train_resample_n is not None:
            raise ValueError("Only one of train_resample_{frac,n} can be provided at a time")
        elif train_resample_frac is not None:
            train_sample_args = dict(frac=train_resample_frac, replace=True)
        elif train_resample_n is not None:
            train_sample_args = dict(n=train_resample_n, replace=True)
        else:
            train_sample_args = dict(frac=1., replace=True)

        if test_resample_frac is not None and test_resample_n is not None:
            raise ValueError("Only one of test_resample_{frac,n} can be provided at a time")
        elif test_resample_frac is not None:
            test_sample_args = dict(frac=test_resample_frac, replace=True)
        elif test_resample_n is not None:
            test_sample_args = dict(n=test_resample_n, replace=True)
        else:
            test_sample_args = dict(frac=1., replace=True)

        ###
        # Balance train data
        resamp_train_df = train_df.groupby(target).apply(lambda df: df.sample(**train_sample_args))
        resamp_train_df.reset_index(drop=True, inplace=True)

        ###
        # Keep test set at natural balance
        resamp_test_df = test_df.sample(**test_sample_args)
        ###

        m = clone(model)
        fm = m.fit(resamp_train_df[features], resamp_train_df[target])

        test_preds = fm.predict(resamp_test_df[features])

        #perf_metrics = gd.model.evaluate.measure_performance(resamp_test_df[target], test_preds)
        perf_metrics = BaseBinaryClassifierModel.performance(resamp_test_df[target], test_preds)

        d = dict(model=fm, **perf_metrics)
        return d


    @staticmethod
    def parallel_from_ns(args):
        (ns, model, features, target,
         resample_n, resample_frac) = args
        return BootstrapFeatureImportances.run_bs_iter(model=model, train_df=ns.train_df, test_df=ns.test_df,
                                                       features=features, target=target,
                                                       resample_n=resample_n, resample_frac=resample_frac)

class DenoisingAutoEncoder(BaseWrangler):
    pass

class BaseBinaryClassifierModel_old(BaseWrangler):
    name = 'base_model'
    baseline_models = [(DecisionTreeClassifier, dict()),
                       (RandomForestClassifier,dict()),
                       (GradientBoostingClassifier, dict())]

    param_search_models =[
        #(LogisticRegression, dict(penalty=['l1', 'l2'],
        #                          C=[.75, .9, 1., 1.1])),
        (DecisionTreeClassifier, dict(max_depth=range(2, 10),
                                      min_samples_split=range(2, 30, 2))),

        (RandomForestClassifier, dict(n_estimators=[5,  15, 30, 60, 100, 150],
                                      max_depth=[2, 3, 5, 7, 13],
                                      min_samples_split=range(2, 20, 3))),

        (GradientBoostingClassifier, dict(learning_rate=[0.085, 0.1, 0.2, ],
                                          n_estimators=[10, 30, 60, 100, 150],
                                          max_depth=[2, 3, 5, 7]))  # ,
        # min_samples_split=range(2, 10))
    ]
    depends = dict()
    target = None
    features = None
    test_size = .25
    resample = True
    n_jobs = 4

    def __init__(self, **kwargs):
        super(BaseBinaryClassifierModel, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if self.run_type == 'production':
            return self.score(*args, **kwargs)
        elif self.run_type == 'training':
            return self.train(*args, **kwargs)

    def score(self, **kwargs):
        raise NotImplemented("Score not implemented on %s"
                            % self.__class__.__name__)

    def train(self, **kwargs):
        raise NotImplemented("Train not implemented on %s"
                             % self.__class__.__name__)

    @staticmethod
    def print_classification_report(y_true, y_pred,
                                    **kwargs):
        print classification_report(y_true, y_pred,
                                    **kwargs)

    @staticmethod
    def performance(y_true, y_pred):
       return dict(
                accuracy=accuracy_score(y_true, y_pred),
                f1=f1_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred),
                recall=recall_score(y_true, y_pred),
            )

    @staticmethod
    def cv_param_search(model, X, y,
                        param_grid,
                        scorer=f1_score, verbose=1,
                        n_jobs=4):
        cv = GridSearchCV(estimator=model, param_grid=param_grid,
                          scoring=make_scorer(scorer),
                          verbose=verbose, n_jobs=n_jobs,
                          pre_dispatch=n_jobs)
        cv_res = cv.fit(X, y)
        return cv_res

    #@classmethod
    def param_search_on_features_and_test(self, model_df, features, target,
                                          models_and_grids=None, resample=None,
                                          test_size=None, n_jobs=None):
        models_and_grids = (self.param_search_models
                            if models_and_grids is None else models_and_grids)
        test_size = test_size if test_size is not None else self.test_size
        resample = resample if resample is not None else self.resample
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs

        logger = self.get_logger()

        ####
        target_s = model_df[target]

        train_df, test_df = train_test_split(model_df,
                                             stratify=target_s,
                                             test_size=test_size)
        train_target_s, test_target_s = train_df[target], test_df[target]

        res_models = dict()
        model_perf = dict()
        for m, pgrid in models_and_grids:
            m_name = m.__name__
            if resample:
                resampler = os.RandomOverSampler
                m = pl.Pipeline([('resampler', resampler()),
                                 (m_name, m())])
                tmp_pgrid = {"%s__%s" %(m_name, param) : vals
                            for param, vals in pgrid.items()}
            else:
                tmp_pgrid = pgrid

            logger.info("Grid searching %s" % m_name)
            res_models[m_name] = self.cv_param_search(model=m,
                                                     X=train_df[features],
                                                     y=train_target_s,
                                                     param_grid=tmp_pgrid,
                                                     n_jobs=n_jobs)

            preds = res_models[m_name].predict(test_df[features])
            self.print_classification_report(test_target_s, preds)

            model_perf[m_name] = dict(
                accuracy=accuracy_score(test_target_s, preds),
                f1=f1_score(test_target_s, preds),
                precision=precision_score(test_target_s, preds),
                recall=recall_score(test_target_s, preds),
            )

        return res_models, model_perf

    #@classmethod
    def train_on_features_and_test(self, model_df, features, target,
                                   models=None, resample=True):
        """

        :param model_df:
        :param features:
        :param target:
        :param models:
        :param resample:
        :return: tuple of dictionaries
            (model name -> model object, model name -> model performance dict)
        """
        models = self.baseline_models if models is None else models

        train_df, test_df = train_test_split(model_df,
                                             stratify=model_df[target],
                                             test_size=.25)
        logger = self.get_logger()

        model_res = dict()
        perf_res = dict()
        for m, kwargs in models:
            m_name = m.__name__

            if resample:
                resampler = os.RandomOverSampler
                m = pl.Pipeline([('resampler', resampler()),
                                          (m_name, m(**kwargs))])

            m = m.fit(train_df[features], train_df[target])
            y_pred = m.predict(test_df[features])
            self.print_classification_report(test_df[target],
                                             y_pred)
            perf_res[m_name] = self.performance(test_df[target],
                                                y_pred)
            logger.info(perf_res)

            model_res[m_name] = m

        return model_res, perf_res


class TorchLogit(OpVertex):
    target = BaseParameter(help_msg='String column name of the target df column')
    features = BaseParameter(None, help_msg='List of string column names of the feature columns'
                                            'If None, all columns minus target are used')
    n_epochs = BaseParameter(1)
    batch_size = BaseParameter(128)
    learning_rate = BaseParameter(1e-5)

    @staticmethod
    def log_loss(actual, pred):
        import torch
        return torch.log(1 + torch.exp(-actual * pred)).mean()

    @staticmethod
    def batch_gen(feat_arr, target_arr, batch_size):
        ix = list(range(feat_arr.shape[0]))
        np.random.shuffle(ix)
        #field_nums = feature_field_nums + [target_field_num]
        def batch_gen():
            done, i = False, 0
            while not done:
                if i > (len(ix) - batch_size):
                    ixes, i = ix[i:], 0
                    done = True
                else:
                    ixes = ix[i:i + batch_size]
                    i = i + batch_size

                X = feat_arr[ixes]
                y = target_arr[ixes]
                yield X.astype(float), y.astype(float)
        return batch_gen()

    def run(self, df):
        import torch
        from tqdm import tqdm_notebook
        feats = df.columns.tolist() if self.features is None else self.features
        n_features = len(feats)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_features, 1)
        )
        opt = torch.optim.SGD(self.model.parameters(),
                              lr=self.learning_rate)
        training_metrics = list()
        for n in range(self.n_epochs):
            g = self.batch_gen(df[feats].values,
                               df[self.target].values,
                               batch_size=128)
            loss_total = 0
            for i, (x, y) in tqdm_notebook(enumerate(g),
                                           total=np.ceil(len(df)/self.batch_size)):
                # x, y = next(g)
                x = torch.tensor(x).float()
                y = torch.tensor(y).reshape(-1, 1).float()
                y = (y * 2) - 1

                # Forward pass: compute predicted y
                y_pred = self.model(x)
                loss = self.log_loss(y_pred.float(), y)
                loss_total += loss
                training_metrics.append(dict(epoch=n,
                                             step=i,
                                             loss=loss.item()))
                opt.zero_grad()
                loss.backward()
                opt.step()
            #print(training_metrics[-5:])

            print("Loss: %f" % (loss_total/float(i+1)))
        _p = list(self.model.parameters())
        coefs_s = pd.Series(_p[0][0].detach().numpy(),
                            index=feats)
        training_mets_df = pd.DataFrame(training_metrics)
        return coefs_s, training_mets_df


