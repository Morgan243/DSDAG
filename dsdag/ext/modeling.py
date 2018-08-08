import uuid
from dsdag.core.op import OpVertex
from dsdag.core.parameter import BaseParameter, DatetimeParameter, RepoTreeParameter
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn import pipeline as pl
from imblearn import over_sampling as os

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


class BaseBinaryClassifierModel(BaseWrangler):
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

    # Improve these to also support integer indexing
    target = BaseParameter(help_msg='String column name of the target df column')
    features = BaseParameter(help_msg='List of string column names of the feature columns')
    n_jobs = BaseParameter(8, help_msg='Number of processes to use with Joblib in sklearn GridSearch')
    resample = BaseParameter(True, help_msg="Whether to use imblearn's random rebalancing")
    train_mode = BaseParameter(True, help_msg="Set False to use what model???")
    test_size = BaseParameter(0.3, help_msg="Ratio of samples to split off for final testing")
    return_model = BaseParameter(False, help_msg="Return the model rather than results")
    model_name = BaseParameter("binary_classifier_%s" % str(uuid.uuid4()).replace('-', '_'))
    model_rt = RepoTreeParameter(None)

    def run(self, df):
        if self.train_mode:
            self.model = None
            ret = self.train(df)
        else:
            ret = self.score(df)

        if self.return_model:
            ret = self.model

        return ret


    def score(self, df, model=None):
        # TODO: select best
        #model_df = self.asr_model_features(df)
        #g = (l for l in self.siu_rt.iterleaves(progress_bar=False)
        #            if self.name in l.name and 'f1' in l.read_metadata())
        #best_m_leaf = max(g, key=lambda l: l.md['f1'])
        #logger.info("Using %s " % best_m_leaf.name)
        #model = best_m_leaf.load()
        if model is None:
            if self.model_rt is None:
                raise ValueError()
            else:
                model = self.model_rt[self.model_name]()

        probas = model.predict_proba(df[self.features])

        s = pd.Series(probas[:, 1], name=self.model_name + '_proba',
                      index = df.index)
                      #index=df.set_index(self.comp_key).index)
        return s
#        raise NotImplemented("Score not implemented on %s"
#                             % self.__class__.__name__)

    def train(self, df):
        train_df, test_df = train_test_split(df,
                                             stratify=df[self.target],
                                             test_size=.25)
        models = dict()
        perf_res = dict()
        best_model, best_model_f1 = None, -np.inf
        for m, mgrid in self.param_search_models:
            m_name = m.__name__
            if self.resample:
                resampler = os.RandomOverSampler
                m = pl.Pipeline([('resampler', resampler()),
                                 (m_name, m())])
                tmp_grid = {"%s__%s" %(m_name, param) : vals
                            for param, vals in mgrid.items()}
            else:
                tmp_grid = mgrid

            m = self.cv_param_search(m,
                                     train_df[self.features],
                                     train_df[self.target],
                                     tmp_grid, n_jobs=self.n_jobs)

            y_pred = m.predict(test_df[self.features])
            print(m_name)
            self.print_classification_report(test_df[self.target],
                                             y_pred)
            perf_res[m_name] = self.performance(test_df[self.target],
                                                y_pred)
            models[m_name] = m

            if perf_res[m_name]['f1'] > best_model_f1:
                best_model_f1 = perf_res[m_name]['f1']
                best_model = m


        if self.model_rt is not None:
            for model_type_name, m in models.items():
                save_name = self.model_name + "_" + model_type_name
                #TODO: Don't overwrite with worse performing model?
                self.model_rt.save(m, save_name,
                                author='pipeline',
                                auto_overwrite=True,
                                **perf_res[model_type_name])


        print("Testing best model (%s)" % str(best_model))
        scores = self.score(df, best_model)

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