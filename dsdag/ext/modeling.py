import uuid
from dsdag.core.op import OpVertex
from dsdag.core.parameter import BaseParameter, DatetimeParameter, UnhashableParameter
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
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
    score_series_name = BaseParameter('proba')

    # WARN: Process assumes that higher scores are better
    scorer_map = dict(
        f1=f1_score,
        accuracy=accuracy_score,
        precision=precision_score,
        recall=recall_score,
        mathews=matthews_corrcoef
    )
    performance_metric = BaseParameter('f1',
                                       help_msg="One of {"  + ", ".join(sorted(scorer_map.keys())) + "}")
                                       #help_msg="One of {'f1', 'accuracy', 'precision', 'recall', 'mathews'}")


    def run(self, df):
        logger = self.get_logger()
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
                msg = "Value of parameter scoring_model_name is not in tree %s" % str(self.siu_rt)
                raise ValueError(msg)

            logger.info("Using %s " % m_leaf.name)
            model = m_leaf.load()
        probas = model.predict_proba(df[self.covariates])

        s = pd.Series(probas[:, 1], name=self.score_series_name,
                      index=df.set_index(self.comp_key).index)
        return s


    def train(self, df):
        logger = self.get_logger()
        if self.model_rt is None:
            logger.warn("models are not being saved")
        train_df, test_df = train_test_split(df,
                                             stratify=df[self.target],
                                             test_size=.25)
        logger.info("Train size: %d, Test size: %d" % (len(train_df), len(test_df)))
        logger.info("Scorer: %s" % self.performance_metric)

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
                tmp_grid = mgrid

            m = self.cv_param_search(m,
                                     train_df[self.features],
                                     train_df[self.target],
                                     tmp_grid,
                                     scorer=self.performance_metric,
                                     n_jobs=self.n_jobs)

            y_pred = m.predict(test_df[self.features])
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