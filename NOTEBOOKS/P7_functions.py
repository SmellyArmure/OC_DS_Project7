# return a dataframe of missing values (total and percent)

def missing_data(data):
    
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()*100)\
        .sort_values(ascending=False)
    return pd.concat([total, percent],
                     axis=1,
                     keys=['Total', 'Percent'])


# Plotting histograms of specified quantitative continuous columns of a dataframe and mean, median and mode values.

import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(df, cols, bins=30, figsize=(12, 7), color='grey',
                    skip_outliers=False, thresh=3, n_cols=3, tight_layout=None,
                    sh_tit=13):

    fig = plt.figure(figsize=figsize)
    n_tot = len(cols)
    n_rows = (n_tot // n_cols)+((n_tot % n_cols) > 0)*1

    for i, c in enumerate(cols, 1):
        mask_na = df[c].notna()
        ser = df[c].loc[mask_na]
        mask_isfin = np.isfinite(ser)
        ser = ser.loc[mask_isfin]

        ax = fig.add_subplot(n_rows, n_cols, i)
        if skip_outliers:
            mask_outl = np.abs(st.zscore(ser)) < thresh
            ser = ser.loc[mask_outl]
        else:
            ser = df[c].loc[mask_na]
        ax.hist(ser, ec='k', bins=bins, color=color)
        title = c if len(c) < 2 * sh_tit else c[:sh_tit] + '...' + c[-sh_tit:]
        ax.set_title(title)
        ax.vlines(ser.mean(), *ax.get_ylim(),  color='red', ls='-', lw=1.5)
        ax.vlines(ser.median(), *ax.get_ylim(), color='green', ls='-.', lw=1.5)
        ax.vlines(ser.mode()[0], *ax.get_ylim(), color='goldenrod', ls='--', lw=1.5)
        ax.legend(['mean', 'median', 'mode'])
        ax.title.set_fontweight('bold')
        # xmin, xmax = ax.get_xlim()
        # ax.set(xlim=(0, xmax/5))
    if tight_layout is not None:
        plt.tight_layout(**tight_layout)
    plt.show()


"""
Plotting histograms of specified quantitative continuous columns of a
dataframe in order to compare histograms of different categories.
"""

import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_cat_histograms(df, ind_tuple, cols, figwidth=20,  # labels, colors, bins=30,
                        skip_outliers=False, thresh=3, n_cols=3, tight_layout=True,
                        sh_tit=13, filter_finite=True):

    df_0_ind, df_1_ind = ind_tuple
    n_tot = len(cols)
    n_rows = (n_tot // n_cols) + ((n_tot % n_cols) > 0) * 1
    figsize = (figwidth, 3)

    # loop on each row
    for j, row in enumerate(range(n_rows), 0):

        fig = plt.figure(figsize=figsize)
        sub_cols = list(cols)[j*n_cols:(j+1)*n_cols]

        for i, col in enumerate(sub_cols, 1):

            ax = fig.add_subplot(1, n_cols, i)

            mask_na = df[col].notna()
            ser = df[col].loc[mask_na]
            if filter_finite:
                mask_isfin = np.isfinite(ser)
                ser = ser.loc[mask_isfin]
            if skip_outliers:
                mask_outl = np.abs(st.zscore(ser)) < thresh
                ser = ser.loc[mask_outl]

            ind_0 = [i for i in ser.index if i in df_0_ind]
            ind_1 = [i for i in ser.index if i in df_1_ind]

            df.loc[ind_0, col].hist(label='repaid', alpha=0.5, ec='k',
                                    bins=30, color='green', ax=ax, density=True)

            df.loc[ind_1, col].hist(label='not repaid', alpha=0.5, ec='k',
                                    bins=30, color='red', ax=ax, density=True)

            plt.legend()

            title = col if len(col) < 2 * sh_tit else col[:sh_tit] + '...' + col[-sh_tit:]
            ax.set_title(title)
            ax.title.set_fontweight('bold')
            if tight_layout:
                plt.tight_layout()
        plt.show()


'''
# Plotting heatmap of the ratio histograms of specified quantitative continuous columns of a
dataframe in order to compare histograms of different categories.
'''

import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_target_ratio_categories(df, col_target, cols, cmap='magma',
                                 figwidth=20, aspect=7, n_cols=5, tight_layout=True,
                                 sh_tit=20, shorten_label=10):

    n_tot = len(cols)
    n_rows = (n_tot // n_cols) + ((n_tot % n_cols) > 0) * 1
    figheight = figwidth/aspect
    figsize = (figwidth, figheight)

    # loop on each row
    for j, row in enumerate(range(n_rows), 0):

        fig = plt.figure(figsize=figsize)  # height
        sub_cols = list(cols)[j*n_cols:(j+1)*n_cols]

        for i, col in enumerate(sub_cols, 1):

            ax = fig.add_subplot(1, n_cols, i)

            ct = pd.crosstab(df[col_target], df[col])
            sns.heatmap(ct/ct.sum(axis=0), cmap=cmap, annot=True, fmt='.2f')

            if len(col) < 2:
                title = col * sh_tit
            else:
                title = col[:sh_tit] + '...' + col[-sh_tit:]

            ax.axes.get_xaxis().get_label().set_visible(False)
            ax.set_title(title)
            ax.title.set_fontweight('bold')

            if shorten_label:
                thr = int(shorten_label)
                lab_x = [item.get_text() for item in ax.get_xticklabels()]
                short_lab_x = [s[:thr] + '...' + s[-thr:] if len(s) > 2 * thr else s for s in lab_x]
                ax.axes.set_xticklabels(short_lab_x)

            if tight_layout:
                plt.tight_layout()
        plt.show()


# draws boxplot of the categories of a dataframe columns and returns the value_counts

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_boxplot_categories(df, col_cat, col_val, n_sh=None,
                            log_scale=False, ax=None):

    n_sh = 100 if n_sh is None else n_sh
    val_cnts = df[col_cat].value_counts()
    sns.boxplot(x=col_cat, y=col_val, data=df, ax=ax)
    if log_scale:
        plt.yscale('log')
    xticklab = [item.get_text()[0: n_sh - 1] + f'... ({val_cnts[item.get_text()]})' if len(item.get_text()) > n_sh else
                item.get_text() + f' ({val_cnts[item.get_text()]})' for item in plt.gca().get_xticklabels()]
    plt.gca().axes.set_xticklabels(xticklab)
    plt.xticks(rotation=45, ha='right')
    return val_cnts


# Plotting bar plots of the main categorical columns

import pandas as pd
import matplotlib.pyplot as plt


def plot_barplots(df, cols, figsize=(12, 7), n_cols=3, shorten_label=False,
                  color='grey'):

    n_tot = len(cols)
    n_rows = (n_tot // n_cols)+((n_tot % n_cols) > 0) * 1

    fig = plt.figure(figsize=figsize)
    for i, c in enumerate(cols, 1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        ser = df[c].value_counts()
        n_cat = ser.shape[0]
        if n_cat > 15:
            ser[0:15].plot.bar(color=color, ec='k', ax=ax)
        else:
            ser.plot.bar(color=color, ec='k', ax=ax)
        ax.set_title(c[0:20]+f' ({n_cat})', fontweight='bold')
        labels = [item.get_text() for item in ax.get_xticklabels()]

        if shorten_label:
            thr = int(shorten_label)
            lab_x = [item.get_text() for item in ax.get_xticklabels()]
            short_lab_x = [s[:thr] + '...' + s[-thr:] if len(s) > 2 * thr else s for s in lab_x]
            ax.axes.set_xticklabels(short_lab_x)

        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()


'''
simple bar plot with shortening label option
'''
from matplotlib import colors


def plot_simple_barplot(x, y, x_lab=None, y_lab=None, title=None,
                        shorten_label=15, figsize=(25, 3), color=None,
                        annotate=False):

    color = None if color is None else colors.get_named_colors_mapping()[color]

    fig = plt.figure(figsize=figsize)
    sns.barplot(x=x, y=y, ec='k', color=color)

    plt.xticks(rotation=45, ha='right')
    ax = plt.gca()
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    ax.set_title(title,
                 fontweight='bold', pad=15)

    if shorten_label:
        thr = int(shorten_label)
        lab_x = [item.get_text() for item in ax.get_xticklabels()]
        short_lab_x = [s[:thr] + '...' + s[-thr:] if len(s) > 2*thr else s for s in lab_x]
        ax.axes.set_xticklabels(short_lab_x)

    if annotate:
        for i, val in enumerate(y):
            ymin, ymax = ax.get_ylim()
            plt.text(i-0.25, val+(ymax-ymin)*1/30, f'{val}')  # fontsize=10)
            ax.set(ylim=(ymin, ymax+(ymax-ymin)*0.005))

    plt.grid()


"""
Plotting heatmap (2 options available, rectangle or triangle )
"""

import seaborn as sns


def plot_heatmap(corr, title, figsize=(8, 4), vmin=-1, vmax=1, center=0,
                 palette=sns.color_palette("coolwarm", 20), shape='rect',
                 fmt='.2f', annot=True, robust=False, fig=None, ax=None):

    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax

    if shape == 'rect':
        mask = None
    elif shape == 'tri':
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = None
        print('ERROR : this type of heatmap does not exist')

    ax = sns.heatmap(corr, mask=mask, cmap=palette, vmin=vmin, vmax=vmax,
                     center=center, annot=annot, annot_kws={"size": 10}, fmt=fmt,
                     square=False, linewidths=.5, linecolor='white',
                     cbar_kws={"shrink": .9, 'label': None}, robust=robust,
                     xticklabels=corr.columns, yticklabels=corr.index,
                     ax=ax)
    ax.tick_params(labelsize=10, top=False, bottom=True,
                   labeltop=False, labelbottom=True)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontweight='bold', fontsize=12)


# Data Preprocessing for quantitative and categorical data with encoding options

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn import metrics


def data_preprocessing(df, var_model, var_target, enc_strat_cat='label'):

    # Data Processing
    df_train = df[var_model+[var_target]].copy('deep')
    if df[var_model].isna().sum().sum() != 0:
        print("ERROR: var_model columns should not contain nan !!!")
        return None, None
    else:
        cat_cols = df_train[var_model].select_dtypes('object').columns
        num_cols = df_train[var_model].select_dtypes(include=np.number).columns
        # # Encoding categorical values
        if enc_strat_cat == 'label':
        # --- OPTION 1: Label Encoding categorical values
            for c in cat_cols:
                df_train[c] = LabelEncoder().fit_transform(df_train[c].values)
        elif enc_strat_cat == 'hashing':
        # --- OPTION 2: Feature hashing of categorical values
            for c in cat_cols:
                df_train[c] = df_train[c].astype('str')
                n_feat = 5
                hasher = FeatureHasher(n_features=n_feat, input_type='string')
                f = hasher.transform(df_train[c])
                arr = pd.DataFrame(f.toarray(), index=df_train.index)
                df_train[[c+'_'+str(i+1) for i in range(n_feat)]] = pd.DataFrame(arr)
                del df_train[c]
                cols = list(df_train.columns)
                cols.remove(var_target)
                df_train = df_train.reindex(columns=cols+[var_target])
        else:
            print("ERROR: Wrong value of enc_strat_cat")
            return None, None
        # # Standardizing quantitative values
        if len(list(num_cols)):
            df_train[num_cols] = \
                      StandardScaler().fit_transform(df_train[num_cols].values)
        # Splitting in X and y, then in training and testing set
        X = df_train.iloc[:, :-1].values
        y = df_train.iloc[:, -1].values
        return X, y


def naive_model_compare_r2(X_tr, y_tr, X_te, y_te, y_pr):

    # Model
    print('--- model: {:.3}'.format(metrics.r2_score(y_te, y_pr)))
    # normal random distribution
    y_pr_rand = np.random.normal(0, 1, y_pr.shape)
    print('--- normal random distribution: {:.3}'
          .format(metrics.r2_score(y_te, y_pr_rand)))
    # dummy regressors
    for s in ['mean', 'median']:
        dum = DummyRegressor(strategy=s).fit(X_tr, y_tr)
        y_pr_dum = dum.predict(X_te)
        print('--- dummy regressor (' + s + ') : r2_score={:.3}'
              .format(metrics.r2_score(y_te, y_pr_dum)))


def naive_model_compare_acc_f1(X_tr, y_tr, X_te, y_te, y_pr, average='weighted'):

    print('ooooooo CLASSIFICATION METRICS oooooooo')

    def f1_prec_recall(yte, ypr):
        prec = metrics.precision_score(yte, ypr, average=average)
        rec = metrics.recall_score(yte, ypr, average=average)
        f1 = metrics.f1_score(yte, ypr, average=average)
        return [f1, prec, rec]
    # Model
    print('--- model: f1={:.3}, precision={:.3}, recall={:.3}'
          .format(*f1_prec_recall(y_te, y_pr)))
    # Dummy classifier
    for s in ['stratified', 'most_frequent', 'uniform']:
        dum = DummyClassifier(strategy=s).fit(X_tr, y_tr)
        y_pr_dum = dum.predict(X_te)
        print('--- dummy class. (' + s
              + '): f1={:.3}, precision={:.3}, recall={:.3}'
              .format(*f1_prec_recall(y_te, y_pr_dum)))


def plot_hist_pred_val(y_te, y_pr, y_pr_, bins=150, xlim=(0, 20), short_lab=False):
    # Plotting dispersion of data to be imputed
    bins = plt.hist(y_te, alpha=0.5, color='b', bins=bins, density=True,
                    histtype='step', lw=3, label='y_te (real val. from test set)')[1]
    ax = plt.gca()
    ax.hist(y_pr, alpha=0.5, color='g', bins=bins, density=True,
            histtype='step', lw=3, label='y_pr (pred. val. from test set)')
    ax.hist(y_pr_, alpha=0.5, color='r', bins=bins, density=True,
            histtype='step', lw=3, label='y_pr_ (pred. val. to be imputed)')
    ax.set(xlim=xlim)
    plt.xticks(rotation=45, ha='right')
    plt.draw()
    if short_lab:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        short_labels = [s[0:7]+'.' if len(s) > 7 else s for s in labels]
        ax.axes.set_xticklabels(short_labels)
    ax.legend(loc=1)
    plt.title("Frequency of values", fontweight='bold', fontsize=12)
    plt.gcf().set_size_inches(6, 2)
    plt.show()


# Works for both quantitative (knnregressor)
# and categorical (knnclassifier) target features

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def model_impute(df, var_model, var_target, enc_strat_cat='label',
                 n_model='knn', clip=None, plot=True):
    
    dict_models = {'knn': (KNeighborsRegressor(),
                           KNeighborsClassifier()),
                   'rf': (RandomForestRegressor(),
                          RandomForestClassifier())}
    
    if df[var_target].isna().sum() == 0:
        print('ERROR: Nothing to impute (target column already filled)')
        return None, None
    else:
        if df[var_target].dtype == 'object':
            # knn classifier
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            model = dict_models[n_model][1]
            gsCV = GridSearchCV(model,
                                {},  #'n_neighbors': [5]
                                cv=skf, return_train_score=True,
                                scoring='f1_weighted')
            mod = 'class'
        elif df[var_target].dtype in ['float64', 'int64']:
            # knn regressor
            kf = KFold(n_splits=5, shuffle=True)
            model = dict_models[n_model][0]
            gsCV = GridSearchCV(model,
                                {}, # 'n_neighbors': [5]
                                cv=kf, return_train_score=True)
            mod = 'reg'
        else:
            kf = None
            model = None
            gsCV = None
            mod = 'reg'
            print("ERROR: dtype of target feature unknown")
        # Data Preprocessing
        X, y = data_preprocessing(df.dropna(subset=var_model+[var_target]),
                                  var_model=var_model, var_target=var_target,
                                  enc_strat_cat=enc_strat_cat)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        # Training KNN
        gsCV.fit(X_tr, y_tr)
        # res = gsCV.cv_results_
        # Predicting test set with the model and clipping
        y_pr = gsCV.predict(X_te)
        try:
            if clip:
                y_pr = y_pr.clip(*clip)  # regressor option only
        except:
            print("ERROR: clip available for regressor option only") 
        # Comparison with naive baselines
        if mod == 'class':
            naive_model_compare_acc_f1(X_tr, y_tr, X_te, y_te, y_pr, average='micro')
        elif mod == 'reg':
            naive_model_compare_r2(X_tr, y_tr, X_te, y_te, y_pr)
        else:
            print("ERROR: check type of target feature...")
        # Predicting using knn
        ind_to_impute = df.loc[df[var_target].isna()].index 
        X_, y_ = data_preprocessing(df.loc[ind_to_impute], var_model=var_model,
                                    var_target=var_target,
                                    enc_strat_cat=enc_strat_cat)
        # Predicting with model
        y_pr_ = gsCV.predict(X_)
        # Plotting histogram of predicted values
        short_lab = True if mod == 'class' else False
        if plot: plot_hist_pred_val(y_te, y_pr, y_pr_, short_lab=short_lab)
        # returning indexes to impute and calculated values
        return ind_to_impute, y_pr_
    

''' Class to filter outliers from X and y from the zscore of the X columns
eliminates a line if the value of one or more features is outlier. 
(CANNOT BE USED IN A PIPELINE !!!) version P4'''

import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin


class ZscoreSampleFilter(BaseEstimator, TransformerMixin):
    def __init__(self, thresh = None, keep=None):
        self.thresh = thresh if thresh is not None else 5
        self.keep = keep if keep is not None else 'any'

    def fit(self, X, y=None):
        self.X_zscore = X.apply(st.zscore, axis=0)
        if self.keep=='all':
            # on garde les lignes si toutes sont des inliers
            self.samplefilter = (np.abs(self.X_zscore) < self.thresh).all(1)
        elif self.keep=='any':
            # on garde les lignes si une seule est un inlier
            self.samplefilter = (np.abs(self.X_zscore) < self.thresh).any(1)
        return self

    def transform(self, X, y=None):  # copy=None
        # X_mod = X.loc[:,self.featurefilter]
        X_mod = X.loc[self.samplefilter]
        if y is not None:
            y_mod = y.loc[self.samplefilter]
            return X_mod, y_mod
        else:
            return X_mod

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)


"""  
Class to filter outliers from X and y from a LOF analysis on X
(CANNOT BE USED IN A PIPELINE !!!)
A threshold is set for selection criteria, 
neg_conf_val (float): threshold for excluding samples with a lower
 negative outlier factor.
 NB: may not be that useful, because we can use LocalOutlierFactor.predict method...
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor


class LOFSampleFilter(BaseEstimator, TransformerMixin):

    def __init__(self, contamination=None, n_neighbors=None, **kwargs):
        self.contamination = contamination if contamination is not None else 0.05
        self.n_neighbors = n_neighbors if n_neighbors is not None else 5
        self.kwargs = kwargs

    def fit(self, X, y=None, *args, **kwargs):
        lcf = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                 contamination=self.contamination,
                                 **self.kwargs)
        samplefilter = pd.Series(lcf.fit_predict(X))
        samplefilter = samplefilter.replace({1: True, # inliners
                                          -1: False}) # outliers
        # computes the filtered dataframe
        self.X_mod = X.loc[samplefilter.values]
        if y is not None:
            self.y_mod = y.loc[samplefilter.values]
        return self

    def transform(self, X, y=None, copy=None):
        if y is not None:
            return self.X_mod, self.y_mod
        else:
            return self.X_mod

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)


"""  
Class to filter outliers from X and y from a LOF analysis on X
(CANNOT BE USED IN A PIPELINE !!!)
A threshold is set for selection criteria, 
score_samples (float): threshold for excluding samples with a lower
score_samples.
NB: may not be that useful, because we can use IsolationForest.predict method...
"""

from sklearn.ensemble import IsolationForest


class IsolForestSampleFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, contamination=None, n_estimators=None, **kwargs):
        self.contamination = contamination if contamination is not None else 0.05
        self.n_estimators = n_estimators if n_estimators is not None else 100
        self.kwargs = kwargs

    def fit(self, X, y=None, *args, **kwargs):
        isolf = IsolationForest(n_estimators=self.n_estimators,
                                contamination=self.contamination,
                                **self.kwargs)
        self.samplefilter = pd.Series(isolf.fit_predict(X))
        self.samplefilter = (self.samplefilter).replace({1: True,  # inliners
                                                         -1: False})  # outliers
        return self

    def transform(self, X, y=None):  # , copy=None
        X_mod = X.loc[self.samplefilter.values]
        if y is not None:
            y_mod = y.loc[self.samplefilter.values]
            return X_mod, y_mod
        else:
            return X_mod

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)


'''
t-SNE wrapper in order to use t-SNE as a dimension reducter as a pipeline step of a 
GridSearch (indeed, tsne doesn't have a transform method but only a fit_transform 
method -> it cannot be applied to another set of data than the one on which it was trained)
'''

from sklearn.manifold import TSNE


class TSNE_wrapper(TSNE):

    def __init__(self, angle=0.5, early_exaggeration=12.0, init='random',
                 learning_rate=200.0, method='barnes_hut', metric='euclidean',
                 min_grad_norm=1e-07, n_components=2, n_iter=1000,
                 n_iter_without_progress=300, n_jobs=None,
                 perplexity=30.0, random_state=None, verbose=0):

        self.angle = angle
        self.early_exaggeration = early_exaggeration
        self.init = init
        self.learning_rate = learning_rate
        self.method = method
        self.metric = metric
        self.min_grad_norm = min_grad_norm
        self.n_components = n_components
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.n_jobs = n_jobs
        self.perplexity = perplexity
        self.random_state = random_state
        self.verbose = verbose

    def transform(self, X):
        return TSNE().fit_transform(X)

    def fit(self, X):
        return TSNE().fit(X)

'''
Computing the trustworthiness category by category
'''

from sklearn.manifold import trustworthiness


def groups_trustworthiness(df, df_proj, ser_clust, n_neighbors=5):
    
    gb_clust = df.groupby(ser_clust)
    tw_clust, li_clust = [], []
    for n_clust, ind_sub_df in gb_clust.groups.items():
        li_clust.append(n_clust)
        tw_clust.append(trustworthiness(df.loc[ind_sub_df],
                                        df_proj.loc[ind_sub_df],
                                        n_neighbors=n_neighbors, metric='euclidean'))
    ser = pd.Series(tw_clust,
                    index=li_clust,
                    name='tw')
    return ser


'''Computes the projection of the observations of X on the two first axes of
a transformation (PCA, UMAP or t-SNE)
The center option (clustering model needed) allows to project the centers
on the two axis for further display, and to return the fitted model
NB: if the model wa already fitted, does not refit.'''

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.utils.validation import check_is_fitted


def prepare_2D_axes(X, y, ser_clust=None, proj=['PCA', 'UMAP', 't-SNE'],  # ['PCA', 'UMAP', 't-SNE']
                    model=None, centers_on=False, random_state=14):

    dict_proj = dict()

    if centers_on:  # Compute and include the centers in the points
        if model is not None:
            model = model.fit(X, y) if not check_is_fitted(model) else model
            # ### all clusterers don't have .cluster_centers method -> changed
            # centers = model.cluster_centers_ 
            # ind_centers = ["clust_" + str(i) for i in range(centers.shape[0])]
            # centers_df = pd.DataFrame(centers,
            #                           index=ind_centers,
            #                           columns=X.columns)
            # all clusterers don't have .predict/labels_ method -> changed
            if hasattr(model, 'labels_'):
                clust = model.labels_
            else:
                clust = model.predict(X)
        else:
            clust = ser_clust
        # calculation of centers
        centers_df = X.assign(clust=clust).groupby('clust').mean()
        X = X.append(centers_df)

    # Projection of all the points through the transformations

    # PCA
    if 'PCA' in proj:
        pca = PCA(n_components=2, random_state=random_state)
        df_proj_PCA_2D = pd.DataFrame(pca.fit_transform(X),
                                      index=X.index,
                                      columns=['PC' + str(i) for i in range(2)])
        dict_proj = dict({'PCA': df_proj_PCA_2D})

    # UMAP
    if 'UMAP' in proj:
        umap = UMAP(n_components=2, random_state=random_state)
        df_proj_UMAP_2D = pd.DataFrame(umap.fit_transform(X),
                                       index=X.index,
                                       columns=['UMAP' + str(i) for i in range(2)])
        dict_proj = dict({'UMAP': df_proj_UMAP_2D})

    # t-SNE
    if 't-SNE' in proj:
        tsne = TSNE(n_components=2, random_state=random_state)
        df_proj_tSNE_2D = pd.DataFrame(tsne.fit_transform(X),
                                       index=X.index,
                                       columns=['t-SNE' + str(i) for i in range(2)])
        dict_proj = dict({'t-SNE': df_proj_tSNE_2D})

    # Separate the clusters centers from the other points if center option in on
    if centers_on:
        dict_proj_centers = {}
        for name, df_proj in dict_proj.items():
            dict_proj_centers[name] = dict_proj[name].loc[centers_df.index]
            dict_proj[name] = dict_proj[name].drop(index=centers_df.index)
        return dict_proj, dict_proj_centers, model
    else:
        return dict_proj


''' Plots the points on two axis (projection choice available : PCA, UMAP, t-SNE)
with clusters coloring if model available (grey if no model given).
NB: if the model wa already fitted, does not refit.'''

import seaborn as sns


def plot_projection(X, y, model=None, plot_only_idx=None,
                    ser_clust=None, proj='PCA',
                    tw_n_neigh=5, title=None, bboxtoanchor=None,
                    figsize=(5, 3), size=1, palette='tab10', show_centers=True,
                    dict_proj=None, dict_proj_centers=None,
                    legend_on=False, fig=None, ax=None, random_state=14):

    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax

    # a1 - if model : computes clusters, clusters centers and plot with colors
    if model is not None:
        
        # compute the list of 2D coordinate of the points if not in arguments
        if dict_proj is None:
        
            # Computes the axes for projection with centers
            # (uses fitted model if already fitted)
            dict_proj, dict_proj_centers, model = prepare_2D_axes(X, y,
                                                                  proj=[proj],
                                                                  model=model,
                                                                  centers_on=True,
                                                                  random_state=random_state)

            # ...or using model already fitted in prepare_2D_axes to get it
            # all clusterers don't have .predict/labels_ method -> changed
            if hasattr(model, 'labels_'):
                clust = model.labels_
            else:
                clust = model.predict(X)
            ser_clust = pd.Series(clust,
                                  index=X.index,
                                  name='Clust')
        
    # a2 - if no model but ser_clust is given, plot with colors
    elif ser_clust is not None:
        
        # Computes the axes for projection
        dict_proj, dict_proj_centers, _ = \
            prepare_2D_axes(X, y, ser_clust=ser_clust, proj=[proj],
                            model=None, centers_on=True,
                            random_state=random_state)

        n_clust = ser_clust.nunique()
        colors = sns.color_palette(palette, n_clust).as_hex()

    # Computing the global trustworthiness
    if tw_n_neigh is not None:
        trustw = trustworthiness(X, dict_proj[proj],
                                 n_neighbors=tw_n_neigh, metric='euclidean')
        trustw = "{:.2f}".format(trustw)
        # Computing the trustworthiness category by category
        ser_tw_clust = groups_trustworthiness(X, dict_proj[proj], ser_clust,
                                              n_neighbors=tw_n_neigh)
#         tw_title = ser_tw_clust
    else:
        trustw = ''

    # b1 - if ser_clust exists (either calculated from model or given)
    if ser_clust is not None:

        # Showing the points, cluster by cluster
        # for i in range(n_clust):
        for i, name_clust in enumerate(ser_clust.unique()):
            ind = ser_clust[ser_clust == name_clust].index
            
            # plot only a selection of points
            if plot_only_idx is not None:
                ind = [i for i in ind if i in plot_only_idx]
                
            ax.scatter(dict_proj[proj].loc[ind].iloc[:, 0],
                       dict_proj[proj].loc[ind].iloc[:, 1],
                       s=size, alpha=0.7, c=colors[i], zorder=1)
            
            if show_centers:
                if tw_n_neigh is not None:
                    tw_title = "{:.2f}".format(ser_tw_clust[name_clust])
                else:
                    tw_title = ''
                # Showing the clusters centers
                ax.scatter(dict_proj_centers[proj].iloc[:, 0].loc[name_clust],
                           dict_proj_centers[proj].iloc[:, 1].loc[name_clust],
                           marker='o', c=colors[i], alpha=0.7, s=150,
                           edgecolor='k',
                           label="{}: {} | tw={}".format(i, name_clust, tw_title),
                           zorder=10)  # for the labels only
                # Showing the clusters centers labels (number)
                ax.scatter(dict_proj_centers[proj].iloc[:, 0].loc[name_clust],
                           dict_proj_centers[proj].iloc[:, 1].loc[name_clust],
                           marker=r"$ {} $".format(i),  #
                           c='k', alpha=1, s=70, zorder=100)
            
            if legend_on:
                plt.legend().get_frame().set_alpha(0.3)
            if bboxtoanchor is not None:
                plt.legend(bbox_to_anchor=bboxtoanchor)
            else: 
                plt.legend()

    # b2 - if no ser_clust: only plot points in grey
    else:
        # Computes the axes for projection without centers
        dict_proj = prepare_2D_axes(X, y,
                                    proj=[proj],
                                    centers_on=False,
                                    random_state=random_state)
        
        # plot only a selection of points
        if plot_only_idx is not None:
            ind = [i for i in dict_proj[proj].index if i in plot_only_idx]
        else:
            ind = list(dict_proj[proj].index)

        # Plotting the points in grey
        ax.scatter(dict_proj[proj].iloc[:, 0].loc[ind],
                   dict_proj[proj].iloc[:, 1].loc[ind],
                   s=size, alpha=0.7, c='grey')

#     title = "Projection: " + proj if title is None else title
    if title is not None:
        ax.set_title(title + f"\n(trustworthiness: {trustw})",
                     fontsize=12, fontweight='bold')
    ax.set_xlabel('proj ax 1'), ax.set_ylabel('proj ax 2')


'''Plotting one given score for all or a selection of the hyperparameters tested with Optunasearchcv
Can choose the aggregation function for the score on all other parameters
option for using pooled standard deviation in stead of regular std'''

from optuna.distributions import LogUniformDistribution 


def plot_hyperparam_tuning_optuna(gs, grid_params, params=None, score='score', figsize=(12, 7),
                                  n_cols=4, title=None):

    if params is not None:
        grid_params = {k: v for (k, v) in grid_params.items() if k in params}

    df = pd.DataFrame(gs.trials_dataframe())

    results = ['user_attrs_mean_train_score',
               'user_attrs_mean_test_score',
               'user_attrs_std_train_score',
               'user_attrs_std_test_score']

    fig = plt.figure(figsize=figsize)

    li_type = ['int64', 'float64', 'datetime64[ns]',
               'datetime64[ns]', 'timedelta64[ns]']

    li_param = [param_name for param_name in grid_params.keys() \
                if df['params_'+param_name].dtype in li_type]

    n_tot = len(li_param)
    n_rows = (n_tot // n_cols) + ((n_tot % n_cols) > 0) * 1

    for idx, param_name in enumerate(li_param, 1):

        if type(grid_params[param_name]) == LogUniformDistribution:
            log_scale_on = True
        else:
            log_scale_on = False

        if df['params_'+param_name].dtype in li_type:

            ax = fig.add_subplot(n_rows, n_cols, idx)

            grouped_df = df.groupby('params_'+param_name)[results].agg(np.mean)
            lw = 2
            param_range = list(grouped_df.index)

            ax.plot(param_range, grouped_df['user_attrs_mean_train_score'], label="Train (CV)",
                    color="darkorange", marker='o', ms=3, lw=lw)
            ax.fill_between(param_range,
                            grouped_df['user_attrs_mean_train_score'] - grouped_df['user_attrs_std_train_score'],
                            grouped_df['user_attrs_mean_train_score'] + grouped_df['user_attrs_std_train_score'],
                            alpha=0.2, color="darkorange", lw=lw)

            ax.plot(param_range, grouped_df['user_attrs_mean_test_score'],
                    label="Test (CV)", marker='o', ms=3, color="navy", lw=lw)
            ax.fill_between(param_range,
                            grouped_df['user_attrs_mean_test_score'] - grouped_df['user_attrs_std_test_score'],
                            grouped_df['user_attrs_mean_test_score'] + grouped_df['user_attrs_std_test_score'],
                            alpha=0.2, color="navy", lw=lw)

            if log_scale_on:
                ax.set_xscale('log')

            ax.set_xlabel(param_name, fontsize=12)
            ymin, ymax = ax.get_ylim()
            # axes[idx].set_ylim(ymin, 0*ymax)
        else:
            continue

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc=1, ncol=1, fontsize=12)

    #     fig.subplots_adjust(bottom=0.25, top=0.85, right=0.97)  
    title = 'Hyperparameters tuning' if title is None else title
    fig.suptitle(title, x=0.4, y=0.95, fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


# When searching for 2 best hyperparameters with gscv : plotting a heatmap of mean_test_score(cv)
# the score displayed for each cell is the one for the best other parameters.

from matplotlib.ticker import FormatStrFormatter


def plot_2D_hyperparam_tuning_optuna(scv, params=None, score=None, annot=False, cmap=None,
                                     title=None, fmt='.4g', ax=None):
    
    df_scv = pd.DataFrame(scv.trials_dataframe())

    col_not_numb = df_scv.select_dtypes(exclude=[np.number])
    col_obj = col_not_numb.columns[col_not_numb.columns.str.contains('params_')]
    df_scv[col_obj] = df_scv[col_obj].applymap(str)

    if params:  # example: params=['enet__alpha', 'enet__l1_ratio']
        params_scv = ['params_'+p for p in params]
    else:
        params_scv = df_scv.columns[df_scv.columns.str.contains('params_')].to_list()
        if len(params_scv)!=2:
            print('WARNING : parameters to display were guessed,\
                provide the params parameter with 2 parameters')
            params_scv = params_scv[0:2]
        else:
            params_scv = params_scv
    # Not suitable for 3D viz : takes the max among all other parameters !!!
    max_scores = df_scv.groupby(params_scv).agg(lambda x: max(x))
    sns.heatmap(max_scores.unstack()['user_attrs_mean_test_score'],
                annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    if title is None:
        title = score
    ax = plt.gca()
    plt.gca().set_title(title)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

'''
Generate 3 plots: the test and training learning curve, the training
samples vs fit times curve, the fit times vs score curve.
'''

from sklearn.model_selection import ShuffleSplit
from matplotlib.lines import Line2D
from sklearn.model_selection import learning_curve
import dill


def plot_learning_curve(name_reg, estimator, X, y, ylim=None, cv=None,
                        scoring='neg_root_mean_squared_error', score_name="Score",
                        file_name=None, dict_learn_curves=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                        c='r', axes=None, title=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))  # plt.subplots(0, 3, figsize=(12, 3))

    if dict_learn_curves is None:
        dict_learn_curves = {}

    # If model with the same name already in dict_models, just takes existing model
    if dict_learn_curves.get(name_reg, np.nan) is not np.nan:
        print('-----Learning curve already exists - taking existing learning curve')
        train_sizes, train_scores, test_scores, fit_times = \
            list(zip(*list(dict_learn_curves[name_reg].items())))[1]
    
    # Else computes new model and add to the dictionnary, and then to the pickle
    else:
        print('----- Learning curve not existing - computing...')

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes, scoring=scoring,
                           return_times=True)
        
        d_ = {'train_sizes': train_sizes,
              'train_scores': train_scores,
              'test_scores': test_scores,
              'fit_times': fit_times}
        dict_learn_curves[name_reg] = d_
        if file_name is not None:
            with open(file_name, "wb") as f:
                dill.dump(dict_learn_curves, f)
            print("-----...learning curve dumped")
        else:
            print("-----...no file name to dump the learning curves dictionary")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.15,
                         color=c)
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.3,
                         color=c)
    axes[0].plot(train_sizes, train_scores_mean, 'o-', mfc=None, ms=3,
                 color=c, ls='dashed', label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', ms=3,
                 color=c, ls='solid',
                 label="Cross-validation score")
    axes[0].set_title("Learning curves")
    if ylim is not None: axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel(score_name)
    
    cust_leg = [Line2D([0], [0], color='k', ls='dashed', lw=2),
                Line2D([0], [0], color='k', ls='solid', lw=2)]
    axes[0].legend(cust_leg, ['Train (CV)', 'Test (CV)'], loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean,
                 'o-', color=c, ms=3)
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, color=c, alpha=0.2)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean,
                 'o-', ms=3, color=c, label=name_reg)
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, color=c, alpha=0.2)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel(score_name)
    axes[2].set_title("Performance of the model")
    if ylim is not None:
        axes[2].set_ylim(*ylim)
    axes[2].legend(loc=2, prop={'size': 10})  # bbox_to_anchor = (0.2,1.1), ncol=4

    plt.gcf().set_facecolor('w')
    if title is not None:
        plt.gcf().suptitle(title, fontsize=15, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        plt.tight_layout()
    return dict_learn_curves


'''
Takes a confusion matrix (best if diagonal maximized) with true categories as
indices and clusters as columns (can be obtained using the function
'confusion_matrix_clust', which ensures the diagonal values are maximum i.e.
the best bijective correponding cat-clut pairs have been found),
then plot the sankey confusion matrix.
Use the option static to plot a static image of the original interactive graph.

NB: the code below needs to be run if you are on colab

    # in order to get orca to be installed in colab (to display static plotly graphs)
    !pip install plotly>=4.0.0
    !wget https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage -O /usr/local/bin/orca
    !chmod +x /usr/local/bin/orca
    !apt-get install xvfb libgtk2.0-0 libgconf-2-4
'''

import plotly.graph_objects as go
from IPython.display import Image
import seaborn as sns


def plot_sankey_confusion_mat(cm, static=False, figsize=(2, 1.7),
                              font_size=14, scale=1, title=None,
                              palette='tab10'):

    n_cat = cm.shape[0]
    n_clust = cm.shape[1]
    source = np.array([n_clust*[i] for i in range(n_cat)]).ravel()
    target = np.array([[i] for i in range(n_cat, n_clust+n_cat)]*n_cat).ravel()
    value = cm.values.ravel()
    nodes_lab = list(cm.index)+list(cm.columns)
    alpha_nodes, alpha_links = 0.7, 0.3
    my_pal = sns.color_palette(palette, max(cm.shape))
    pal_nodes_cat = list([f'rgba({r},{g},{b},{alpha_nodes})'
                          for r, g, b in my_pal[:n_cat]])
    pal_nodes_clust = list([f'rgba({r},{g},{b},{alpha_nodes})'
                            for r, g, b in my_pal[:n_clust]])
    nodes_colors = (pal_nodes_cat + pal_nodes_clust)

    pal_links = list([f'rgba({r},{g},{b},{alpha_links})' for r, g, b in my_pal[:n_cat]])
    dict_links_colors = dict(zip(range(n_cat), pal_links))
    links_colors = np.vectorize(dict_links_colors.__getitem__)(source)

    # Prepare the graph
    fig = go.Figure(data=[go.Sankey(node=dict(pad=15,
                                              thickness=20,
                                              line=dict(color="black",
                                                        width=0.5),
                                              label=nodes_lab,
                                              color=nodes_colors),
                                    link=dict(source=source,
                                              target=target,
                                              value=value,  # label = ,
                                              color=links_colors))])
    # title
    fig.update_layout(title_text=title, font_size=font_size)
    if static:
        w, h = figsize
        img_bytes = fig.to_image(format="png", width=w, height=h, scale=scale)
        # Image(img_bytes)
        return img_bytes
    else:
        fig.show()


'''
For each of the variables in 'main_cols', plot a boxplot of the whole data (X_all),
then a swarmplot of the 20 nearest neighbors' variable values (X_neigh),
and the values of the applicant customer (X_cust)
'''


def plot_boxplot_var_by_target(X_all, y_all, X_neigh, y_neigh, X_cust,
                               main_cols, figsize=(15, 4)):

    df_all = pd.concat([X_all[main_cols], y_all.to_frame(name='TARGET')], axis=1)
    df_neigh = pd.concat([X_neigh[main_cols], y_neigh.to_frame(name='TARGET')], axis=1)
    df_cust = X_cust[main_cols].to_frame('values').reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)

    # random sample of customers of the train set
    df_melt_all = df_all.reset_index()
    df_melt_all.columns = ['index'] + list(df_melt_all.columns)[1:]
    df_melt_all = df_melt_all.melt(id_vars=['index', 'TARGET'], # SK_ID_CURR
                                   value_vars=main_cols,
                                   var_name="variables",
                                   value_name="values")
    sns.boxplot(data = df_melt_all, x='variables', y='values', hue='TARGET', linewidth=1,
                width=0.4, palette=['tab:green', 'tab:red'], showfliers=False, saturation=0.5,
                ax=ax)

    # 20 nearest neighbors
    df_melt_neigh = df_neigh.reset_index()
    df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
    df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],  # SK_ID_CURR
                                       value_vars=main_cols,
                                       var_name="variables",
                                       value_name="values")
    sns.swarmplot(data = df_melt_neigh, x='variables', y='values', hue='TARGET', linewidth=1,
                  palette=['darkgreen', 'darkred'], marker='o', edgecolor='k', ax=ax)

    # applicant customer
    df_melt_cust = df_cust.rename(columns={'index': "variables"})
    sns.swarmplot(data=df_melt_cust, x='variables', y='values', linewidth=1, color='y',
                  marker='o', size=10, edgecolor='k', label='applicant customer', ax=ax)

    # legend
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h[:5])

    plt.xticks(rotation=20)
    plt.show()

    return fig

# '''permutation importance using sklearn '''
# from sklearn.inspection import permutation_importance

# def plot_perm_importance(model, name_reg, X, y, scoring='r2',
#                          dict_perm_imp=None, file_name=None, figsize=(12,3)):

#     # If model with the same name already in dict_models, just takes existing model
#     if dict_perm_imp.get(name_reg, np.nan) is not np.nan:
#         print('-----Permutation importance - taking existing model')
#         ser = dict_perm_imp[name_reg]
#     # Else computes new model and add to the dictionnary, and then to the pickle
#     else:
#         print('-----Permutation importance not existing - computing...')
#         results = permutation_importance(model, X, y, scoring=scoring)
#         ser = pd.Series(results.importances_mean, index = X.columns)
        
#     dict_perm_imp[name_reg] = ser

#     with open(file_name, "wb") as f:
#         dill.dump(dict_perm_imp, f)
#     print("-----...model dumped")

#     fig, ax = plt.subplots()
#     ser.sort_values(ascending=False).plot.bar(color='grey');
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right" )
#     fig.set_size_inches(figsize)
#     plt.show()

#     return dict_perm_imp
# '''Plotting the feature importance of a model'''


# def plot_model_feat_imp(name_reg, model, figsize=(15, 3)):
#     # Getting the names of the transformed columns
#     step_ct = model.named_steps['preproc'].named_steps['cust_trans']
#     col_names = step_ct.get_feature_names()
#     # Getting the list of the coefficients (wether usinf 'coef_' or 'feature_importances')
#     step_reg = model.named_steps[name_reg]
#     if hasattr(step_reg, "coef_"):
#         col_coefs = step_reg.coef_
#     elif hasattr(step_reg, "feature_importances_"):
#         col_coefs = step_reg.feature_importances_
#     else:
#         print("ERROR: This regressor has no 'coef_' or 'feature_importances_' attribute")
#     nb_feat = col_coefs.size
#     ser = pd.Series(col_coefs, index=col_names[:nb_feat])

#     fig, ax = plt.subplots()
#     ser.sort_values(ascending=False).plot.bar(color='red');
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
#     fig.set_size_inches(figsize)
#     plt.show()


"""
Affiche les valeurs des clients en fonctions de deux paramètres en montrant leur classe
Compare l'ensemble des clients par rapport aux plus proches voisins et au client choisi.
X = données pour le calcul de la projection
ser_clust = données pour la classification des points (2 classes) (pd.Series)
n_display = items à tracer parmi toutes les données
plot_highlight = liste des index des plus proches voisins
X_cust = pd.Series des data de l'applicant customer
figsize=(10, 6) 
size=10
fontsize=12
columns=None : si None, alors projection sur toutes les variables, si plus de 2 projection
"""

from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE


def plot_scatter_projection(X, ser_clust, n_display, plot_highlight, X_cust,
                            figsize=(10, 6), size=10, fontsize=12, columns=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    X_all = pd.concat([X, X_cust], axis=0)
    ind_neigh = list(plot_highlight.index)
    customer_idx = X_cust.index[0]

    columns = X_all.columns if columns is None else columns

    if len(columns) == 2:
        # if only 2 columns passed
        df_data = X_all.loc[:, columns]
        ax.set_title('Two features compared', fontsize=fontsize + 2, fontweight='bold')
        ax.set_xlabel(columns[0], fontsize=fontsize)
        ax.set_ylabel(columns[1], fontsize=fontsize)

    elif len(columns) > 2:
        # if more than 2 columns passed
        # Compute T-SNE projection
        tsne = TSNE(n_components=2, random_state=14)
        df_proj = pd.DataFrame(tsne.fit_transform(X_all),
                               index=X_all.index,
                               columns=['t-SNE' + str(i) for i in range(2)])
        trustw = trustworthiness(X_all, df_proj, n_neighbors=5, metric='euclidean')
        trustw = "{:.2f}".format(trustw)
        ax.set_title(f't-SNE projection (trustworthiness={trustw})',
                     fontsize=fontsize + 2, fontweight='bold')
        df_data = df_proj
        ax.set_xlabel("projection axis 1", fontsize=fontsize)
        ax.set_ylabel("projection axis 2", fontsize=fontsize)

    else:
        # si une colonne seulement
        df_data = pd.concat([X_all.loc[:, columns], X_all.loc[:, columns]], axis=1)
        ax.set_title('One feature', fontsize=fontsize + 2, fontweight='bold')
        ax.set_xlabel(columns[0], fontsize=fontsize)
        ax.set_ylabel(columns[0], fontsize=fontsize)

    # Showing points, cluster by cluster
    colors = ['green', 'red']
    for i, name_clust in enumerate(ser_clust.unique()):
        ind = ser_clust[ser_clust == name_clust].index

        if n_display is not None:
            display_samp = random.sample(set(list(X.index)), 200)
            ind = [i for i in ind if i in display_samp]
        # plot only a random selection of random sample points
        ax.scatter(df_data.loc[ind].iloc[:, 0],
                   df_data.loc[ind].iloc[:, 1],
                   s=size, alpha=0.7, c=colors[i], zorder=1,
                   label=f"Random sample ({name_clust})")
        # plot nearest neighbors
        ax.scatter(df_data.loc[ind_neigh].iloc[:, 0],
                   df_data.loc[ind_neigh].iloc[:, 1],
                   s=size * 5, alpha=0.7, c=colors[i], ec='k', zorder=3,
                   label=f"Nearest neighbors ({name_clust})")

    # plot the applicant customer
    ax.scatter(df_data.loc[customer_idx:customer_idx].iloc[:, 0],
               df_data.loc[customer_idx:customer_idx].iloc[:, 1],
               s=size * 10, alpha=0.7, c='yellow', ec='k', zorder=10,
               label="Applicant customer")

    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.legend(prop={'size': fontsize - 2})

    return fig


'''
Plots ROC Curve of a model and the baseline of a model
'''

def plot_roc_curve_(y, y_pred_proba, name_model, color='b', fig=None, ax=None):
    
    
    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax
    
    # tests if a baseline is already existing (color black)
    leg = ax.get_legend()
    if leg is not None:
        baseline_on = not ('k' in [line.get_c() for line in leg.get_lines()])
    else:
        baseline_on = True

    ### BASELINE
    if baseline_on:
        # generate a no skill prediction (majority class)
        y_prob_baseline = [0 for _ in range(len(y))]
        # computes auc
        auc_baseline = roc_auc_score(y, y_prob_baseline)
        # calculate the curves
        fpr_baseline, tpr_baseline, _ = roc_curve(y, y_prob_baseline)
        # plot baseline
        ax.plot(fpr_baseline, tpr_baseline, linestyle='--', color='k',
                label='Baseline'+' (AUC={:.2f})'.format(auc_baseline))
        
    ### MODEL
    # calculate roc auc scores
    auc_model = roc_auc_score(y.values, y_pred_proba)
    # calculate the curves
    fpr_model, tpr_model, thresh = roc_curve(y, y_pred_proba)
    ax.plot(fpr_model, tpr_model, linestyle='-', color=color,
            marker=None, label=name_model+' (AUC={:.2f})'.format(auc_model))
    
    # axis labels
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title("ROC Curve")
    ax.legend()
    
    return auc_model
#         return plt

    
'''
Plots PR Curve of a model and the baseline of a model
'''    
       
def plot_pr_curve_(y, y_pred_proba, name_model, color='r', fig=None, ax=None):
    
    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax
    
    # tests if a baseline is already existing (color black)
    leg = ax.get_legend()
    if leg is not None:
        baseline_on = not ('k' in [line.get_c() for line in leg.get_lines()])
    else:
        baseline_on = True
        
    if baseline_on:
        ### BASELINE
        # generate a no skill prediction (majority class)
        y_prob_baseline = [0 for _ in range(len(y))]
        # calculate pr auc scores
        prauc_baseline = average_precision_score(y, y_prob_baseline)
        # calculate pr curves
        prec_baseline, rec_baseline, _ = precision_recall_curve(y, y_prob_baseline)
        # plot the precision-recall curves
        baseline = len(y[y==1]) / len(y)
        ax.plot([0, 1], [baseline, baseline], color='k', linestyle='--',
                label='Baseline'+' (AUC={:.2f})'.format(prauc_baseline))
    
    ### MODEL
    # calculate pr auc scores
    prauc_model = average_precision_score(y.values, y_pred_proba) # auc of the precision recall is avg precision !!
    # calculate pr curves
    prec_model, rec_model, thresh = precision_recall_curve(y, y_pred_proba)
    # plot the precision-recall curves
    ax.plot(rec_model, prec_model, marker=None, color=color,
            label=name_model+' (AUC={:.2f})'.format(prauc_model))
    
    # axis labels
    ax.set_xlabel('Recall (TP/Ptrue)')
    ax.set_ylabel('Precision (TP/Ppred)')
    ax.set_title("Precision Recall Curve")
    ax.legend()
    
    return prauc_model
#     return plt

'''
Plots the roc curve, the precision recal curve of a model for the probability of a class n.
'''
    
def plot_roc_pr_threshold_tuning(y, X, name_model, model, n=1,
                                 optim_score='gain_level', n_thresh=30,
                                 figsize=(15,7), palette='tab10', verbose=False,
                                 li_scores=['accuracy','bal_accuracy','recall_score','precision_score',
                                            'f1_score','f5_score','cohen_kapp','mcc_score','gain_level'],
                                title=''):

    # compute predict (default thresh 0.5)
    y_pred = model.predict(X)

    # get the probability of the model for class n
    y_pred_proba = model.predict_proba(X)
    y_pred_proba_n = y_pred_proba[:,n]


    fig = plt.figure(figsize=figsize)

    # plot roc curve
    ax1 = fig.add_subplot(2,2,1)
    plot_roc_curve_(y, y_pred_proba_n, name_model, 'r', fig, ax1)
    
    # plot pr curve
    ax2 = fig.add_subplot(2,2,2)
    plot_pr_curve_(y, y_pred_proba_n, name_model, 'b', fig, ax2)
    

    # ##### CURVE OF CHOOSEN SCORE as a function of threshold

    
    # Computes all the scores for some thresholds (column name)
    df_res = pd.DataFrame()
    for i, th in enumerate(np.linspace(0, 1, n_thresh), 1):

        # set threshold of the model for class n
        y_pred_proba_n_th = (y_pred_proba[:,n] >= th).astype(bool)
        # store into a dataframe
        df_res = model_scores(y, y_pred_proba_n_th, y_pred_proba_n,
                              th, df_res, li_scores)
        if verbose:
            print('\r', str(i)+f'/{n_thresh}')

    df_res = df_res.T.reset_index().rename(columns={'index': 'threshold'})
    score_max = df_res[optim_score].max()
    thresh_max = df_res.loc[df_res[optim_score].argmax(), 'threshold']

    ax3 = fig.add_subplot(2,1,2)
    
    my_pal = sns.color_palette(palette, len(li_scores))
#     my_pal = ['b', 'r', 'g', 'purple', 'orange',
#              'pink', 'brown', 'yellow', 'grey']
    
    for sc, col in zip(li_scores, my_pal):
        df_res.plot('threshold', sc, marker=None, ax=ax3, label=sc, color=col)
    
    ax3.set_xlabel('threshold')
    ax3.set_ylabel('score')
    ax3.text(thresh_max*1.02 , score_max*0.7,
             '(thresh={:.2f},\nscore={:.2f})'.format(thresh_max, score_max), fontsize=12)#, fontweight='bold')
    ax3.vlines(thresh_max, 0, 1, color='k', linestyle='dashed')  
    ax3.plot(thresh_max, score_max, marker='o', color='k', linestyle=None)  
    ax3.set_title(title)
    ax3.legend(bbox_to_anchor=(1,1))
    
    # show the plot
    fig.suptitle(f'Threshold tuning for {name_model}', fontweight='bold')
    plt.grid()
#     plt.tight_layout(rect=[0,0,1,0.95])
    fig.subplots_adjust(hspace=0.45)
    plt.show()

    return thresh_max, score_max, df_res