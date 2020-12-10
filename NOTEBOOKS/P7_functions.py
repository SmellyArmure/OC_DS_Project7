# return a dataframe of missing values (total and percent)

def missing_data(data):
    
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100)\
                                .sort_values(ascending = False)
    return pd.concat([total, percent],
                     axis=1,
                     keys=['Total', 'Percent'])

# Plotting histograms of specified quantitative continuous columns of a dataframe and mean, median and mode values.

import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_histograms(df, cols, bins=30, figsize=(12,7), color = 'grey',
                    skip_outliers=False, thresh=3, n_cols=3, tight_layout=None,
                    sh_tit = 13):

    fig = plt.figure(figsize=figsize)
    n_tot = len(cols)
    n_rows = (n_tot//n_cols)+((n_tot%n_cols)>0)*1

    for i, c in enumerate(cols,1):
        mask_na = df[c].notna()
        ser = df[c].loc[mask_na]
        mask_isfin = np.isfinite(ser)
        ser = ser.loc[mask_isfin]

        ax = fig.add_subplot(n_rows,n_cols,i)
        if skip_outliers:
            mask_outl = np.abs(st.zscore(ser))<thresh
            ser = ser.loc[mask_outl]
        else:
            ser = df[c].loc[mask_na]
        ax.hist(ser, ec='k', bins=bins, color=color)
        title = c if len(c)<2*sh_tit else c[:sh_tit]+'...'+c[-sh_tit:]
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



'''
# Plotting histograms of specified quantitative continuous columns of a
dataframe in order to compare histograms of different categories.
'''

import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_cat_histograms(df, ind_tuple, cols, labels, colors, bins=30, figwidth=20,
                    skip_outliers=False, thresh=3, n_cols=3, tight_layout=True,
                    sh_tit = 13, filter_finite=True):

    df_0_ind, df_1_ind = ind_tuple
    n_tot = len(cols)
    n_rows = (n_tot//n_cols)+((n_tot%n_cols)>0)*1
    # figheight = 2*n_rows
    figsize = (figwidth, 3)

    # loop on each row
    for j, row in enumerate(range(n_rows), 0):

        fig = plt.figure(figsize=figsize) # height
        sub_cols = list(cols)[j*n_cols:(j+1)*n_cols]

        for i, col in enumerate(sub_cols, 1):

            ax = fig.add_subplot(1,n_cols,i)

            mask_na = df[col].notna()
            ser = df[col].loc[mask_na]
            if filter_finite:
                mask_isfin = np.isfinite(ser)
                ser = ser.loc[mask_isfin]
            if skip_outliers:
                mask_outl = np.abs(st.zscore(ser))<thresh
                ser = ser.loc[mask_outl]

            ind_0 = [i for i in ser.index if i in df_0_ind]
            ind_1 = [i for i in ser.index if i in df_1_ind]

            df.loc[ind_0, col].hist(label='repaid', alpha=0.5, ec='k',
                                            bins=30, color='green',ax=ax, density=True)

            df.loc[ind_1, col].hist(label='not repaid', alpha=0.5, ec='k',
                                            bins=30, color='red',ax=ax, density=True)

            plt.legend()

            title = col if len(col)<2*sh_tit else\
                                     col[:sh_tit]+'...'+col[-sh_tit:]
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
                                 sh_tit = 20, shorten_label=10):

    n_tot = len(cols)
    n_rows = (n_tot//n_cols)+((n_tot%n_cols)>0)*1
    figheight = figwidth/aspect
    figsize = (figwidth, figheight)

    # loop on each row
    for j, row in enumerate(range(n_rows), 0):

        fig = plt.figure(figsize=figsize) # height
        sub_cols = list(cols)[j*n_cols:(j+1)*n_cols]

        for i, col in enumerate(sub_cols, 1):

            ax = fig.add_subplot(1,n_cols,i)

            ct = pd.crosstab(df[col_target], df[col])
            sns.heatmap(ct/ct.sum(axis=0), cmap='magma', annot=True, fmt='.2f')

            title = col if len(col)<2*sh_tit else\
                                     col[:sh_tit]+'...'+col[-sh_tit:]

            ax.axes.get_xaxis().get_label().set_visible(False)
            ax.set_title(title)
            ax.title.set_fontweight('bold')

            if shorten_label:
                thr = int(shorten_label)
                lab_x = [item.get_text() for item in ax.get_xticklabels()]
                short_lab_x = [s[:thr]+'...'+s[-thr:] if len(s)>2*thr else s for s in lab_x]
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

    n_sh=100 if n_sh is None else n_sh
    val_cnts = df[col_cat].value_counts()
    sns.boxplot(x=col_cat, y=col_val, data=df, ax=ax)
    if log_scale:
        plt.yscale('log')
    xticklab = [item.get_text()[0:n_sh-1]+f'... ({val_cnts[item.get_text()]})' if len(item.get_text())>n_sh else \
                item.get_text()+f' ({val_cnts[item.get_text()]})' for item in plt.gca().get_xticklabels()]
    plt.gca().axes.set_xticklabels(xticklab)
    plt.xticks(rotation=45, ha='right')
    return val_cnts


# Plotting bar plots of the main categorical columns

import pandas as pd
import matplotlib.pyplot as plt

def plot_barplots(df, cols, figsize=(12,7), n_cols=3, shorten_label=False,
                  color='grey'):

    n_tot = len(cols)
    n_rows = (n_tot//n_cols)+((n_tot%n_cols)>0)*1

    fig = plt.figure(figsize=figsize)
    for i, c in enumerate(cols,1):
        ax = fig.add_subplot(n_rows,n_cols,i)
        ser = df[c].value_counts()
        n_cat = ser.shape[0]
        if n_cat>15:
            ser[0:15].plot.bar(color=color,ec='k', ax=ax)
        else:
            ser.plot.bar(color=color,ec='k',ax=ax)
        ax.set_title(c[0:20]+f' ({n_cat})', fontweight='bold')
        labels = [item.get_text() for item in ax.get_xticklabels()]

        if shorten_label:
            thr = int(shorten_label)
            lab_x = [item.get_text() for item in ax.get_xticklabels()]
            short_lab_x = [s[:thr]+'...'+s[-thr:] if len(s)>2*thr else s for s in lab_x]
            ax.axes.set_xticklabels(short_lab_x)

        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()


'''
simple bar plot with shortening label option
'''
from matplotlib import colors

def plot_simple_barplot(x, y, x_lab=None, y_lab=None, title=None,
                        shorten_label=15, figsize=(25,3), color=None,
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
        short_lab_x = [s[:thr]+'...'+s[-thr:] if len(s)>2*thr else s for s in lab_x]
        ax.axes.set_xticklabels(short_lab_x)

    if annotate:
        for i, val in enumerate(y):
            ymin, ymax = ax.get_ylim()
            plt.text(i-0.25, val+(ymax-ymin)*1/30, f'{val}')#,fontsize=10)
            ax.set(ylim=(ymin, ymax+(ymax-ymin)*0.005))

    plt.grid()

# Plotting heatmap (2 options available, rectangle or triangle )
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



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn import metrics

# Data Preprocessing for quantitative and categorical data with encoding options
def data_preprocessing(df, var_model, var_target, enc_strat_cat='label'):
    ## Data Processing
    df_train = df[var_model+[var_target]].copy('deep')
    if df[var_model].isna().sum().sum()!=0 :
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
        X = df_train.iloc[:,:-1].values
        y = df_train.iloc[:,-1].values
        return X, y

def naive_model_compare_r2(X_tr, y_tr, X_te, y_te, y_pr):
    # Model
    print('--- model: {:.3}'.format(metrics.r2_score(y_te, y_pr)))
    # normal random distribution
    y_pr_rand = np.random.normal(0,1, y_pr.shape)
    print('--- normal random distribution: {:.3}'\
          .format(metrics.r2_score(y_te, y_pr_rand)))
    # dummy regressors
    for s in ['mean', 'median']:
        dum = DummyRegressor(strategy=s).fit(X_tr, y_tr)
        y_pr_dum = dum.predict(X_te)
        print('--- dummy regressor ('+ s +') : r2_score={:.3}'\
              .format(metrics.r2_score(y_te, y_pr_dum)))

def naive_model_compare_acc_f1(X_tr, y_tr, X_te, y_te, y_pr, average='weighted'):
    print('ooooooo CLASSIFICATION METRICS oooooooo')
    def f1_prec_recall(yte, ypr):
        prec = metrics.precision_score(yte, ypr, average=average)
        rec = metrics.recall_score(yte, ypr, average=average)
        f1 = metrics.f1_score(yte, ypr, average=average)
        return [f1, prec, rec]
    # Model
    print('--- model: f1={:.3}, precision={:.3}, recall={:.3}'\
                                             .format(*f1_prec_recall(y_te, y_pr)))
    # Dummy classifier
    for s in ['stratified','most_frequent','uniform']:
        dum = DummyClassifier(strategy=s).fit(X_tr, y_tr)
        y_pr_dum = dum.predict(X_te)
        print('--- dummy class. ('+ s\
              +'): f1={:.3}, precision={:.3}, recall={:.3}'\
                                             .format(*f1_prec_recall(y_te, y_pr_dum)))

def plot_hist_pred_val(y_te, y_pr, y_pr_, bins=150, xlim=(0,20), short_lab=False):
    # Plotting dispersion of data to be imputed
    bins = plt.hist(y_te, alpha=0.5, color='b', bins=bins, density=True,
                    histtype='step', lw=3, label='y_te (real val. from test set)')[1]
    ax=plt.gca()
    ax.hist(y_pr, alpha=0.5, color='g', bins=bins, density=True,
            histtype='step', lw=3, label='y_pr (pred. val. from test set)');
    ax.hist(y_pr_, alpha=0.5, color='r', bins=bins, density=True,
            histtype='step', lw=3, label='y_pr_ (pred. val. to be imputed)');
    ax.set(xlim=xlim)
    plt.xticks(rotation=45, ha='right')
    plt.draw()
    if short_lab:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        short_labels = [s[0:7]+'.' if len(s)>7 else s for s in labels]
        ax.axes.set_xticklabels(short_labels)
    ax.legend(loc=1)
    plt.title("Frequency of values", fontweight='bold', fontsize=12)
    plt.gcf().set_size_inches(6,2)
    plt.show()

# Works for both quantitative (knnregressor)
# and categorical (knnclassifier) target features

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
    
    if df[var_target].isna().sum()==0:
        print('ERROR: Nothing to impute (target column already filled)')
        return None, None
    else :
        if df[var_target].dtype =='object':
            # knn classifier
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            model = dict_models[n_model][1]
            gsCV = GridSearchCV(model,
                            {}, #'n_neighbors': [5]
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
            print("ERROR: dtype of target feature unknown")
        ## Data Preprocessing
        X, y = data_preprocessing(df.dropna(subset=var_model+[var_target]),
                                var_model=var_model, var_target=var_target,
                                enc_strat_cat=enc_strat_cat)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        ## Training KNN
        gsCV.fit(X_tr, y_tr)
        res = gsCV.cv_results_
        ## Predicting test set with the model and clipping
        y_pr = gsCV.predict(X_te)
        try:
            if clip: y_pr = y_pr.clip(*clip) # regressor option only
        except:
            print("ERROR: clip available for regressor option only") 
        # Comparison with naive baselines
        if mod == 'class':
            naive_model_compare_acc_f1(X_tr,y_tr,X_te,y_te,y_pr,average='micro')
        elif mod == 'reg':
            naive_model_compare_r2(X_tr,y_tr,X_te,y_te,y_pr)
        else:
            print("ERROR: check type of target feature...")
        ## Predicting using knn
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



''' Builds a customizable column_transformer which parameters can be optimized in a GridSearchCV
CATEGORICAL : three differents startegies for 3 different types of
categorical variables:
- low cardinality: customizable strategy (strat_low_card)
- high cardinality: customizable strategy (strat_high_card)
- boolean or equivalent (2 categories): ordinal
QUANTITATIVE (remainder): 
- StandardScaler

-> EXAMPLE (to use apart from gscv):
cust_enc = CustTransformer(thresh_card=12,
                       strat_binary = 'ord',
                       strat_low_card = 'ohe',
                       strat_high_card = 'loo',
                       strat_quant = 'stand')
cust_enc.fit(X_tr, y1_tr)
cust_enc.transform(X_tr).shape, X_tr.shape

-> EXAMPLE (to fetch names of the modified dataframe):
small_df = df[['Outlier', 'Neighborhood', 'CertifiedPreviousYear',
               'NumberofFloors','ExtsurfVolRatio']]
# small_df.head(2)
cust_trans = CustTransformer()
cust_trans.fit(small_df)
df_enc = cust_trans.transform(small_df)
cust_trans.get_feature_names(small_df)

'''
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.preprocessing import *
import numpy as np
import pandas as pd


class CustTransformer(BaseEstimator):

    def __init__(self, thresh_card=12,
                 strat_binary='ord', strat_low_card='ohe',
                 strat_high_card='bin', strat_quant='stand'):
        self.thresh_card = thresh_card
        self.strat_binary = strat_binary
        self.strat_low_card = strat_low_card
        self.strat_high_card = strat_high_card
        self.strat_quant = strat_quant
        self.dict_enc_strat = {'binary': strat_binary,
                               'low_card': strat_low_card,
                               'high_card': strat_high_card,
                               'numeric': strat_quant}

    def d_type_col(self, X):
        bin_cols = (X.nunique()[X.nunique() <= 2].index)
        X_C_cols = X.select_dtypes(include=['object', 'category'])
        C_l_card_cols = \
            X_C_cols.nunique()[X_C_cols.nunique() \
                .between(3, self.thresh_card)].index
        C_h_card_cols = \
            X_C_cols.nunique()[X_C_cols.nunique() > self.thresh_card].index
        Q_cols = [c for c in X.select_dtypes(include=[np.number]).columns \
                  if c not in bin_cols]
        d_t = {'binary': bin_cols,
               'low_card': C_l_card_cols,
               'high_card': C_h_card_cols,
               'numeric': Q_cols}
        d_t = {k: v for k, v in d_t.items() if len(v)}
        # print(d_t)
        return d_t

    def get_feature_names(self, X, y=None):
        if self.has_num and self.has_cat:
            self.ct_cat.fit(X, y)
            cols = self.ct_cat.get_feature_names() + self.num_cols
        elif self.has_num and not self.has_cat:
            cols = self.num_cols
        elif not self.has_num and self.has_cat:
            self.ct_cat.fit(X, y)
            cols = self.ct_cat.get_feature_names()
        else:
            cols = None
        return cols

    def fit(self, X, y=None):
        # Dictionary to translate strategies
        d_enc = {'ohe': ce.OneHotEncoder(),
                 'hash': ce.HashingEncoder(),
                 'ord': ce.OrdinalEncoder(),
                 'loo': ce.LeaveOneOutEncoder(),
                 'bin': ce.BinaryEncoder(),
                 'stand': StandardScaler(),
                 'minmax': MinMaxScaler(),
                 'maxabs': MaxAbsScaler(),
                 'robust': RobustScaler(quantile_range=(25, 75)),
                 'norm': Normalizer(),
                 'quant_uni': QuantileTransformer(output_distribution='uniform'),
                 'quant_norm': QuantileTransformer(output_distribution='normal'),
                 'boxcox': PowerTransformer(method='box-cox'),
                 'yeo': PowerTransformer(method='yeo-johnson'),
                 'log': FunctionTransformer(func=lambda x: np.log1p(x),
                                            inverse_func=lambda x: np.expm1(x)),
                 'none': FunctionTransformer(func=lambda x: x,
                                             inverse_func=lambda x: x),
                 }

        # # dictionnaire liste des transfo categorielles EXISTANTES
        d_t = self.d_type_col(X)
        # numerics
        self.has_num = ('numeric' in d_t.keys())
        # categoricals
        self.has_cat = len([s for s in d_t.keys() if s in ['binary', 'low_card', 'high_card']]) > 0
        if self.has_cat:
            list_trans = []  # dictionnaire des transfo categorielles EXISTANTES
            for k, v in d_t.items():
                if k != 'numeric':
                    list_trans.append((k, d_enc[self.dict_enc_strat[k]], v))

            self.cat_cols = []  # liste des colonnes catégorielles à transformer
            for k, v in self.d_type_col(X).items():
                if k != 'numeric': self.cat_cols += (list(v))

            self.ct_cat = ColumnTransformer(list_trans)
            self.cat_trans = Pipeline([("categ", self.ct_cat)])

        if self.has_num:
            self.num_trans = Pipeline([("numeric", d_enc[self.strat_quant])])
            self.num_cols = d_t['numeric']

        if self.has_num and self.has_cat:
            self.column_trans = \
                ColumnTransformer([('cat', self.cat_trans, self.cat_cols),
                                   ('num', self.num_trans, self.num_cols)])
        elif self.has_num and not self.has_cat:
            self.column_trans = \
                ColumnTransformer([('num', self.num_trans, self.num_cols)])
        elif not self.has_num and self.has_cat:
            self.column_trans = ColumnTransformer([('cat', self.cat_trans, self.cat_cols)])
        else:
            print("The dataframe is empty : no transformation can be done")
            
        return self.column_trans.fit(X, y)
    
    def transform(self, X, y=None):
        return pd.DataFrame(self.column_trans.transform(X),
                            index=X.index,
                            columns=self.get_feature_names(X, y))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return pd.DataFrame(self.column_trans.transform(X),
                            index=X.index,
                            columns=self.get_feature_names(X, y))
        # if y is None:
        #     self.fit(X)
        #     return pd.DataFrame(self.column_trans.transform(X),
        #                         index=X.index,
        #                         columns=self.get_feature_names(X, y))
        # else:
        #     self.fit(X, y)
        #     return pd.DataFrame(self.column_trans.transform(X),
        #                         index=X.index,
        #                         columns=self.get_feature_names(X, y))



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

    def fit(self,X):
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


def prepare_2D_axes(X, y, ser_clust=None, proj=['PCA', 'UMAP', 't-SNE'],
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
            #### all clusterers don't have .predict/labels_ method -> changed
            if hasattr(model, 'labels_'):
                clust = model.labels_
            else:
                clust = model.predict(X)
        else:
            clust = ser_clust
        # calculation of centers
        centers_df = X.assign(clust=clust).groupby('clust').mean()
        X = X.append(centers_df)

    ## Projection of all the points through the transformations

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

def plot_projection(X, y, model=None, ser_clust = None, proj='PCA',
                    tw_n_neigh=5, title=None, bboxtoanchor=None,
                    figsize=(5, 3), size=1, palette='tab10',
                    legend_on=False, fig=None, ax=None, random_state=14):

    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax

    # a1 - if model : computes clusters, clusters centers and plot with colors
    if model is not None:

        # Computes the axes for projection with centers
        # (uses fitted model if already fitted)
        dict_proj, dict_proj_centers, model = prepare_2D_axes(X, y,
                                                              proj=[proj],
                                                              model=model,
                                                              centers_on=True,
                                                              random_state=random_state)

        # ...or using model already fitted in prepare_2D_axes to get it
        #### all clusterers don't have .predict/labels_ method -> changed
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
    trustw = trustworthiness(X, dict_proj[proj],
                            n_neighbors=tw_n_neigh, metric='euclidean')
    # Computing the trustworthiness category by category
    ser_tw_clust = groups_trustworthiness(X, dict_proj[proj], ser_clust,
                                          n_neighbors=tw_n_neigh)

    # b1 - if ser_clust exists (either calculated from model or given)
    if ser_clust is not None:

        # Showing the points, cluster by cluster
        # for i in range(n_clust):
        for i, name_clust in enumerate(ser_clust.unique()):
            ind = ser_clust[ser_clust == name_clust].index
            ax.scatter(dict_proj[proj].loc[ind].iloc[:, 0],
                       dict_proj[proj].loc[ind].iloc[:, 1],
                       s=size, alpha=0.7, c=colors[i], zorder=1)

            # Showing the clusters centers
            ax.scatter(dict_proj_centers[proj].iloc[:, 0].loc[name_clust],
                        dict_proj_centers[proj].iloc[:, 1].loc[name_clust],
                        marker='o', c=colors[i], alpha=0.7, s=150,
                       edgecolor='k',
                       label="{}: {} | tw={:0.2f}".format(i, name_clust,
                                                          ser_tw_clust[name_clust]),
                       zorder=10) # for the labels only
            # Showing the clusters centers labels (number)
            ax.scatter(dict_proj_centers[proj].iloc[:, 0].loc[name_clust],
                        dict_proj_centers[proj].iloc[:, 1].loc[name_clust],
                        marker=r"$ {} $".format(i),#
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
        # Plotting the point in grey
        ax.scatter(dict_proj[proj].iloc[:, 0],
                   dict_proj[proj].iloc[:, 1],
                   s=size, alpha=0.7, c='grey')

    title = "Projection: " + proj + "(trustworthiness: {:.2f})".format(trustw)\
             if title is None else title
    ax.set_title(title + "\n(trustworthiness: {:.2f})".format(trustw),
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('ax 1'), ax.set_ylabel('ax 2')