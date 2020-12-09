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
                 fmt='.2f', robust=False, fig=None, ax=None):

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
                     center=center, annot=True, annot_kws={"size": 10}, fmt=fmt,
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