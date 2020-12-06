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