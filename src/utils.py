import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def convert(df):
    replace = {'No': 0, 'Yes': 1}
    replace2 = {'Male': 0, 'Female': 1}
    bool_col = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    df[bool_col] = df[bool_col].replace(replace)
    df['gender'] = df['gender'].replace(replace2)
    
    return df

def revert(df):
    replace = {0: 'No', 1: 'Yes'}
    replace2 = {0: 'Male', 1: 'Female'}
    bool_col = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    df[bool_col] = df[bool_col].replace(replace)
    df['gender'] = df['gender'].replace(replace2)
    
    return df

def plot_cat(df, nrows=8, ncols=2):
    ncount = len(df)
    cat_cols = df.select_dtypes(exclude=np.number).columns.drop('Churn')
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=(12,24))
    fig.suptitle('Churn rate/occurrence')
    for i,col in enumerate(cat_cols):
        ax = sns.countplot(y=col, hue='Churn', data=df, ax=axes[i//ncols][i%ncols])
        ax.set_xlabel('')
        for p in ax.patches:
            x=p.get_bbox().get_points()[1,0]
            y=p.get_bbox().get_points()[:,1]
            perc = 100.*x/ncount
            if perc > 10:
                ax.annotate('{:.2f}%'.format(perc), (x/2, y.mean()), 
                        ha='center', va='center') # set the alignment of the text
            else:
                ax.annotate('{:.2f}%'.format(perc), (x, y.mean()), 
                        ha='left', va='center') # set the alignment of the text
    return fig

def plot_num(df, nrows=3, ncols=1):
    fig, axes = plt.subplots(nrows, ncols)
    num_cols = df.select_dtypes(include=np.number).columns
    for i,col in enumerate(num_cols):
        return
