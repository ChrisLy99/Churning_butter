import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from re import finditer

def convert(df):
    """Turns original binary values to 0/1"""
    replace = {'No': 0, 'Yes': 1}
    replace2 = {'Male': 0, 'Female': 1}
    bool_col = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    df[bool_col] = df[bool_col].replace(replace)
    df['gender'] = df['gender'].replace(replace2)
    
    return df

def revert(df):
    """Turns 0/1 to original binary values"""
    replace = {0: 'No', 1: 'Yes'}
    replace2 = {0: 'Male', 1: 'Female'}
    bool_col = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    df[bool_col] = df[bool_col].replace(replace)
    df['gender'] = df['gender'].replace(replace2)
    
    return df

def plot_cat(df, nrows=8, ncols=2):
    """Return a plotgrid of all categorical columns"""
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

def camel_case_split(identifier):
    """ref -> https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python"""
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def clean_str(s):
    return ' '.join(camel_case_split(s)).capitalize()

def plot_num(df, nrows=2, ncols=2):
    """Return a plotgrid of all numerical columns"""
    num_cols = df.select_dtypes(include=np.number).columns
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=(12,8))
    fig.suptitle('Churn occurrence')
    for i,col in enumerate(num_cols):
        ax = sns.histplot(x=col, hue='Churn', data=df, kde=True, ax=axes[i//ncols][i%ncols])
        ax.set_xlabel(f'{clean_str(col)}')
    return fig