###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

from IPython.display import display
import util as ut
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns


def heatmap(features):
    # Correlation Matrix Heatmap
    f, ax = plt.subplots(figsize=(10, 6))
    corr = features.corr()
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',linewidths=.05)
    f.subplots_adjust(top=0.93)
    t= f.suptitle('Feature Attributes Correlation Heatmap', fontsize=14)
    
def histogram(features):
    features.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)    
    plt.tight_layout(rect=(0, 0, 1.2, 1.2))
    

def histogram_feature(df,feat,title,xlabel,ylabel):
    fig = plt.figure(figsize = (6,4))
    title = fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.85, wspace=0.3)

    ax = fig.add_subplot(1,1, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel) 
    ax.text(400, 400, r'$\mu$='+str(round(df[feat].mean(),2)),fontsize=12)
    freq, bins, patches = ax.hist(df[feat], color='steelblue', bins=10,edgecolor='black', linewidth=1)  
    


