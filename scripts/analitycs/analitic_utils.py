import seaborn as sns
import matplotlib.pyplot as plt
from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

PALETTE = sns.color_palette('Dark2', 10)

def plot_bar_label_distribution(df1:PandasDataFrame,title:str,):
    ax = sns.barplot(data=df1,  palette=PALETTE, x = 'class',hue='class', y='percentage')
    ax.set(title=title)
    ax.set(xlabel='clases', ylabel='porcentaje')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)