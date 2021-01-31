from statsmodels.stats.weightstats import _zstat_generic
from statsmodels.stats.weightstats import _tstat_generic
from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

def plottest(pvalue, significance, testtype, testdist, df=None):

    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 4.1*sigma, mu + 4.1*sigma, 1000)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})
    plt.rcParams["font.size"] = 12
    plt.rcParams['hatch.linewidth'] = 2

    if testdist=='z':
        y = stats.norm.pdf(x, mu, sigma)
        fig.suptitle('Z-distribution')
        ax2.set_xlabel('z-score')

        if testtype=='smaller' or testtype=='larger':
            score_crit = abs(stats.norm.ppf(significance))
            score_obs = abs(stats.norm.ppf(pvalue))
        elif testtype=='two-sided':
            score_crit = abs(stats.norm.ppf(significance/2))
            score_obs = abs(stats.norm.ppf(pvalue/2))
        else:
            raise Exception("The testtype must be 'smaller', 'larger' or 'two-sided'")

    elif testdist=='t':
        if df==None:
            raise Exception("You must have to pass the degrees of freedom")
        else:
            y = stats.t.pdf(x, df, mu,  sigma)
            fig.suptitle('T-distribution (df=%d)' %(df))
            ax2.set_xlabel('t-score')

        if testtype=='smaller' or testtype=='larger':
            score_crit = abs(stats.t.ppf(significance, df))
            score_obs = abs(stats.t.ppf(pvalue, df))
        elif testtype=='two-sided':
            score_crit = abs(stats.t.ppf(significance/2, df))
            score_obs = abs(stats.t.ppf(pvalue/2, df))
        else:
            raise Exception("The testtype must be 'smaller', 'larger' or 'two-sided'")


    ax1.plot(x, y, color='black', lw=2)

    if testtype=='smaller':
        ax1.fill_between(x, 0, y, where=x<-score_obs, color='blue', alpha=.4, hatch='\\'*3, label='p-value', lw=3)
        ax1.fill_between(x, 0, y, where=x<-score_crit, color='red', alpha=.3, hatch='/'*3, label='significance', lw=3)

        ax2.plot([-4.1,-score_crit], [1,1], color='red', alpha=.7, label='critical region')
        ax2.axvline(-score_crit, color='black', alpha=.4, ls=':')
        ax2.axvline(-score_obs, color='black', alpha=.4, ls=':')
        ax2.plot(-score_crit, 1, 'ro', color='red', alpha=.7, label='crit')
        ax2.plot(-score_obs, 1, 'ro', color='blue', alpha=.7, label='obs')

    elif testtype=='larger':
        ax1.fill_between(x, 0, y, where=x>score_obs, color='blue', alpha=.4, hatch='\\'*3, label='p-value', lw=3)
        ax1.fill_between(x, 0, y, where=x>score_crit, color='red', alpha=.3, hatch='/'*3, label='significance', lw=3)

        ax2.plot([score_crit,4.1],[1,1], color='red', alpha=.7, label='critical region')
        ax2.axvline(score_crit, color='black', alpha=.4, ls=':')
        ax2.axvline(score_obs, color='black', alpha=.4, ls=':')
        ax2.plot(score_crit, 1, 'ro',color='red', alpha=.7, label='crit')
        ax2.plot(score_obs, 1, 'ro', color='blue', alpha=.7, label='obs')

    elif testtype=='two-sided':
        ax1.fill_between(x, 0, y, where=x>score_obs, color='blue', alpha=.4, hatch='\\'*3, label='p-value', lw=3)
        ax1.fill_between(x, 0, y, where=x<-score_obs, color='blue', alpha=.4, hatch='\\'*3, lw=3)
        ax1.fill_between(x, 0, y, where=x>score_crit, color='red', alpha=.3, hatch='/'*3, label='significance', lw=3)
        ax1.fill_between(x, 0, y, where=x<-score_crit, color='red', alpha=.3, hatch='/'*3, lw=3)

        ax2.plot([-4.1,-score_crit], [1,1], color='red', alpha=.7, label='critical region')
        ax2.plot([score_crit,4.1], [1,1], color='red', alpha=.7)
        ax2.axvline(-score_crit, color='black', alpha=.4, ls=':')
        ax2.axvline(-score_obs, color='black', alpha=.4, ls=':')
        ax2.axvline(score_crit, color='black', alpha=.4, ls=':')
        ax2.axvline(score_obs, color='black', alpha=.4, ls=':')
        ax2.plot(-score_crit, 1, 'ro', color='red', alpha=.7, label='crit')
        ax2.plot(score_crit, 1, 'ro', color='red', alpha=.7)
        ax2.plot(-score_obs, 1, 'ro', color='blue', alpha=.7, label='obs')
        ax2.plot(score_obs, 1, 'ro', color='blue', alpha=.7)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(left=False, labelleft=False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(True)
    ax2.tick_params(left=False, labelleft=False)
    ax2.set_xticks(list(range(-3,4,1)))
    ax2.set_xticklabels([str(i) for i in range(-3,4,1)])

    lines=[]
    labels=[]

    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    ax1.legend(lines, labels, bbox_to_anchor=(1,0.7), loc='upper left')

    plt.show()

class Clustering_Visualization:
    
    def __init__(self, dataframe): 
        # dataframe = pandas dataframe de onde serão extraídos os dados
        
        self.data = dataframe
        
    def plot2D(self, xlabel, ylabel, clusters):
        # xlabel = string, nome da coluna do dataframe que formará o eixo x
        # ylabel = string, nome da coluna do dataframe que formará o eixo y
        # clusters = array, quantidade de clusters em cada subgráfico

        n = len(clusters)
        
        plt.subplots(1, n, figsize=(n*5, 4))

        for i, c in enumerate(clusters):
            model = KMeans(n_clusters=c)
            model.fit(self.data[[xlabel,ylabel]])
            labels = model.labels_
            centroids = model.cluster_centers_

            plt.subplot(1, n, i+1)

            sns.scatterplot(x=xlabel, y=ylabel, data=self.data, hue=labels, palette=sns.color_palette("tab10", c), legend=False).set(title='K-Means %d clusters' %c)
            sns.scatterplot(x=centroids[:,0], y=centroids[:,1], **{'marker':'o', 's':150, 'color':'white', 'edgecolor':'black'})
            sns.scatterplot(x=centroids[:,0], y=centroids[:,1], **{'marker':'x', 's':40, 'color':'red', 'linewidth':2})

        plt.show()
        
    def elbow(self, labels, max_clusters):
        # label = array of string, nomes das colunas do dataframe a considerar
        # max_clusters = positive int, número máximo de clusters a iterar
        
        inertia = list()
        
        for c in range(1, max_clusters+1):
            model = KMeans(n_clusters=c)
            model.fit(self.data[labels])
            inertia.append(model.inertia_)
        
        sns.lineplot(x=range(1, max_clusters+1), y=inertia, **{'marker':'o'}).set(xticks=range(1, max_clusters+1), 
                                                                                  xlabel='Número de clusters k', 
                                                                                  ylabel='Soma total das distâncias',
                                                                                  title='Método do Cotovelo')
        plt.show()
        