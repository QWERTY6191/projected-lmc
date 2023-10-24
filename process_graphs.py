import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

##--------------------------------------------------------------------------------------------------------------------
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes
sns.color_palette('deep')

##---------------------------------------------------------------------------------------------------------------------

## Plots for lmc paper

def plot_var(dfs, v, metric, xlabel, ylabel, scale, mods_to_plot, kwargs_mod, equal_axes=False):
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    full_labels = []
    for df in dfs:
        df_temp = df[df['model'].isin(mods_to_plot)]
        lineplot = df_temp[[metric, 'model',v]].copy()
        try:
            lineplot = lineplot.pivot(index='model', columns=v, values=metric).T
        except:
            return lineplot
        lineplot.reset_index(inplace=True)
        lineplot = lineplot.set_index(v)
        labels = lineplot.columns.values
        xvals = lineplot.index.values

        if scale=='logy':
            plotfunc = ax.semilogy
        elif scale=='logx':
            plotfunc = ax.semilogx
        elif scale=='loglog':
            plotfunc = ax.loglog
        elif scale=='lin':
            plotfunc = ax.plot

        for mod in labels:
            plotfunc(xvals, lineplot[mod].values, **kwargs_mod[mod])
            full_labels.append(mod)

    ax.grid(True, which="both")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title='Models', labels=full_labels, title_fontsize=13)
    if equal_axes:
        ax.set_aspect('equal', adjustable='box')
    plt.show()
    plt.close('all')
    return

fancy_labels = {'mean_err_abs':r'Average L1 error', 'mu_common':r'$\mu_{str}$ (fraction of structured noise)',
                'n_tasks':r'Number of tasks', 'n_lat':r'Number of latent processes',
                'n_points':r'Number of training points', 'n_lat_hid':r'$N_{lat}^{noise}$ (rank of task noise covariance)',
                'mu_noise':r'$\mu_{noise}$ (fraction of noise in the observations)',
                'max_scales':r'Maximum lengthscale of the latent data',
                'n_sucess_runs':r'Number of fully converged training runs',
                'H_corr':r'Correlation between true and estimated mixing matrices',
                'PVA':r'Predictive Variance adequacy',
                'lscales_mismatch':r'Mismatch between true and estimated lenghtscales',
                'mean_err_quant95':r'95 \% error quantile'}

## Define styles for all models

# kwargs_mod = {'diagproj':{'ls':'--', 'lw':3, 'c':'k', 'marker':'o', 'markersize':8},
#             'oilmm':{'ls':'-', 'lw':3, 'c':'c', 'marker':'D', 'markersize':8},
#             'exact':{'ls':':', 'lw':3, 'c':'m', 'marker':'+', 'markersize':10},
#             'var':{'ls':'-.', 'lw':3, 'c':'y', 'marker':'x', 'markersize':10},
#               }
kwargs_mod = {'diagproj':{'ls':'--', 'lw':2, 'c':'b', 'marker':'^', 'markersize':8},
            'proj':{'ls':'-.', 'lw':2, 'c':'g', 'marker':'x', 'markersize':8},
            'bdn':{'ls':':', 'lw':2, 'c':'m', 'marker':'*', 'markersize':8},
            'bdn_diag':{'ls':':', 'lw':2, 'c':'c', 'marker':'v', 'markersize':8},
            'oilmm':{'ls':'--', 'lw':2, 'c':'r', 'marker':'+', 'markersize':8},
            'var':{'ls':'-', 'lw':3, 'c':'k', 'marker':'o', 'markersize':10},
            'exact':{'ls':'-', 'lw':3, 'c':'y', 'marker':'o', 'markersize':10}
              }

variables = ['n_tasks', 'n_lat', 'n_lat_hid', 'n_points', 'mu_noise', 'mu_common', 'max_scales', 'lik_rank']
all_models = np.array(['exact', 'var', 'proj', 'diagproj', 'oilmm', 'bdn', 'bdn_diag'])


## Define graph choices
mods_to_plot = all_models#[[0,1,3,4]]#[:-2]
var_plot = variables[4]
metric = 'mean_err_abs'
scale = 'loglog'
equal_axes=False

## Select the study of interest
prefix = '_void'
postfix = '_reject'
postfix += ''

if __name__=='__main__':
    path = 'results_lmc/parameter_study_' + var_plot + prefix + postfix + '.csv'
    df = pd.read_csv(path, index_col=0)
    doubled = True
    if doubled:
        df = df.iloc[:len(df)//2, :]
    dfs = [df]

    ## For the eventual merging of runs made with different models
    
    # prefix = '_void'  
    # postfix = '_reject'
    # postfix += ''
    # o_paths = ['results_lmc/parameter_study_' + var_plot + prefix + postfix + '.csv']
    # for opath in o_paths:
    #     temp = pd.read_csv(opath, index_col=0)
    #     dfs.append(temp)
    #     # new_mods = temp['model'].unique()
    #     # df = df[~df['model'].isin(new_mods)]
    #     # df = pd.concat([df, temp], axis=0)

    xlabel = var_plot
    ylabel = metric
    xlabel_f, ylabel_f = fancy_labels[xlabel], fancy_labels[ylabel]

    lineplot = plot_var(dfs, var_plot, metric, xlabel_f, ylabel_f, scale=scale, mods_to_plot=mods_to_plot, kwargs_mod=kwargs_mod, equal_axes=equal_axes)
