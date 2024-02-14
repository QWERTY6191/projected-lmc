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
## Plot inputs

variables = ['p', 'q', 'q_noise', 'n', 'mu_noise', 'mu_str', 'max_scale', 'lik_rank']
all_models = np.array(['ICM', 'var', 'proj', 'diagproj', 'oilmm', 'bdn', 'bdn_diag'])
metrics = ['mean_err_abs', 'PVA', 'RMSE', 't_per_iter', 'train_time']

mods_to_plot = all_models[[1,3,4,6]]
v = variables[0] # chose the x-axis variable here
metric = metrics[0] # chose the y-axis variable here
n_runs = 20 # to identify the file name
##---------------------------------------------------------------------------------------------------------------------

## >>> Reproducing the plots from the paper : just uncomment the desired line and run the script <<<

mods_to_plot, v, metric, n_runs = all_models[[0,1,3,4,6]], variables[4], metrics[2], 40   # Figure 1
# mods_to_plot, v, metric, n_runs = all_models[[0,1,3,4,6]], variables[5], metrics[2], 20   # Figure 2
# mods_to_plot, v, metric, n_runs = all_models[[1,3,4,6]], variables[2], metrics[1], 10   # Figure 3
# mods_to_plot, v, metric, n_runs = all_models[[0,1,3,4,6]], variables[0], metrics[4], 20   # Figure 4
# mods_to_plot, v, metric, n_runs = all_models[[0,1,3,4,6]], variables[0], metrics[3], 20   # Figure 5a
# mods_to_plot, v, metric, n_runs = all_models[[0,1,3,4,6]], variables[1], metrics[3], 20   # Figure 5b
# mods_to_plot, v, metric, n_runs = all_models[[0,1,3,4,6]], variables[0], metrics[2], 20   # Figure 6a
# mods_to_plot, v, metric, n_runs = all_models[[0,1,3,4,6]], variables[1], metrics[2], 20   # Figure 6b

prefix = '_void'
postfix = '_{0}runs'.format(n_runs)
prefix += '' # if experiments names are further customized
postfix += '' # if experiments names are further customized
error_bars=False # only available if metrics is 'mean_err_abs' (L1 error)
##---------------------------------------------------------------------------------------------------------------------
## Setup

xlabel = v
ylabel = metric
path = 'results/parameter_study_' + v + prefix + postfix + '.csv'
df = pd.read_csv(path, index_col=0)
doubled = 'reject' in postfix # see file experiments.py for explanation of the 'reject' flag (not used in the paper )
if doubled:
    df = df.iloc[:len(df)//2, :]
df['t_per_iter'] = df['train_time'] / df['n_iter']
dfs = [df]

## To eventually plot models from different files

# o_prefix = '_void'
# o_postfix += ''
# o_paths = ['results/parameter_study_' + var_plot + prefix + postfix + '.csv']
# for opath in o_paths:
#     temp = pd.read_csv(opath, index_col=0)
#     dfs.append(temp)
#     # new_mods = temp['model'].unique()
#     # df = df[~df['model'].isin(new_mods)]
#     # df = pd.concat([df, temp], axis=0)

plot_styles = {'diagproj':{'ls':'--', 'lw':2, 'c':'b', 'marker':'^', 'markersize':8},
            'proj':{'ls':'-.', 'lw':2, 'c':'g', 'marker':'x', 'markersize':8},
            'bdn':{'ls':':', 'lw':2, 'c':'m', 'marker':'*', 'markersize':8},
            'bdn_diag':{'ls':':', 'lw':2, 'c':'c', 'marker':'v', 'markersize':8},
            'oilmm':{'ls':'--', 'lw':2, 'c':'r', 'marker':'+', 'markersize':8},
            'var':{'ls':'-', 'lw':3, 'c':'k', 'marker':'o', 'markersize':10},
            'ICM':{'ls':'-', 'lw':3, 'c':'y', 'marker':'o', 'markersize':10}
            }

fancy_labels = {
                'mu_str':r'$\mu_{str}$ (fraction of structured noise)',
                'n':r'Number of training points',
                'p':r'Number of tasks',
                'q':r'Number of latent processes',
                'q_noise':r'$q_{noise}$ (number of noise latent processes)',
                'mu_noise':r'$\mu_{noise}$ (fraction of noise in the observations)',
                'max_scale':r'Maximum lengthscale of the latent data',
                'RMSE':r'RMSE',
                'mean_err_abs':r'Average L1 error',
                'PVA':r'Predictive Variance adequacy',
                'train_time':r'Training time (s)',
                't_per_iter':r'Time per training iteration (s)',
                }

scales_dict = {
    't_per_iter':dict(zip(variables, ['lin']*len(variables))),
    'train_time':dict(zip(variables, ['lin']*len(variables))),
    'PVA':{'p':'lin', 'q':'lin', 'q_noise':'lin', 'n':'lin', 'mu_noise':'logx', 'mu_str':'lin', 'max_scale':'logx', 'lik_rank':'lin'},
    'RMSE':{'p':'lin', 'q':'lin', 'q_noise':'lin', 'n':'lin', 'mu_noise':'loglog', 'mu_str':'lin', 'max_scale':'logx', 'lik_rank':'lin'},
}

xlabel_f, ylabel_f = fancy_labels[xlabel], fancy_labels[ylabel]
scale = scales_dict[metric][v]
equal_axes=(scale=='loglog')

##---------------------------------------------------------------------------------------------------------------------
def make_plot(dfs, v, metric, xlabel, ylabel, scale, mods_to_plot, plot_styles, equal_axes=False, error_bars=False):
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    full_labels = []
    for df in dfs:
        df_temp = df[df['model'].isin(mods_to_plot)]
        if error_bars:
            lineplot = df_temp[[metric, 'mean_err_quant05', 'mean_err_quant95', 'model',v]].copy()
        else:
            lineplot = df_temp[[metric, 'model',v]].copy()
            
        if error_bars:
            lineplot_l = lineplot.pivot(index='model', columns=v, values='mean_err_quant05').copy().T
            lineplot_u = lineplot.pivot(index='model', columns=v, values='mean_err_quant95').copy().T

        lineplot = lineplot.pivot(index='model', columns=v, values=metric).T
        lineplot.reset_index(inplace=True)
        lineplot = lineplot.set_index(v)
        labels = lineplot.columns.values
        xvals = lineplot.index.values
        if error_bars:
            lineplot_l.reset_index(inplace=True)
            lineplot_l = lineplot_l.set_index(v)
            lineplot_u.reset_index(inplace=True)
            lineplot_u = lineplot_u.set_index(v)

        if scale=='logy':
            plotfunc = ax.semilogy
        elif scale=='logx':
            plotfunc = ax.semilogx
        elif scale=='loglog':
            plotfunc = ax.loglog
        else:
            plotfunc = ax.plot

        for mod in labels:
            plotfunc(xvals, lineplot[mod].values, **plot_styles[mod])
            full_labels.append(mod)
            if error_bars:
                plt.fill_between(xvals, lineplot_l[mod].values, lineplot_u[mod].values, color=plot_styles[mod]['c'], alpha=0.2)

    if metric=='PVA':
        ax.axhline(y=0., linestyle='--', color='g')
        ax.text(50, 0, 'Optimal PVA value', color='g', ha='right', va='bottom')
    ax.grid(True, which="both")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title='Models', labels=full_labels, title_fontsize=13)
    if equal_axes:
        ax.set_aspect('equal', adjustable='box')
    plt.show()
    plt.close('all')
    return lineplot

lineplot = make_plot(dfs, v, metric, xlabel_f, ylabel_f, scale=scale, mods_to_plot=mods_to_plot, plot_styles=plot_styles, equal_axes=equal_axes, error_bars=error_bars)
##---------------------------------------------------------------------------------------------------------------------
"""
## Weather prediction plot

import matplotlib.dates as mdates
df = pd.read_csv('results/preds_tidal_div4_14days_q_n_mix_void.csv')
df = df[(df['Date'] >= '2020-06-05') & (df['Date'] <= '2020-06-12')]
df['Date'] = pd.to_datetime(df['Date'])
test_indices = np.where(df['pred0']!=0.)[0]
sub_indices = np.arange(0, test_indices[0])
sup_indices = np.arange(test_indices[-1]+1, len(df))
df_sub, df_sup = df.iloc[sub_indices, :], df.iloc[sup_indices, :]
test_indices = df.index.values[test_indices]

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(df_sub['Date'], df_sub['bramblemet'], color='blue')
ax.scatter(df_sub['Date'], df_sub['bramblemet'], label='training data', color='blue', marker='.')
ax.plot(df_sup['Date'], df_sup['bramblemet'], color='blue')
ax.scatter(df_sup['Date'], df_sup['bramblemet'], color='blue', marker='.')
ax.scatter(df.loc[test_indices, 'Date'], df.loc[test_indices, 'bramblemet'], label='test data', color='k', marker='x')
ax.plot(df.loc[test_indices, 'Date'], df.loc[test_indices, 'pred0'], color='red', label='prediction')
ax.fill_between(df.loc[test_indices, 'Date'], df.loc[test_indices, 'lower0'], df.loc[test_indices, 'upper0'], color='red', alpha=0.2)
ax.set_xlabel('Date')
ax.set_ylabel('Tide height (m)')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(df_sub['Date'], df_sub['cambermet'], color='blue')
ax.scatter(df_sub['Date'], df_sub['cambermet'], label='training data', color='blue', marker='.')
ax.plot(df_sup['Date'], df_sup['cambermet'], color='blue')
ax.scatter(df_sup['Date'], df_sup['cambermet'], color='blue', marker='.')
ax.scatter(df.loc[test_indices, 'Date'], df.loc[test_indices, 'cambermet'], label='test data', color='k', marker='x')
ax.plot(df.loc[test_indices, 'Date'], df.loc[test_indices, 'pred1'], color='red', label='prediction')
ax.fill_between(df.loc[test_indices, 'Date'], df.loc[test_indices, 'lower1'], df.loc[test_indices, 'upper1'], 
                color='red', alpha=0.2)
ax.set_xlabel('Date')
ax.set_ylabel('Tide height (m)')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.legend()
plt.show()
"""