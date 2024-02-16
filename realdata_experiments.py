import numpy as np
np.random.seed(seed=12)
import pandas as pd
import torch
torch.manual_seed(12)
torch.backends.cuda.matmul.allow_tf32 = False
import gpytorch as gp
import time
from projected_lmc import *

##------------------------------------------------------------------
## Generic setup 

print_metrics=True  # if True, performance metrics are printed at each run (dosen't affect exported results)
print_loss=True # if True, loss is printed after freq_print iteration (dosen't affect exported results)
freq_print=1000 # to further customize experiment name
train_ind_rat = 1.5 # ratio between number of training points and number of inducing points for the variational model
# For experiments where all models have inducing points, this variable is overriden
gpu = False
export_results = True
appendix = '' # to further customize experiment name
experiments = ['ship', 'neutro', 'sarcos', 'tidal']
models_to_run = ['proj','diagproj','oilmm', 'var','bdn', 'bdn_diag', 'ICM']
v_test_2 = 'void'
##----------------------------------------------

## >>> Reproducing the experiments of the paper : just chose the experiment of interest here, then run the script <<<
experiment = experiments[3]

## There is only one subtelty : for models 'var' and 'ICM', two options have been tested for the likelihood rank, 0 and n_tasks.
## Running these two options by setting v_test_2=lik_rank would result in duplicated results for all other models.
## Therefore, if you want to perform this test on 'var' and 'ICM', you can uncomment the following lines.

# models_to_run = ['var','ICM']
# v_test_2 = 'lik_rank'
##----------------------------------------------

def compute_metrics(y_test, y_pred, lower, upper, noise, loss, H_guess_hid, n_iter, train_time, pred_time, print_metrics=True, test_mask=None):
    if test_mask is not None:  # this can be used to compute metrics on a subset of outputs
        y_test = y_test[test_mask]
        y_pred = y_pred[test_mask]
        lower = lower[test_mask]
        upper = upper[test_mask]
    delta = y_test - y_pred
    errs_abs = torch.abs(delta)
    sigma_pred = (upper - lower) / 4
    alpha_CI = torch.mean((torch.abs(y_pred - y_test) < 2 * sigma_pred).float())
    err2 = errs_abs ** 2
    R2_list = 1 - torch.mean(err2, dim=0) / torch.var(y_test, dim=0)
    PVA_list = torch.log(torch.mean(err2 / sigma_pred ** 2, dim=0))
    noise_full = noise * (H_guess_hid**2).sum()  # mean of the diagonal coefficients

    errs_abs = errs_abs.cpu().numpy()
    metrics = {}
    metrics['n_iter'] = n_iter
    metrics['train_time'] = train_time
    metrics['pred_time'] = pred_time
    metrics['loss'] = loss
    metrics['noise'] = noise_full.cpu().numpy()
    metrics['R2'] = R2_list.mean().cpu().numpy()
    metrics['RMSE'] = torch.sqrt(err2.mean()).cpu().numpy()
    metrics['mean_err_abs'], metrics['max_err_abs'] = errs_abs.mean(), errs_abs.max()
    metrics['mean_err_quant05'], metrics['mean_err_quant95'], metrics['mean_err_quant99'] = np.quantile(errs_abs, np.array([0.05, 0.95, 0.99]))
    metrics['mean_sigma'] = sigma_pred.mean().cpu().numpy()
    metrics['PVA'] = PVA_list.mean().cpu().numpy()
    metrics['alpha_CI'] = alpha_CI.mean().cpu().numpy()
    if print_metrics:
        for key, value in metrics.items():
            print(key, value)
    return metrics

def run_models(models_to_run, models_with_sched, q, lik_rank, n_tasks, n_points, X, Y, X_test, Y_test, lrs, 
                n_iters, lr_min, loss_tresh, patience, print_metrics, print_loss, freq_print, gpu, 
                train_ind_rat, n_ind_points, run_key, results, test_mask=None, renorm_func=None, mean_type=None,
                kernel_type=None, lambda_lr=None, ker_kwargs={}):
    
    ## Defining models
    kernel = gp.kernels.MaternKernel if kernel_type is None else kernel_type
    Mean = gp.means.ZeroMean if mean_type is None else mean_type
    decomp = None
    likelihoods, models, mlls, optimizers, schedulers = {}, {}, {}, {}, {}
      
    if 'ICM' in models_to_run:
        likelihoods['ICM'] = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank)
        models['ICM'] = MultitaskGPModel(X, Y, likelihoods['ICM'], n_tasks=n_tasks, init_lmc_coeffs=True,
                            n_latents=q, mean_type=Mean, kernel_type=kernel, decomp=decomp, n_inducing_points=n_ind_points, 
                            fix_diagonal=False, model_type='ICM', ker_kwargs=ker_kwargs)
        
    if 'LMC' in models_to_run: # naive implementation of LMC, not recommended for use (poor performance, very low scalability)
        likelihoods['LMC'] = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank)
        models['LMC'] = MultitaskGPModel(X, Y, likelihoods['LMC'], n_tasks=n_tasks, init_lmc_coeffs=True,
                            n_latents=q, mean_type=Mean, kernel_type=kernel, decomp=decomp, n_inducing_points=n_ind_points,
                            fix_diagonal=False, model_type='LMC', ker_kwargs=ker_kwargs)

    if 'var' in models_to_run:
        likelihoods['var'] = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank)
        TI_rat = train_ind_rat if n_ind_points is None else n_points / n_ind_points
        models['var'] = VariationalMultitaskGPModel(X, train_y=Y, n_tasks=n_tasks, init_lmc_coeffs=True,
                            mean_type=Mean, kernel_type=kernel, n_latents=q, decomp=decomp,
                            train_ind_ratio=TI_rat, seed=0, distrib=gpytorch.variational.CholeskyVariationalDistribution,
                            ker_kwargs=ker_kwargs)
        
    if 'proj' in models_to_run:
        models['proj'] = ProjectedGPModel(X, Y, n_tasks, q, proj_likelihood=None,
                                        mean_type=Mean,  kernel_type=kernel, decomp=decomp,
                                        init_lmc_coeffs=True, BDN=False, diagonal_B=False, diagonal_R=False, 
                                        scalar_B=False, ker_kwargs=ker_kwargs, n_inducing_points=n_ind_points)
        likelihoods['proj'] = models['proj'].likelihood

    if 'diagproj' in models_to_run:
        models['diagproj'] = ProjectedGPModel(X, Y, n_tasks, q, proj_likelihood=None,
                                        mean_type=Mean, kernel_type=kernel, decomp=decomp,
                                        init_lmc_coeffs=True, BDN=False, diagonal_B=True, diagonal_R=False,
                                        scalar_B=False, ker_kwargs=ker_kwargs, n_inducing_points=n_ind_points)
        likelihoods['diagproj'] = models['diagproj'].likelihood

    if 'oilmm' in models_to_run:
        models['oilmm'] = ProjectedGPModel(X, Y, n_tasks, q, proj_likelihood=None,
                                        mean_type=Mean, kernel_type=kernel, decomp=decomp,
                                        init_lmc_coeffs=True, BDN=True, diagonal_B=True, diagonal_R=True,
                                        scalar_B=True, ker_kwargs=ker_kwargs, n_inducing_points=n_ind_points)
        likelihoods['oilmm'] = models['oilmm'].likelihood

    if 'bdn' in models_to_run:
        models['bdn'] = ProjectedGPModel(X, Y, n_tasks, q, proj_likelihood=None,
                                        mean_type=Mean, kernel_type=kernel, decomp=decomp,
                                        init_lmc_coeffs=True, BDN=True, diagonal_B=False, diagonal_R=False,
                                        scalar_B=False, ker_kwargs=ker_kwargs, n_inducing_points=n_ind_points)
        likelihoods['bdn'] = models['bdn'].likelihood

    if 'bdn_diag' in models_to_run:
        models['bdn_diag'] = ProjectedGPModel(X, Y, n_tasks, q, proj_likelihood=None,
                                        mean_type=Mean, kernel_type=kernel, decomp=decomp,
                                        init_lmc_coeffs=True, BDN=True, diagonal_B=True, diagonal_R=False,
                                        scalar_B=False, ker_kwargs=ker_kwargs, n_inducing_points=n_ind_points)
        likelihoods['bdn_diag'] = models['bdn_diag'].likelihood

    ##------------------------------------------------------------------
    ## Configuring optimization
    if gpu:
        X = X.cuda()
        Y = Y.cuda()
        for name in models_to_run:
            models[name] = models[name].cuda()
            likelihoods[name] = likelihoods[name].cuda()

    for name in models_to_run:
        if kernel_type == gp.kernels.SpectralMixtureKernel:
            attribute_string = 'covar_module'
            if name=='ICM':
                attribute_string += '.data_covar_module'
            if n_ind_points is not None and name!='var':
                attribute_string += '.base_kernel'
            attributes = attribute_string.split('.')
            obj = models[name]
            for attr in attributes:
                obj = getattr(obj, attr)
            obj.initialize_from_data(X, Y)  # Spectral Mixture Kernel has to be carefully initialized
        models[name].train()
        likelihoods[name].train()

    if 'LMC' in models_to_run:
        mlls['LMC'] = gp.mlls.ExactMarginalLogLikelihood(likelihoods['LMC'], models['LMC'])
        optimizers['LMC'] = torch.optim.AdamW(models['LMC'].parameters(), lr=lrs['LMC'])  # Includes GaussianLikelihood parameters
    if 'ICM' in models_to_run:
        mlls['ICM'] = gp.mlls.ExactMarginalLogLikelihood(likelihoods['ICM'], models['ICM'])
        optimizers['ICM'] = torch.optim.AdamW(models['ICM'].parameters(), lr=lrs['ICM'])  # Includes GaussianLikelihood parameters
    if 'var' in models_to_run:
        mlls['var'] = gp.mlls.VariationalELBO(likelihoods['var'], models['var'], num_data=n_points)
        optimizers['var'] = torch.optim.AdamW([{'params': models['var'].parameters()}, {'params': likelihoods['var'].parameters()}], lr=lrs['var'])
    if 'proj' in models_to_run:
        mlls['proj'] = ProjectedLMCmll(likelihoods['proj'], models['proj'])
        optimizers['proj'] = torch.optim.AdamW(models['proj'].parameters(), lr=lrs['proj'])
    if 'diagproj' in models_to_run:
        mlls['diagproj'] = ProjectedLMCmll(likelihoods['diagproj'], models['diagproj'])
        optimizers['diagproj'] = torch.optim.AdamW(models['diagproj'].parameters(), lr=lrs['diagproj'])
    if 'oilmm' in models_to_run:
        mlls['oilmm'] = ProjectedLMCmll(likelihoods['oilmm'], models['oilmm'])
        optimizers['oilmm'] = torch.optim.AdamW(models['oilmm'].parameters(), lr=lrs['oilmm'])
    if 'bdn' in models_to_run:
        mlls['bdn'] = ProjectedLMCmll(likelihoods['bdn'], models['bdn'])
        optimizers['bdn'] = torch.optim.AdamW(models['bdn'].parameters(), lr=lrs['bdn'])
    if 'bdn_diag' in models_to_run:
        mlls['bdn_diag'] = ProjectedLMCmll(likelihoods['bdn_diag'], models['bdn_diag'])
        optimizers['bdn_diag'] = torch.optim.AdamW(models['bdn_diag'].parameters(), lr=lrs['bdn_diag'])

    for name in models_to_run:
        if name in models_with_sched:
            if lambda_lr is None:
                schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(optimizers[name], gamma=np.exp(np.log(lr_min / lrs[name]) / n_iters[name]))
            else:
                schedulers[name] = torch.optim.lr_scheduler.LambdaLR(optimizers[name], lambda_lr)

    ##------------------------------------------------------------------
    ## Training models

    times, last_losses = {}, {}
    effective_n_iters = n_iters.copy()
    for name in models_to_run:
        print(' \n Training {0} model ... \n'.format(name))
        start = time.time()
        plateau_id = 0
        for i in range(n_iters[name]):
            optimizers[name].zero_grad()
            with gp.settings.cholesky_jitter(1e-5):
                output_train = models[name](X)
                loss = -mlls[name](output_train, Y)
                if print_loss and i%freq_print==0:
                    print(loss.item())
                loss.backward()
                optimizers[name].step()
            if name in schedulers:
                schedulers[name].step()

            new_loss = loss.item()
            if i>0 and np.abs( 1 - new_loss / last_losses[name]) < loss_tresh:
                plateau_id += 1
                if plateau_id > patience :
                    effective_n_iters[name] = i
                    break
            last_losses[name] = new_loss
        times[name] = time.time() - start

    ##------------------------------------------------------------------
    ## Making predictions

    for name in models_to_run:
        models[name].eval()
        likelihoods[name].eval()
        if gpu:
            models[name] = models[name].cpu()
            likelihoods[name] = likelihoods[name].cpu()

    for name in models_to_run:
    # all these algebra options have been tested to have little impact on results. Warning on the skip_posterior_variances option !!
        with torch.no_grad(),\
                    gp.settings.skip_posterior_variances(state=False), \
                    gp.settings.skip_logdet_forward(state=False), \
                    gp.settings.cg_tolerance(1e-0),\
                    gp.settings.eval_cg_tolerance(1e-2),\
                    gp.settings.max_cholesky_size(128), \
                    gp.settings.max_lanczos_quadrature_iterations(20), \
                    gp.settings.max_preconditioner_size(15), \
                    gp.settings.max_root_decomposition_size(100), \
                    gp.settings.min_preconditioning_size(2000), \
                    gp.settings.num_trace_samples(10), \
                    gp.settings.preconditioner_tolerance(1e-3), \
                    gp.settings.tridiagonal_jitter(1e-5), \
                    gp.settings.cholesky_jitter(1e-3):

            print(' \n Making predictions for {0} model...'.format(name))
            start = time.time()
            if hasattr(models[name], 'full_likelihood'):  # we have to compute the full likelihood of projected models
                full_likelihood = models[name].full_likelihood()
            else:
                full_likelihood = likelihoods[name]
            if hasattr(full_likelihood, 'task_noise_covar_factor'):
                H_guess_hid = full_likelihood.task_noise_covar_factor.squeeze()
            elif hasattr(full_likelihood, 'task_noises'):
                H_guess_hid = full_likelihood.task_noises.squeeze()
            else:
                H_guess_hid = torch.ones(n_tasks)
            observed_pred = full_likelihood(models[name](X_test))
            pred_time = time.time() - start
            pred_y = observed_pred.mean.squeeze()
            lower, upper = observed_pred.confidence_region()
            if renorm_func is not None:
                pred_y, lower, upper = renorm_func(pred_y), renorm_func(lower), renorm_func(upper)
            global_noise = full_likelihood.noise.squeeze() if hasattr(full_likelihood, 'noise') else 1.
            ##------------------------------------------------------------------
            ## Computing, displaying and storing performance metrics

            if print_metrics:
                if hasattr(likelihoods[name], 'task_noises'):
                    print('Noises for {0} model:'.format(name), global_noise * likelihoods[name].task_noises.cpu().numpy()) # for non-projected models if their likelihood is diagonal
                elif hasattr(models[name], 'full_likelihood'):  # for projected models
                    print('Noises for {0} model:'.format(name), likelihoods[name].noise.squeeze().cpu().numpy())
            metrics = compute_metrics(Y_test, pred_y, lower.squeeze(), upper.squeeze(), global_noise, last_losses[name],H_guess_hid, 
                                     effective_n_iters[name], times[name], pred_time, print_metrics=print_metrics, test_mask=test_mask)
            metrics.update(v)
            metrics['model'] = name
            metrics['eps'] = models[name].current_eps if hasattr(models[name], 'current_eps') else 0.
            results[name + run_key] = metrics
    return results, models

##----------------------------------------------

## Tidal height experiment
if experiment=='tidal':
    import os
    from scipy.interpolate import interp1d
    from datetime import datetime
    root = '_experiments/bramblemet/'
    def detrend_data(x, y, degree=1):
        coef = np.polyfit(x, y, degree)
        return y - np.polyval(coef, x)
    degree = 2 # degree of the polynomial detrending
    ndiv = 4 # subsampling factor
    n_ind_points = None
    if __name__ == "__main__":
        torch.set_default_dtype(torch.float32)
        start_date = '2020-06-01'
        end_date = '2020-06-15'
        dico = {}
        stations = ['bramblemet', 'cambermet', 'chimet', 'sotonmet']
        for station in stations:
            df = pd.read_csv(os.path.join(root,'{0}.csv.gz'.format(station)), compression='gzip', low_memory=False)
            df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M')
            df = df.loc[(df['Date'] >= start_date) & (df['Date'] < end_date)]
            df['time_num'] = df['Date'].apply(lambda x: x.timestamp())
            values = df['DEPTH'].values
            if 'time_num' not in dico:  # create a reference time vector with values between 0 and 1
                ref_time = df['time_num'].values
                ref_time_norm = ref_time / ref_time.max()
                ref_time_norm = ref_time_norm - ref_time_norm[0]
                dico['time_num'] = ref_time_norm
                dico['Date'] = df['Date'].values
            else:
                f = interp1d(df['time_num'].values, values) # align all time series on the same reference time vector
                values = f(ref_time)
            dico[station] = detrend_data(ref_time_norm, values, degree=degree)
        df = pd.DataFrame(dico).set_index('Date').astype(np.float32)
        
        df = df.iloc[::ndiv] # subsampling the time series by a factor ndiv
        X, Y = df['time_num'].values[:,None], df.drop('time_num', axis=1).values
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        num_days = (end_date - start_date).days
        test_indices = np.arange(len(df)//2, len(df)//2 + len(df)//num_days) # test set is one day in the middle of the time series
        X, X_test = np.delete(X, test_indices, axis=0), X[test_indices]
        Y, Y_test = np.delete(Y, test_indices, axis=0), Y[test_indices]
        n_points, n_tasks = Y.shape
        X, Y, X_test, Y_test = torch.as_tensor(X), torch.as_tensor(Y), torch.as_tensor(X_test), torch.as_tensor(Y_test)

        lr_min = 1e-3
        lr_max = 1e-1
        loss_tresh = 1e-4
        last_epoch = 5000
        patience = 500
        n_iters = dict(zip(models_to_run, [10000]*len(models_to_run)))
        lrs = dict(zip(models_to_run, [lr_max]*len(models_to_run)))
        models_with_sched = models_to_run
        mean_type = None
        kernel_type = gp.kernels.SpectralMixtureKernel
        v = {
            'q': 3, 
            'lik_rank': 0,
            'n_mix':5,
            'void' : [0.]}
        v_vals = {
            'q' : range(1, n_tasks+1), 
            'lik_rank' : [0, n_tasks],
            'n_mix': range(2,6),
            'void' : [0.]}
        v_test_0 = 'q'
        v_test_1 = 'n_mix'
        appendix += 'div{0}_{1}days'.format(ndiv, num_days)
        if n_ind_points is not None:
            appendix += '_{0}ind'.format(n_ind_points)
        appendix += '' # to further customize experiment name
        path = 'results/realdata_study_'+ experiment + '_' + appendix + '_' + v_test_0 + '_'+ v_test_1 + '_' + v_test_2 + '.csv'
        print(path + '\n')
        results = {}
        min_err = 1.
        for i_v, vval in enumerate(v_vals[v_test_0]):
            for i_v1, vval1 in enumerate(v_vals[v_test_1]):
                for i_v2, vval2 in enumerate(v_vals[v_test_2]):
                    v[v_test_0] = vval
                    v[v_test_1] = vval1
                    v[v_test_2] = vval2
                    q, lik_rank = v['q'], v['lik_rank']
                    run_key = '_' + v_test_0 + '_' + v_test_1 + '_' + v_test_2 + '_{0}_{1}_{2}'.format(i_v, i_v1, i_v2)
                    results, models = run_models(models_to_run, models_with_sched, q, lik_rank, n_tasks, n_points, X, Y, X_test, Y_test, lrs, 
                            n_iters, lr_min, loss_tresh, patience, print_metrics, print_loss, freq_print, gpu, 
                            train_ind_rat, n_ind_points, run_key, results, mean_type=mean_type, kernel_type=kernel_type,
                            ker_kwargs={'num_mixtures':v['n_mix']})
                    
                    for model in models_to_run: # we keep the best model for illustrating predictions (Figure7 in the paper) 
                        if results[model + run_key]['RMSE']< min_err:
                            best_model = models[model]
                            min_err = results[model + run_key]['RMSE']

        res_df = pd.DataFrame.from_dict(results, orient='index')
        if export_results:
            res_df.to_csv(path)

        ## Illustrating predictions
        df[['pred'+str(i) for i in range(n_tasks)]] = np.zeros((len(df), n_tasks))
        df[['lower'+str(i) for i in range(n_tasks)]] = np.zeros((len(df), n_tasks))
        df[['upper'+str(i) for i in range(n_tasks)]] = np.zeros((len(df), n_tasks))
        best_model.eval()
        best_model.likelihood.eval()
        full_likelihood = best_model.full_likelihood()
        observed_pred = full_likelihood(best_model(X_test))
        pred_y = observed_pred.mean.squeeze()
        lower, upper = observed_pred.confidence_region()
        test_indices = df.index[test_indices]
        for i in range(n_tasks):
            df.loc[test_indices, 'pred'+str(i)] = pred_y[:,i].detach().cpu().numpy()
            df.loc[test_indices, 'lower'+str(i)] = lower[:,i].detach().cpu().numpy()
            df.loc[test_indices, 'upper'+str(i)] = upper[:,i].detach().cpu().numpy()
            df.to_csv(path.replace('realdata_study', 'preds'))

##----------------------------------------------
## Ship
if experiment=='ship':
    root = '_experiments/ship/'
    ndiv = 5 # subsampling factor
    n_ind_points = 500
    if __name__ == "__main__":
        torch.set_default_dtype(torch.float32)
        data = pd.read_csv(root + "data.txt", sep=r"\s+", engine="python", dtype=str, header=None).astype(np.float32)
        data = data.iloc[::ndiv]
        X = data.iloc[:, [0, 16, 17]].values
        Y = data.drop([0, 1, 8, 11, 16, 17], axis=1).values
        X, X_test = X[:-100], X[-100:]
        Y, Y_test = Y[:-100], Y[-100:]
        Mean, Std = Y.mean(axis=0), Y.std(axis=0)
        Y, Y_test = (Y - Mean) / Std, (Y_test - Mean) / Std
        n_points, n_tasks = Y.shape
        X, Y, X_test, Y_test = torch.as_tensor(X), torch.as_tensor(Y), torch.as_tensor(X_test), torch.as_tensor(Y_test)

        lr_min = 1e-3
        lr_max = 1e-2
        loss_tresh = 1e-4
        last_epoch = 5000
        patience = 500
        lambda_f = lambda i : i/last_epoch*lr_min/lr_max + (last_epoch-i)/last_epoch*lr_max if i <= last_epoch else lr_min/lr_max
        n_iters = dict(zip(models_to_run, [50000]*len(models_to_run)))
        lrs = dict(zip(models_to_run, [lr_max]*len(models_to_run)))
        models_with_sched = models_to_run
        mean_type = None
        v = {
            'q': n_tasks, 
            'lik_rank': 0,
            'void' : [0.]}
        v_vals = {
            'q' : range(1,11), 
            'lik_rank' : [0,n_tasks],
            'void' : [0.]}
        v_test = 'q'
        appendix += 'div{0}'.format(ndiv)
        if n_ind_points is not None:
            appendix += '_{0}ind'.format(n_ind_points)
        appendix += ''
        path = 'results/realdata_study_'+ experiment + '_' + appendix + '_' + v_test + '_'+ v_test_2 + '.csv'
        print(path + '\n')
        results = {}
        for i_v, vval in enumerate(v_vals[v_test]):
            for i_v2, vval2 in enumerate(v_vals[v_test_2]):
                v[v_test] = vval
                v[v_test_2] = vval2
                q, lik_rank = v['q'], v['lik_rank']
                run_key = '_' + v_test + '_' + v_test_2 + '_{0}_{1}'.format(i_v, i_v2)
                results, models = run_models(models_to_run, models_with_sched, q, lik_rank, n_tasks, n_points, X, Y, X_test, Y_test, lrs, 
                        n_iters, lr_min, loss_tresh, patience, print_metrics, print_loss, freq_print, gpu, 
                        train_ind_rat, n_ind_points, run_key, results, mean_type=mean_type, lambda_lr=lambda_f)
        df = pd.DataFrame.from_dict(results, orient='index')
        if export_results:
            df.to_csv(path)

##----------------------------------------------
## Neutronics
if experiment=='neutro':
    n_ind_points = None
    if __name__ == "__main__":
        root = '_experiments/neutro_data/'
        torch.set_default_dtype(torch.float32)
        X = torch.load(root + 'train_x_sobol256.pt')
        X_test = torch.load(root + 'test_x_LHS512.pt')
        Y = torch.load(root + 'train_data_02g_FA_Lchain.pt')
        Y_test = torch.load(root + 'test_data_02g_FA_Lchain.pt')
        n_points, n_tasks = Y.shape

        lr_min = 1e-3
        lr_max = 1e-2
        loss_tresh = 1e-5
        patience = 500
        last_epoch = 5000
        lambda_f = lambda i : i/last_epoch*lr_min/lr_max + (last_epoch-i)/last_epoch*lr_max if i <= last_epoch else lr_min/lr_max
        n_iters = dict(zip(models_to_run, [100000]*len(models_to_run)))
        lrs = dict(zip(models_to_run, [lr_max]*len(models_to_run)))
        models_with_sched = models_to_run
        v = {
            'n_lat': 20, 
            'lik_rank': 0,
            'void' : [0.]}
        v_vals = {
            'n_lat' : range(1,n_tasks+1), 
            'lik_rank' : [0, n_tasks],
            'void' : [0.]}
        v_test = 'void'
        if n_ind_points is not None:
            appendix += '{0}ind'.format(n_ind_points)
        appendix += ''
        path = 'results/realdata_study_'+ experiment + '_' + appendix + '_' + v_test + '_'+ v_test_2 + '.csv'
        print(path + '\n')
        results = {}
        for i_v, vval in enumerate(v_vals[v_test]):
            for i_v2, vval2 in enumerate(v_vals[v_test_2]):
                v[v_test] = vval
                v[v_test_2] = vval2
                n_lat, lik_rank = v['n_lat'], v['lik_rank']
                run_key = '_' + v_test + '_' + v_test_2 + '_{0}_{1}'.format(i_v, i_v2)
                results, models = run_models(models_to_run, models_with_sched, n_lat, lik_rank, n_tasks, n_points, X, Y, X_test, Y_test, lrs, 
                        n_iters, lr_min, loss_tresh, patience, print_metrics, print_loss, freq_print, gpu, 
                        train_ind_rat, n_ind_points, run_key, results, lambda_lr=lambda_f)
        df = pd.DataFrame.from_dict(results, orient='index')
        if export_results:
            df.to_csv(path)

##----------------------------------------------
## SARCOS
if experiment=='sarcos':
    ndiv = 10 # subsampling factor
    n_ind_points = 500
    from scipy.io import loadmat
    root = '_experiments/SARCOS/'
    if __name__ == "__main__":
        torch.set_default_dtype(torch.float32)
        train_data = loadmat(root + 'sarcos_inv.mat')['sarcos_inv'].astype(np.float32)[::ndiv,:]
        test_data = loadmat(root + 'sarcos_inv_test.mat')['sarcos_inv_test'].astype(np.float32)
        X, Y = train_data[:, :21], train_data[:, 21:]
        X_test, Y_test = test_data[:, :21], test_data[:, 21:]
        Mean, Std = Y.mean(axis=0), Y.std(axis=0)
        Y, Y_test = (Y - Mean) / Std, (Y_test - Mean) / Std
        n_points, n_tasks = Y.shape
        X, Y, X_test, Y_test = torch.as_tensor(X), torch.as_tensor(Y), torch.as_tensor(X_test), torch.as_tensor(Y_test)

        lr_min = 1e-3
        loss_tresh = 1e-4
        patience = 500
        lrs = {'IGP':1e-2,'ICM':1e-2,'LMC':1e-2, 'var':1e-2, 'proj':1e-2, 'diagproj':1e-2, 'oilmm':1e-2, 'bdn':1e-2, 'bdn_diag':1e-2}
        n_iters = {'IGP':10000,'ICM':10000,'LMC':10000,'var':15000, 'proj':10000, 'diagproj':10000, 'oilmm':10000, 'bdn':10000, 'bdn_diag':10000}
        models_with_sched = models_to_run
        v = {
            'q': n_tasks, 
            'lik_rank': 0,
            'void' : [0.]}
        v_vals = {
            'q' : range(1,n_tasks+1), 
            'lik_rank' : [0, n_tasks],
            'void' : [0.]}
        v_test = 'q'
        appendix += 'div{0}'.format(ndiv)
        if n_ind_points is not None:
            appendix += '_{0}ind'.format(n_ind_points)
        path = 'results/realdata_study_'+ experiment + '_' + appendix + '_' + v_test + '_'+ v_test_2 + '.csv'
        print(path + '\n')
        results = {}
        for i_v, vval in enumerate(v_vals[v_test]):
            for i_v2, vval2 in enumerate(v_vals[v_test_2]):
                v[v_test] = vval
                v[v_test_2] = vval2
                q, lik_rank = v['q'], v['lik_rank']
                run_key = '_' + v_test + '_' + v_test_2 + '_{0}_{1}'.format(i_v, i_v2)
                results, models = run_models(models_to_run, models_with_sched, q, lik_rank, n_tasks, n_points, X, Y, X_test, Y_test, lrs, 
                        n_iters, lr_min, loss_tresh, patience, print_metrics, print_loss, freq_print, gpu, 
                        train_ind_rat, n_ind_points, run_key, results)
        df = pd.DataFrame.from_dict(results, orient='index')
        if export_results:
            df.to_csv(path)
    
