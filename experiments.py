from projected_lmc import *
import numpy as np
import torch
gpu = False
if gpu:
    torch.set_default_dtype(torch.float32)
else:
    torch.set_default_dtype(torch.float64)
import gpytorch as gp
import pandas as pd
import time
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

##----------------------------------------------------------------------------------------------------------------------
## Setting default parameters

v = {  # default parameter values
'n' : 500,
'p' : 100,
'q' : 25,
'q_guess' : 25,  # q_guess is here to investigate model misspecification (q is the number of latent processes of the data)
'q_noise' : 25,
'q_noise_guess' : 25,  # q_noise_guess is here to investigate model misspecification (q_noise is the number of latent processes of the data)
'mu_noise' : 1e-1,
'mu_str' : 0.9,
'max_scale' : 0.5,
'void' : 0.
}

v_vals = {  # values to be tested
'n' : range(200, 1001, 100),
'p' : range(50, 201, 25),
'q' : range(10, 91, 10),
'q_guess' : range(10, 91, 10),
'q_noise' : range(10, 91, 10),
'q_noise_guess' : range(10, 91, 10),
'mu_noise' : np.logspace(-3, np.log10(0.5), 10),
'mu_str' : np.linspace(1e-3, 1., 10),
'max_scales' : np.linspace(0.1, 2., 10),
'void' : [0.]
}

models_to_run = ['proj', 'diagproj', 'oilmm', 'ICM', 'var', 'bdn', 'bdn_diag']
models_with_sched = models_to_run
v_test = 'void' # replace by the parameter to be tested (or perform a one-point experiment)
v_test_2 = 'void' # if not 'void', interaction between two parameters is tested
train_ind_rat = 1.5 # ratio between number of training points and number of inducing points for the variational model
n_ind_points = None  # if not None, all models will use an inducing point approximation with ths number of inducing points
n_random_runs = 1 # number of random repetitions of the experiment (each one with different data)
##----------------------------------------------------------------------------------------------------------------------

## Reproducing the paper's results

v['mu_str'], v_test, n_random_runs = 0.99, 'mu_noise', 40 # Figure 1
# v_test, n_random_runs = 'mu_str', 20 # Figure 2
# v_test, n_random_runs = 'q_noise', 10 # Figure 3
# v_test, n_random_runs = 'p', 20 # Figure 4, 5a/ and 6a/
# v_test, n_random_runs = 'q', 20 # Figure 5b/ and 6b/
##----------------------------------------------------------------------------------------------------------------------

## Results formatting
print_metrics=True  # if True, performance metrics are printed at each run (dosen't affect exported results)
print_loss=True # if True, loss is printed after freq_print iteration (dosen't affect exported results)
freq_print=1000
reject_nonconverged_runs = False 
# if True, output dataframe will contain one part with all results stored, and one where results with errors much larger 
# than the data noise are discarded. Not used in the paper (no convergence issues)
appendix = '_reject' if reject_nonconverged_runs else ''
appendix += '' # to further customize experiment name
if n_ind_points is not None:
    appendix += '_{0}ind'.format(n_ind_points)
landmarks = [1] + list(range(10, n_random_runs + 1, 10))
path = 'results/parameter_study_' + v_test + '_' + v_test_2 + appendix + '.csv'
export_results = True

##----------------------------------------------------------------------------------------------------------------------

## Other settings
min_scale = 0.01
n_test = 2500 # number of test points
lr_min = 1e-3
loss_tresh = 1e-4 # threshold for loss plateau detection
patience = 500 # number of iterations without loss decrease before stopping training
lrs = {'ICM':1e-2, 'var':1e-2, 'proj':1e-2, 'diagproj':1e-2, 'oilmm':1e-2, 'bdn':1e-2, 'bdn_diag':1e-2}
# n_iters = {'ICM':10000, 'var':10000, 'proj':10000, 'diagproj':10000, 'oilmm':10000, 'bdn':10000, 'bdn_diag':10000}
n_iters = {'ICM':100, 'var':100, 'proj':100, 'diagproj':100, 'oilmm':100, 'bdn':100, 'bdn_diag':100}
all_models = ['proj', 'diagproj', 'oilmm', 'ICM', 'var', 'bdn', 'bdn_diag']

##------------------------------------------------------------------------------
def compute_metrics(y_test, y_pred, lower, upper, noise, loss, H_guess_hid, n_iter, train_time, pred_time, print_metrics=True):
    delta = y_test - y_pred
    errs_abs = torch.abs(delta)
    sigma_pred = (upper - lower) / 4
    alpha_CI = torch.mean((torch.abs(pred_y - y_test) < 2 * sigma_pred).float())
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
    metrics['noise'] = noise_full.cpu().numpy()  # not a reliable metric, different treatment and results between models
    metrics['R2'] = R2_list.mean().cpu().numpy()
    metrics['RMSE'] = torch.sqrt(err2.mean()).cpu().numpy()
    metrics['mean_err_abs'], metrics['max_err_abs'] = errs_abs.mean(), errs_abs.max()
    metrics['mean_err_quant05'], metrics['mean_err_quant95'], metrics['mean_err_quant99'] = np.quantile(errs_abs, np.array([0.05, 0.95, 0.99]))
    metrics['mean_sigma'] = sigma_pred.mean().cpu().numpy()
    metrics['PVA'] = PVA_list.mean().cpu().numpy()
    metrics['alpha_CI'] = alpha_CI.mean().cpu().numpy()  # proportion of points within 2 sigma (should be around 0.95)
    if print_metrics:
        for key, value in metrics.items():
            print(key, value)
    return metrics

##------------------------------------------------------------------------------
## Running experiments
results = {}
n_succes_runs = dict(zip(models_to_run, np.zeros(len(models_to_run))))
if __name__=='__main__':
    print(path + '\n')
    for i_run in range(n_random_runs):
        print('\n Random run number {} : \n'.format(i_run))
        np.random.seed(i_run)
        torch.manual_seed(i_run)
        for i_v, vval in enumerate(v_vals[v_test]):
            for i_v2, vval2 in enumerate(v_vals[v_test_2]):
                v[v_test] = vval
                v[v_test_2] = vval2
                p, q, q_noise, n, mu_noise, mu_str, max_scale, q_noise_guess = \
                    v['p'], v['q'], v['q_noise'], v['n'], v['mu_noise'], v['mu_str'], v['max_scale'], v['q_noise_guess']
                run_key = '_' + v_test + '_' + v_test_2 + '_{0}_{1}'.format(i_v, i_v2)
                ##------------------------------------------------------------------

                ## Generating artificial data
                lscales = torch.linspace(min_scale, max_scale, q)
                lscales_hid = torch.linspace(min_scale, max_scale, q_noise)
                ker_list = [gp.kernels.MaternKernel() for i in range(q)]
                ker_list_hid = [gp.kernels.MaternKernel() for i in range(q_noise)]
                for i in range(q):
                    ker_list[i].lengthscale = lscales[i]
                for i in range(q_noise):
                    ker_list_hid[i].lengthscale = lscales_hid[i]

                X_train = torch.linspace(-1, 1, n)
                X_test = 2*torch.rand(n_test) - 1
                X = torch.cat([X_train, X_test], dim=0)
                H_true = torch.randn(size=(q, p))
                lat_gp_dist = [gp.distributions.MultivariateNormal(torch.zeros_like(X), kernel(X)) for kernel in ker_list]
                gp_vals = torch.stack([dist.sample() for dist in lat_gp_dist])
                Y_sig = gp_vals.T @ H_true * (1 - mu_noise)
                H_true = H_true.numpy()

                ## structured noise
                H_true_hid = torch.randn(size=(q_noise, p))
                gp_vals_hid_com = torch.randn((q_noise, n + n_test))
                Y_noise_com = gp_vals_hid_com.T @ H_true_hid * mu_str

                ## unstructured noise
                noise_levels = torch.rand(p) + 0.1
                gp_vals_hid_spec = torch.sqrt(noise_levels)[:,None]*torch.randn((p, n + n_test)) # homosk noise
                Y_noise_spec = gp_vals_hid_spec.T * (1 - mu_str)

                Y_noise = (Y_noise_com + Y_noise_spec) * mu_noise
                sig_true = H_true_hid.T @ H_true_hid * mu_str + torch.diag_embed(noise_levels) * (1 - mu_str)
                Y = Y_sig + Y_noise
                X = X[:,None]
                X_test, Y_test = X[n:], Y[n:]
                X, Y = X[:n], Y[:n]
                ##------------------------------------------------------------------

                ## Defining models
                kernel_type = gp.kernels.MaternKernel
                mean_type = gp.means.ZeroMean
                decomp = None
                ker_kwargs = {}
                likelihoods, models, mlls, optimizers, schedulers = {}, {}, {}, {}, {}
                if v_test!= 'q_noise_guess':  # if q_noise_guess is not the parameter to be tested, we use a full rank noise (general Sigma matrix)
                    q_noise_guess, v['q_noise_guess'] = p, p
                q_mod = q if v_test!= 'q_guess' else v['q_guess'] # if q_guess is not the parameter to be tested, we use the true value (no misspecification)

                if 'ICM' in models_to_run:
                    likelihoods['ICM'] = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=p, rank=q_noise_guess)
                    models['ICM'] = MultitaskGPModel(X, Y, likelihoods['ICM'], n_tasks=p, init_lmc_coeffs=True,
                                        n_latents=q_mod, mean_type=mean_type, kernel_type=kernel_type, decomp=decomp,
                                        fix_diagonal=False, model_type='ICM', n_inducing_points=n_ind_points, ker_kwargs=ker_kwargs)

                if 'var' in models_to_run:
                    likelihoods['var'] = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=p, rank=q_noise_guess)
                    models['var'] = VariationalMultitaskGPModel(X, train_y=Y, n_tasks=p, init_lmc_coeffs=True,
                                        mean_type=mean_type, kernel_type=kernel_type, n_latents=q_mod, decomp=decomp,
                                        train_ind_ratio=train_ind_rat, seed=0, distrib=gp.variational.CholeskyVariationalDistribution,
                                        ker_kwargs=ker_kwargs)
                    
                if 'proj' in models_to_run:
                    models['proj'] = ProjectedGPModel(X, Y, p, q_mod, proj_likelihood=None,
                                                    mean_type=mean_type,  kernel_type=kernel_type, decomp=decomp,
                                                    init_lmc_coeffs=True, BDN=False, diagonal_B=False, diagonal_R=False, 
                                                    scalar_B=False, ker_kwargs=ker_kwargs, n_inducing_points=n_ind_points)
                    likelihoods['proj'] = models['proj'].likelihood

                if 'diagproj' in models_to_run:
                    models['diagproj'] = ProjectedGPModel(X, Y, p, q_mod, proj_likelihood=None,
                                                    mean_type=mean_type, kernel_type=kernel_type, decomp=decomp,
                                                    init_lmc_coeffs=True, BDN=False, diagonal_B=True, diagonal_R=False,
                                                    scalar_B=False, ker_kwargs=ker_kwargs, n_inducing_points=n_ind_points)
                    likelihoods['diagproj'] = models['diagproj'].likelihood

                if 'oilmm' in models_to_run:
                    models['oilmm'] = ProjectedGPModel(X, Y, p, q_mod, proj_likelihood=None,
                                                    mean_type=mean_type, kernel_type=kernel_type, decomp=decomp,
                                                    init_lmc_coeffs=True, BDN=True, diagonal_B=True, diagonal_R=True,
                                                    scalar_B=True, ker_kwargs=ker_kwargs, n_inducing_points=n_ind_points)
                    likelihoods['oilmm'] = models['oilmm'].likelihood

                if 'bdn' in models_to_run:
                    models['bdn'] = ProjectedGPModel(X, Y, p, q_mod, proj_likelihood=None,
                                                    mean_type=mean_type, kernel_type=kernel_type, decomp=decomp,
                                                    init_lmc_coeffs=True, BDN=True, diagonal_B=False, diagonal_R=False,
                                                    scalar_B=False, ker_kwargs=ker_kwargs, n_inducing_points=n_ind_points)
                    likelihoods['bdn'] = models['bdn'].likelihood

                if 'bdn_diag' in models_to_run:
                    models['bdn_diag'] = ProjectedGPModel(X, Y, p, q_mod, proj_likelihood=None,
                                                    mean_type=mean_type, kernel_type=kernel_type, decomp=decomp,
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
                    models[name].train()
                    likelihoods[name].train()

                if 'ICM' in models_to_run:
                    mlls['ICM'] = gp.mlls.ExactMarginalLogLikelihood(likelihoods['ICM'], models['ICM'])
                    optimizers['ICM'] = torch.optim.AdamW(models['ICM'].parameters(), lr=lrs['ICM'])  # Includes GaussianLikelihood parameters
                if 'var' in models_to_run:
                    mlls['var'] = gp.mlls.VariationalELBO(likelihoods['var'], models['var'], num_data=n)
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
                        schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(optimizers[name], gamma=np.exp(np.log(lr_min / lrs[name]) / n_iters[name]))
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
                            # def closure():  # for lfbgs optimizer
                            #     optimizers[name].zero_grad()
                            #     output_train = models[name](X)
                            #     loss = -mlls[name](output_train, Y)
                            #     loss.backward()
                            #     return loss
                            # optimizers[name].step(closure)
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
                # All these algebra options have been tested to have little impact on results.
                # The skip_posterior_variances option is here to be able to compute posterior mean for model ICM, even when
                # covariance computation would saturate memory. It should be deactivated when possible.
                    with torch.no_grad(),\
                                gp.settings.skip_posterior_variances(state=(name=='ICM')), \
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
                        H_guess_hid = full_likelihood.task_noise_covar_factor.squeeze()
                        observed_pred = full_likelihood(models[name](X_test))
                        pred_time = time.time() - start
                        pred_y = observed_pred.mean
                        lower, upper = observed_pred.confidence_region()
                        global_noise = full_likelihood.noise.squeeze() if hasattr(full_likelihood, 'noise') else 1.
                        ##------------------------------------------------------------------

                        ## Computing, displaying and storing performance metrics
                        if print_metrics:
                            if hasattr(likelihoods[name], 'task_noises'):
                                print('Noises for {0} model:'.format(name), global_noise * likelihoods[name].task_noises.cpu().numpy()) # for non-projected models if their likelihood is diagonal
                            elif hasattr(models[name], 'full_likelihood'):  # for projected models
                                print('Noises for {0} model:'.format(name), likelihoods[name].noise.squeeze().cpu().numpy())
                        metrics = compute_metrics(Y_test, pred_y, lower, upper, global_noise, last_losses[name], H_guess_hid, 
                                                  effective_n_iters[name], times[name], pred_time, print_metrics=print_metrics)
                        metrics.update(v)
                        metrics['model'] = name
                        results[name + run_key] = metrics
        ##------------------------------------------------------------------
        ## Exporting results
        if i_run == 0:
            df = pd.DataFrame.from_dict(results, orient='index')
            updated_cols = df.columns.difference(list(v.keys()) + ['model'])
            df[updated_cols] = 0.
            if reject_nonconverged_runs:
                df_conv = df.copy()
                df_conv.rename(mapper={label: label + '_conv' for label in df.index}, axis='index', inplace=True)
                n_succes_runs = dict(zip(df_conv.index, np.zeros(len(df.index))))

        df[updated_cols] += pd.DataFrame.from_dict(results, orient='index')[updated_cols]
        if reject_nonconverged_runs:
            for label in df.index:
                if results[label]['mean_err_abs'] < max(0.2, 5. * mu_noise):
                    df_conv.loc[label + '_conv', updated_cols] += pd.DataFrame.from_dict(results, orient='index').loc[
                        label, updated_cols]
                    n_succes_runs[label + '_conv'] += 1

        if (i_run + 1) in landmarks:
            df_part = df.copy()
            df_part[updated_cols] = df_part[updated_cols] / (i_run + 1)
            if reject_nonconverged_runs:
                df_conv_part = df_conv.copy()
                df_conv_part['n_sucess_runs'] = pd.DataFrame.from_dict(n_succes_runs, orient='index')
                df_part['n_sucess_runs'] = pd.DataFrame((i_run + 1) * np.ones(len(n_succes_runs)), index=df.index)
                for label in df_conv_part.index:
                    df_conv_part.loc[label, updated_cols] = df_conv_part.loc[label, updated_cols] / n_succes_runs[label]
                df_part = pd.concat([df_part, df_conv_part], axis=0)
            
            if export_results:
                partial_path = path[:-4] + '_{0}runs.csv'.format(i_run + 1)
                df_part.to_csv(partial_path)
