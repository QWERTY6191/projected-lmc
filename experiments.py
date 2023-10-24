from projected_lmc import *
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import gpytorch as gp
import pandas as pd
import time
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

def compute_metrics(y_test, y_pred, lower, upper, noise, loss, lscales_true, lscales_guess,
                    H_true, H_guess, H_true_hid, H_guess_hid, n_iter, train_time, pred_time, print_metrics=True):
    delta = y_test - y_pred
    errs_abs = torch.abs(delta)
    sigma_pred = (upper - lower) / 4
    alpha_CI = torch.mean((torch.abs(pred_y - y_test) < 2 * sigma_pred).float())
    err2 = errs_abs ** 2
    Q2_list = 1 - torch.mean(err2, dim=0) / torch.var(y_test, dim=0)
    PVA_list = torch.log(torch.mean(err2 / sigma_pred ** 2, dim=0))
    
    # for comparison, we reorganize the infered parameters so that they match the order of the true values
    idx = torch.argsort(lscales_guess)
    H_guess, lscales_guess = H_guess[idx], lscales_guess[idx]
    H_corr = 0.
    for i in range(len(H_true)): # the sign indeterminacy of the rows of H_guess imposes to compute the correlation row after row
        H_corr += np.abs(np.dot(H_true[i], H_guess[i]))
    H_corr = H_corr / (np.linalg.norm(H_guess) * np.linalg.norm(H_true))

    idx_true = torch.flip(torch.argsort(torch.linalg.norm(H_true_hid, dim=1)),(0,))
    idx_guess = torch.flip(torch.argsort(torch.linalg.norm(H_guess_hid, dim=1)),(0,))
    H_true_hid, H_guess_hid = H_true_hid[idx_true], H_guess_hid[idx_guess]
    H_hid_corr = 0.
    for i in range(len(H_true_hid)): # the sign indeterminacy of the rows of H_guess imposes to compute the correlation row after row
        H_hid_corr += torch.abs(torch.dot(H_true_hid[i], H_guess_hid[i]))
    H_hid_corr = H_hid_corr / (torch.linalg.norm(H_guess_hid) * torch.linalg.norm(H_true_hid))
    noise_full = noise * (H_guess_hid**2).sum()  # mean of the diagonal coefficients

    errs_abs = errs_abs.numpy()
    metrics = {}
    metrics['n_iter'] = n_iter
    metrics['train_time'] = train_time
    metrics['pred_time'] = pred_time
    metrics['loss'] = loss
    metrics['noise'] = noise_full.numpy()
    metrics['Q2'] = Q2_list.mean().numpy()
    metrics['mean_err_abs'], metrics['max_err_abs'] = errs_abs.mean(), errs_abs.max()
    metrics['mean_err_quant05'], metrics['mean_err_quant95'], metrics['mean_err_quant99'] = np.quantile(errs_abs, np.array([0.05, 0.95, 0.99]))
    metrics['mean_sigma'] = sigma_pred.mean().numpy()
    metrics['PVA'] = PVA_list.mean().numpy()
    metrics['alpha_CI'] = alpha_CI.mean().numpy()
    metrics['H_corr'] = H_corr
    metrics['H_hid_corr'] = H_hid_corr.cpu().detach().numpy()
    metrics['lscales_mismatch'] = torch.linalg.norm(lscales_true - lscales_guess).numpy()
    if print_metrics:
        for key, value in metrics.items():
            print(key, value)
    return metrics

##------------------------------------------------------------------------------
## Setting parameters

min_scale = 0.1

lr_min = 1e-3
loss_thresh = 1e-4
patience = 500
lrs = {'exact':1e-2, 'var':1e-2, 'proj':1e-2, 'diagproj':1e-2, 'oilmm':1e-2, 'bdn':1e-2, 'bdn_diag':1e-2}
n_iters = {'exact':5000, 'var':5000, 'proj':5000, 'diagproj':5000, 'oilmm':5000, 'bdn':5000, 'bdn_diag':5000}
all_models = ['proj', 'diagproj', 'oilmm', 'exact', 'var', 'bdn', 'bdn_diag']

v = {
'n_tasks' : 10,
'n_lat' : 2,
'n_lat_guess' : 2,
'n_lat_noise' : 3,
'n_points' : 50,
'mu_noise' : 5e-2,
'mu_str' : 0.99,
'max_scale' : 0.5,
'lik_rank' : 10,
'void' : 0.
}

v_vals = {
'n_tasks' : range(4, 21, 2),
'n_lat' : range(2, 11),
'n_lat_guess' : range(1, 11),
'n_lat_noise' : range(1, 11),
'n_points' : range(20, 171, 30),
'mu_noise' : np.logspace(-3, np.log10(0.5), 10),
'mu_str' : np.linspace(1e-3, 1., 10),
'max_scales' : np.linspace(0.1, 2., 10),
'lik_rank' : range(11),
'void' : [0.]
}

results = {}

##----------------------------------------------------------------------------------------------------------------------
models_to_run = ['proj', 'diagproj', 'oilmm', 'exact', 'var', 'bdn', 'bdn_diag']
models_with_sched = models_to_run
v_test = 'n_lat'
v_test_2 = 'void'
print_metrics=True
print_loss=False
realistic_H=True

n_random_runs = 100
reject_nonconverged_runs = False
export_results = True
appendix = '_reject' if reject_nonconverged_runs else ''
appendix += ''   # to further customize experiment name
path = 'results_lmc/parameter_study_' + v_test + '_' + v_test_2 + appendix + '.csv'

n_succes_runs = dict(zip(models_to_run, np.zeros(len(models_to_run))))
if __name__=='__main__':
    for i_run in range(n_random_runs):
        np.random.seed(i_run)
        torch.manual_seed(i_run)
        for i_v, vval in enumerate(v_vals[v_test]):
            for i_v2, vval2 in enumerate(v_vals[v_test_2]):
                v[v_test] = vval
                v[v_test_2] = vval2
                n_tasks, n_lat, n_lat_noise, n_points, mu_noise, mu_str, max_scale, lik_rank = \
                    v['n_tasks'], v['n_lat'], v['n_lat_noise'], v['n_points'], v['mu_noise'], v['mu_str'], v['max_scale'], v['lik_rank']
                run_key = '_' + v_test + '_' + v_test_2 + '_{0}_{1}'.format(i_v, i_v2)
                ##------------------------------------------------------------------
                ## Generating artificial data

                lscales = torch.linspace(min_scale, max_scale, n_lat)
                lscales_hid = torch.linspace(min_scale, max_scale, n_lat_noise)
                ker_list = [gp.kernels.MaternKernel() for i in range(n_lat)]
                ker_list_hid = [gp.kernels.MaternKernel() for i in range(n_lat_noise)]
                for i in range(n_lat):
                    ker_list[i].lengthscale = lscales[i]
                for i in range(n_lat_noise):
                    ker_list_hid[i].lengthscale = lscales_hid[i]

                X_train = torch.linspace(-1, 1, n_points)
                X_test = 2*torch.rand(500) - 1
                X = torch.cat([X_train, X_test], dim=0)
                if realistic_H:
                    captor_coords = np.zeros((n_tasks, 2))
                    captor_coords[:,0] = np.linspace(-1,1, n_tasks)
                    sources_coords = np.random.randn(n_lat, 2)
                    dists = cdist(sources_coords, captor_coords)
                    amps = np.linspace(1, n_lat, n_lat).astype(float)
                    H_true = torch.as_tensor(amps[:,None] / dists)
                else:
                    H_true = torch.randn(size=(n_lat, n_tasks))

                lat_gp_dist = [gp.distributions.MultivariateNormal(torch.zeros_like(X), kernel(X)) for kernel in ker_list]
                gp_vals = torch.stack([dist.sample() for dist in lat_gp_dist])
                Y_sig = gp_vals.T @ H_true * (1 - mu_noise)
                H_true = H_true.numpy()

                ## structured noise
                H_true_hid = torch.randn(size=(n_lat_noise, n_tasks))
                # lat_gp_dist_hid = [gp.distributions.MultivariateNormal(torch.zeros_like(X), kernel(X,X)) for kernel in ker_list_hid] # heterosk noise
                lat_gp_dist_hid_com = [gp.distributions.MultivariateNormal(torch.zeros_like(X), torch.eye(len(X))) for i in range(n_lat_noise)] # homosk noise
                gp_vals_hid_com = torch.stack([dist.sample() for dist in lat_gp_dist_hid_com])
                Y_noise_com = gp_vals_hid_com.T @ H_true_hid * mu_str

                ## unstructured noise
                noise_levels = torch.rand(n_tasks) + 0.1
                gp_dist_hid_spec = [gp.distributions.MultivariateNormal(torch.zeros_like(X), noise_levels[i] * torch.eye(len(X))) for i in range(n_tasks)] # homosk noise
                gp_vals_hid_spec = torch.stack([dist.sample() for dist in gp_dist_hid_spec])
                Y_noise_spec = gp_vals_hid_spec.T * (1 - mu_str)

                Y_noise = (Y_noise_com + Y_noise_spec) * mu_noise
                sig_true = H_true_hid.T @ H_true_hid * mu_str + torch.diag_embed(noise_levels) * (1 - mu_str)
                Y = Y_sig + Y_noise
                # plt.figure()   # for synthetic data vizualization
                # for i in range(n_lat):
                #     plt.plot(X, gp_vals[i,:])
                # plt.show()

                X = X[:,None]
                X_test, Y_test = X[n_points:], Y[n_points:]
                X, Y = X[:n_points], Y[:n_points]

                ##------------------------------------------------------------------
                ## Defining models

                m_names = ['proj', 'diagproj', 'oilmm', 'exact', 'var', 'bdn', 'bdn_diag']
                kernel = gp.kernels.MaternKernel
                Mean = gp.means.ZeroMean
                decomp = None
                ker_kwargs = {}
                likelihoods, models, mlls, optimizers, schedulers = {}, {}, {}, {}, {}
                if v_test!= 'lik_rank':
                    lik_rank, v['lik_rank'] = n_tasks, n_tasks
                if v_test!= 'n_lat_guess':
                    n_lat_mod = n_lat
                else:
                    n_lat_mod = v['n_lat_guess']

                likelihoods['exact'] = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank)
                models['exact'] = MultitaskGPModel(X, Y, likelihoods['exact'], n_tasks=n_tasks, init_lmc_coeffs=True,
                                    n_latents=n_lat_mod, mean_type=Mean, kernel_type=kernel, decomp=decomp,
                                    fix_diagonal=True, model_type='LMC', ker_kwargs=ker_kwargs)

                likelihoods['var'] = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank)
                models['var'] = VariationalMultitaskGPModel(X, train_y=Y, n_tasks=n_tasks, init_lmc_coeffs=True,
                                    mean_type=Mean, kernel_type=kernel, n_latents=n_lat_mod, decomp=decomp,
                                    train_ind_ratio=1.5, seed=0, distrib=gpytorch.variational.CholeskyVariationalDistribution,
                                    ker_kwargs=ker_kwargs)

                models['proj'] = ProjectedGPModel(X, Y, n_tasks, n_lat_mod, proj_likelihood=None,
                                                   mean_type=Mean,  kernel_type=kernel, decomp=decomp,
                                                   init_lmc_coeffs=True, BDN=False, diagonal_B=False, diagonal_R=False, 
                                                  scalar_B=False, ker_kwargs=ker_kwargs)
                likelihoods['proj'] = models['proj'].likelihood

                models['diagproj'] = ProjectedGPModel(X, Y, n_tasks, n_lat_mod, proj_likelihood=None,
                                                   mean_type=Mean, kernel_type=kernel, decomp=decomp,
                                                   init_lmc_coeffs=True, BDN=False, diagonal_B=True, diagonal_R=False,
                                                scalar_B=False, ker_kwargs=ker_kwargs)
                likelihoods['diagproj'] = models['diagproj'].likelihood

                models['oilmm'] = ProjectedGPModel(X, Y, n_tasks, n_lat_mod, proj_likelihood=None,
                                                   mean_type=Mean, kernel_type=kernel, decomp=decomp,
                                                   init_lmc_coeffs=True, BDN=True, diagonal_B=True, diagonal_R=True,
                                                   scalar_B=True, ker_kwargs=ker_kwargs)
                likelihoods['oilmm'] = models['oilmm'].likelihood

                models['bdn'] = ProjectedGPModel(X, Y, n_tasks, n_lat_mod, proj_likelihood=None,
                                                   mean_type=Mean, kernel_type=kernel, decomp=decomp,
                                                   init_lmc_coeffs=True, BDN=True, diagonal_B=False, diagonal_R=False,
                                                   scalar_B=False, ker_kwargs=ker_kwargs)
                likelihoods['bdn'] = models['bdn'].likelihood

                models['bdn_diag'] = ProjectedGPModel(X, Y, n_tasks, n_lat_mod, proj_likelihood=None,
                                                   mean_type=Mean, kernel_type=kernel, decomp=decomp,
                                                   init_lmc_coeffs=True, BDN=True, diagonal_B=True, diagonal_R=False,
                                                   scalar_B=False, ker_kwargs=ker_kwargs)
                likelihoods['bdn_diag'] = models['bdn_diag'].likelihood

                ##------------------------------------------------------------------
                ## Configuring optimization

                for name in m_names:
                    models[name].train()
                    likelihoods[name].train()

                mlls['exact'] = gp.mlls.ExactMarginalLogLikelihood(likelihoods['exact'], models['exact'])
                mlls['var'] = gp.mlls.VariationalELBO(likelihoods['var'], models['var'], num_data=n_points)
                mlls['proj'] = ProjectedLMCmll(likelihoods['proj'], models['proj'])
                mlls['diagproj'] = ProjectedLMCmll(likelihoods['diagproj'], models['diagproj'])
                mlls['oilmm'] = ProjectedLMCmll(likelihoods['oilmm'], models['oilmm'])
                mlls['bdn'] = ProjectedLMCmll(likelihoods['bdn'], models['bdn'])
                mlls['bdn_diag'] = ProjectedLMCmll(likelihoods['bdn_diag'], models['bdn_diag'])

                optimizers['exact'] = torch.optim.AdamW(models['exact'].parameters(), lr=lrs['exact'])  # Includes GaussianLikelihood parameters
                optimizers['var'] = torch.optim.AdamW([{'params': models['var'].parameters()}, {'params': likelihoods['var'].parameters()}], lr=lrs['var'])
                optimizers['proj'] = torch.optim.AdamW(models['proj'].parameters(), lr=lrs['proj'])
                optimizers['diagproj'] = torch.optim.AdamW(models['diagproj'].parameters(), lr=lrs['diagproj'])
                optimizers['oilmm'] = torch.optim.AdamW(models['oilmm'].parameters(), lr=lrs['oilmm'])
                optimizers['bdn'] = torch.optim.AdamW(models['bdn'].parameters(), lr=lrs['bdn'])
                optimizers['bdn_diag'] = torch.optim.AdamW(models['bdn_diag'].parameters(), lr=lrs['bdn_diag'])

                for name in m_names:
                    if name in models_with_sched:
                        schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(optimizers[name], gamma=np.exp(np.log(lr_min / lrs[name]) / n_iters[name]))
                # schedulers['var'] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers['var'], T_0=400, T_mult=2, eta_min=1e-4)  # other schedulers have been tried as well

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
                            if hasattr(models[name], 'compute_latent_distrib'): # projected models see inputs only via their latent functions
                                output_train = models[name].compute_latent_distrib(X)
                            else:
                                output_train = models[name](X)
                            loss = -mlls[name](output_train, Y)
                            if print_loss and i%100==0:
                                print(loss.item())
                            loss.backward()

                            # def closure():  # for lfbgs optimizer
                            #     optimizers[name].zero_grad()
                            #     if hasattr(models[name], 'compute_latent_distrib'):
                            #         output_train = models[name].compute_latent_distrib(X)
                            #     else:
                            #         output_train = models[name](X)
                            #     loss = -mlls[name](output_train, Y)
                            #     loss.backward()
                            #     return loss
                            # optimizers[name].step(closure)

                            optimizers[name].step()
                        if name in schedulers:
                            schedulers[name].step()

                        new_loss = loss.item()
                        if i>0 and np.abs(last_losses[name] - new_loss) < loss_thresh:
                            plateau_id += 1
                            if plateau_id > patience :
                                effective_n_iters[name] = i
                                break
                        last_losses[name] = new_loss
                    times[name] = time.time() - start

                ##------------------------------------------------------------------
                ## Making predictions

                for name in m_names:
                    models[name].eval()
                    likelihoods[name].eval()

                # all these algebra options have been tested to have little impact on results
                with torch.no_grad(), \
                            gp.settings.skip_logdet_forward(state=True), \
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

                    for name in models_to_run:
                        print(' \n Making predictions for {0} model...'.format(name))
                        start = time.time()
                        if hasattr(models[name], 'full_likelihood'):  # we have to compute the full likelihood of projected models
                            full_likelihood = models[name].full_likelihood()
                        else:
                            full_likelihood = likelihoods[name]
                        if hasattr(models[name], 'variational_strategy'):
                            H_guess = models[name].variational_strategy.lmc_coefficients.numpy()
                        else:
                            H_guess = models[name].lmc_coefficients().numpy()
                        H_guess_hid = full_likelihood.task_noise_covar_factor.squeeze()
                        observed_pred = full_likelihood(models[name](X_test))
                        pred_time = time.time() - start
                        pred_y = observed_pred.mean
                        lower, upper = observed_pred.confidence_region()
                        lscales_guess = models[name].lscales().squeeze()
                        global_noise = full_likelihood.noise.squeeze() if hasattr(full_likelihood, 'noise') else 1.
                        ##------------------------------------------------------------------
                        ## Computing, displaying and storing performance metrics

                        if print_metrics:
                            if hasattr(likelihoods[name], 'task_noises'):
                                print('Noises for {0} model:'.format(name), global_noise * likelihoods[name].task_noises.numpy()) # for non-projected models if their likelihood is diagonal
                            elif hasattr(models[name], 'full_likelihood'):  # for projected models
                                print('Noises for {0} model:'.format(name), likelihoods[name].noise.squeeze().numpy())
                        metrics = compute_metrics(Y_test, pred_y, lower, upper, global_noise, last_losses[name], lscales, lscales_guess,
                                                       H_true, H_guess, H_true_hid, H_guess_hid, effective_n_iters[name], times[name], pred_time, print_metrics=print_metrics)
                        metrics.update(v)
                        metrics['model'] = name
                        results[name + run_key] = metrics

        ##------------------------------------------------------------------
        ## Exporting results
        if i_run == 0:
            df = pd.DataFrame.from_dict(results, orient='index')
            updated_cols = df.columns.difference(list(v.keys()) + ['model'])
            df[updated_cols] = 0.  # for the elimination of eventual divergent runs
            if reject_nonconverged_runs:
                df_conv = df.copy()
                df_conv.rename(mapper={label: label + '_conv' for label in df.index}, axis='index', inplace=True)
                n_succes_runs = dict(zip(df_conv.index, np.zeros(len(df.index))))

        df[updated_cols] += pd.DataFrame.from_dict(results, orient='index')[updated_cols]
        if reject_nonconverged_runs:
            for label in df.index:
                if results[label]['mean_err_abs'] < max(0.2, 4. * mu_noise):
                    df_conv.loc[label + '_conv', updated_cols] += pd.DataFrame.from_dict(results, orient='index').loc[
                        label, updated_cols]
                    n_succes_runs[label + '_conv'] += 1

    df[updated_cols] = df[updated_cols] / (i_run + 1)
    if reject_nonconverged_runs:
        df_conv['n_sucess_runs'] = pd.DataFrame.from_dict(n_succes_runs, orient='index')
        df['n_sucess_runs'] = pd.DataFrame((i_run + 1) * np.ones(len(n_succes_runs)), index=df.index)
        for label in df_conv.index:
            df_conv.loc[label, updated_cols] = df_conv.loc[label, updated_cols] / n_succes_runs[label]
        df = pd.concat([df, df_conv], axis=0)

    if export_results:
        df.to_csv(path)