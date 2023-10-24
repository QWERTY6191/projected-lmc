# projected-lmc

## Requirements and installation

The package only requires a recent version of gpytorch and of its dependencies (in particular torch). Some auxiliary functions are imported from scikit-learn to perform efficient SVD.
Packages `pandas` and `seaborn` are listed in requirements, but are only used to reproduce graphs from the article.

To install, simply do:
```
pip install projectedlmc
```


## Models construction and usage

Models are built in the standard gpytorch way. File `experiments.py` (reproducing results from the article) gives all necessary examples, but we shortly restate them below.

### Exact single-output model

Not displayed in `experiments.py`, but this model is the building block from which exact LMC/IMC and Projected LMC inherit.
First create a likelihood with :
```
likelihood = gp.likelihoods.GaussianLikelihood()
```
Then create the model :
```
model = ExactGPModel(X, Y, likelihood, mean_type=Mean, kernel_type=kernel, decomp=decomp, ker_kwargs=ker_kwargs)
```
(All fields are described in the documentation).
Note that this class can also generate an independent multitask GP (i.e a batch of independent single-task GPs trained simultaneously) with the optional argument `n_tasks`.

To go into training mode, do :
```
model.train()
likelihood.train()
```
Specify a loss function, an optimizer, and optionnaly a scheduler, with :
```
mll = gp.mlls.ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(lr_min / lr) / n_iter))
```

The training loop then looks like :
```
for i in range(n_iter):
                optimizer.zero_grad()
                with gp.settings.cholesky_jitter(1e-5):
                    output_train = model(X)
                    loss = -mll(output_train, Y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
```
And predictions can be made through :
```
model.eval()
likelihood.eval()
observed_pred = full_likelihood(model(X_test))
pred_y = observed_pred.mean
lower, upper = observed_pred.confidence_region()
```
Note that, as for all other models, the helper methods `.lscales()` and `.outputscale()` help inspect the optimized parameters of the GP:
```
print(model.lscales())
print(model.outputscale())
```

### Exact LMC or IMC model

The data arrays X and Y must have shape `n_points x n_dim` and `n_points x n_tasks` respectively, and predictions will have the same shape convention. All syntax is identical to this of the above single-output GP ; the only difference is in the likelihood, which is now a multitask one :
```
likelihood = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank)
```
The input "rank" is the rank of the cross-tasks noise covariance matrix ; `rank=0` corresponds to a diagonal matrix.

The model also has extra inputs compared to the previous case :
```
model = MultitaskGPModel(X, Y, likelihood, n_tasks=n_tasks, n_latents=n_lat, model_type='LMC', 
                    mean_type=Mean, kernel_type=kernel, decomp=decomp,
                    init_lmc_coeffs=True, fix_diagonal=True, ker_kwargs=ker_kwargs)
```
One of course has to specify the number of tasks and latent functions, but also the type of model between "LMC" and "IMC" (whether or not latent processes have different kernels), and additional optional parameters `init_lmc_coeffs` and `fix_diagonal` - covered in the documentation.    

### variational model

The variational LMC model functions a bit differently. As before, one starts by defining a multitask likelihood :
```
likelihood = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank)
```
And a model :
```
model = VariationalMultitaskGPModel(X, n_tasks=n_tasks, n_latents=n_lat_mod, train_ind_ratio=1.5, seed=0, 
                    distrib=gpytorch.variational.CholeskyVariationalDistribution,
                    init_lmc_coeffs=True, train_y=Y, 
                    mean_type=Mean, kernel_type=kernel,  decomp=decomp, ker_kwargs=ker_kwargs)
```
Here, a `train_ind_ratio` (ratio between the number of training points and the smaller number of inducing points) must be specified as the inducing points approximation is used. Inducing points locations  are learned and initialized with a Sobol' sequence, which random scrambling is controlled by parameter `seed`. If the ratio is set to 1, the behavior is different : inducing points are fixed at the location of input points.
A variational distribution `distrib` must also be specified - see the gpytorch documentation on this topic. The default `gpytorch.variational.CholeskyVariationalDistribution` is a safe pick in all cases.
Another difference with previous cases is that the `train_Y` input is not even necessary : it is only specified in the above example for LMC coefficients initialization. In the same vein, the model doesn't take a likelihood as an input : they remain separated.

The marginal log-likelihood is replaced by a lower bound, the ELBO:
```
mll = gp.mlls.VariationalELBO(likelihood, model, num_data=n_points)
```
And likelihood and model parameters are optimized separately :
```
optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=lr)
```

### Projected models

Projected LMC models, introduced in the afferent article, present other subtleties. First, the likelihood here is a batched gaussian likelihood, which dimension is the number of **latent processes** instead of this of tasks :
```
proj_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_lat]))
```
One is automatically generated by the model if none is provided at instantiation. This latent-level likelihood, independent over latent processes, corresponds to the *projected noise* in the article.
It will be stored in the model attribute `.likelihood`, while the full task-level likelihood can be generated by the method `.full_likelihood()`.

The model has extra options, corresponding to the simplifications depicted in the article (see the documentation):
```
model = ProjectedGPModel(X, Y, n_tasks, n_lat, proj_likelihood=proj_likelihood,
                                   mean_type=Mean,  kernel_type=kernel, decomp=decomp,
                                   BDN=False, diagonal_B=False, scalar_B=False, diagonal_R=False,  
                                   init_lmc_coeffs=True, ker_kwargs=ker_kwargs)
```

The MLL function is here a custom one :
```
mll = ProjectedLMCmll(proj_likelihood, model)
```

The last major difference is that **the loss is not computed by directly calling the model on the training data**. As described in the paper, the projected loss is decomposed into independent single-output losses of the latent processes, plus additional noise-related terms. The training syntax illustrates this notion, using the method `compute_latent_distrib` :
```
for i in range(n_iter):
    optimizer.zero_grad()
    with gp.settings.cholesky_jitter(1e-5):
        output_train = model.compute_latent_distrib(X)
        loss = -mll(output_train, Y)
        loss.backward()
        optimizer.step()
        scheduler.step()
```
`model.compute_latent_distrib(X)` computes the $N_{lat}$-dimensional values of the latent processes at test locations X, while `model(X)` computes the task-level predictions.

At prediction time, in `gpytorch`, modelled noise is usually added to GP covariance by calling `likelihood(model(X_test))`. Here, the full likelihood (by opposition to the projected one) has to be called instead :
```
full_likelihood = model.full_likelihood()
observed_pred = full_likelihood(model(X_test))
```

Finally, note that the various quantities described in the paper ($\mathbf{H, TY, \Sigma_{P}}$...) can be accessed through helper methods : `.projection_matrix()`, `.project_data(train_Y)`, etc.


## Experiments reproduction

### Results generation

File `experiments.py` has been used to generate all experimental data of the paper. It allows one to inspect the effect of varying one parameter (or two, or more with slight modifications) of the synthetic data on several models trained side-by-side. These parameters, described in the article under the same name, are : `n_tasks, n_lat, n_lat_noise, n_points, mu_noise, mu_str, max_scale`. Two additional ones are `n_lat_guess` and `lik_rank`, controling the number of latent functions of the *model* and *model noise* respectively (and not of the data and data noise), making it possible to assess the impact of model misspecification. Default parameter values are contained in the dictionnary `v` and their range of variation for parametric studies in dictionnary 'v_vals'.

To perform a parametric study, just specify the above-described parameter default values and ranges, the included models in the list `models_to_run`, and the test parameter `v_test` (plus eventally a second parameter `v_test_2` for a cross-variables study). A csv file with appropriate name will be outputed, detailing data specifications and performance metrics (defined in the function compute_metrics) for all models ; its name can be modulated through the field 'appendix' or directly. Additional options are available :

+ `loss_thresh` and `patience` : parameters of the stopping criterion (see paper annex D)
+ training parameters `n_iters` and `lrs` (maximal number of iterations and learning rate);
+ `models_with_sched` : what models to endow with the suggested exponential decay scheduler (otherwise, individualized schedulers for each model could be specified by directly editing the script);
+  `print_metrics` : whether to display performance metrics in console after each run;
+  `print_loss` : whether to display loss at the end of each 100 iterations of model training, to inspect convergence;
+  `realistic_H` : whether to replace normally-distributed coefficients of the mixing matrix $\mathbf{H}$ with the (still random) physical toy data described in annex E of the paper;
+  `reject_nonconverged_runs`: in some extreme or misspecified setups, some models frequently jump out of local minima. In most cases they are able to recover afterwards, but sometimes they fail to do so in the prescribed iterations budget, leading to poor predictive performance - a situation which never happened in experiments of the article. If this option is set to `True`, the output csv file is divided into two parts : one including all runs, and another including only converged runs and specifying the number of successful ones. Runs are rejected if accuracy is perceived as abnormally low (errors four times larger than the data noise).
+  `n_random_runs`: number of random repetitions of the test.

Data and model characteristics can of course be modified directly in the body of the script.


### Graph processing

Script `graph_processing.py` has been used to generate all graphs of the paper, and enables easy visualization of experimental results. For inspecting a given simple parametric study, one can simply fill the self-explanatory fields `mods_to_plot`, `var_plot`, `metric`, `scale` (chosing between 'logx', 'logy', 'loglog' and 'lin') and `equal_axes`, and then run the script of call the function `plot_var`. All styles can of course be further customized.
