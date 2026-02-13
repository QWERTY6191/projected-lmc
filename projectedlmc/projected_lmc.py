import math
import copy
import torch
import gpytorch as gp
from gpytorch.means.mean import Mean
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
import numpy as np
from torch import Tensor
from scipy.stats import qmc
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator, \
KroneckerProductAddedDiagLinearOperator, KroneckerProductDiagLinearOperator, ConstantDiagLinearOperator, \
DiagLinearOperator, PsdSumLinearOperator
from linear_operator.operators.dense_linear_operator import to_linear_operator
from functools import reduce #, lru_cache
from typing import Union, List
import psutil
import warnings

##----------------------------------------------------------------------------------------------------------------------

## Custom means

class SplineKernel(gp.kernels.Kernel):
    is_stationary = True

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return (1 + x1**2 + x1**3 / 3).prod(dim=-1)
        mins = torch.min(x1.unsqueeze(-2), x2.unsqueeze(-3))
        maxes = torch.max(x1.unsqueeze(-2), x2.unsqueeze(-3))
        oned_vals = 1 + mins*maxes + 0.5 * mins**2 * (maxes - mins/3) 
        return oned_vals.prod(dim=-1)

class PolynomialMean(gp.means.mean.Mean):
    def __init__( self, input_size, batch_shape=torch.Size(), bias=True, degree=3):
        super().__init__()
        for i in range(degree+1):
            self.register_parameter(name="weights_{0}".format(i),
                                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None
        self.degree = degree

    def forward( self, x: Tensor)-> Tensor:
        """

        Args:
            x: input data to be evaluated at

        Returns:
            A tensor of the values of the mean function at evaluation points.
        """
        res = 0
        for i in range(1, self.degree + 1):
            res += (x ** i).matmul(getattr(self, 'weights_{0}'.format(i))).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

class LinearMean(gp.means.mean.Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

    def basis_matrix( self, x ):
        return torch.hstack([x, torch.ones((len(x), 1), device=x.device)])

##----------------------------------------------------------------------------------------------------------------------
## utilitaries

class LeaveOneOutPseudoLikelihood(gp.mlls.exact_marginal_log_likelihood.ExactMarginalLogLikelihood):
    def __init__( self, likelihood, model, train_x, train_y):
        super().__init__(likelihood=likelihood, model=model)
        self.likelihood = likelihood
        self.model = model
        self.train_x = train_x
        self.train_y = train_y

    def forward( self, function_dist: gp.distributions.MultivariateNormal, target: Tensor, *params ) -> Tensor:
        output = self.likelihood(function_dist, *params)
        sigma2, target_minus_mu = self.model.compute_loo(output, self.train_x, self.train_y, self.model, likelihood=self.likelihood,
                                              multitask=isinstance(self.model, VariationalMultitaskGPModel))
        term1 = -0.5 * sigma2.log()
        term2 = -0.5 * target_minus_mu.pow(2.0) / sigma2
        res = (term1 + term2).sum(dim=-1)

        res = self._add_other_terms(res, params)
        # Scale by the amount of data we have and then add on the scaled constant
        num_data = target.size(-1)
        return res.div_(num_data) - 0.5 * math.log(2 * math.pi)

def handle_covar_( kernel: Kernel, dim: int, decomp: Union[List[List[int]], None]=None, n_funcs:int=1,
                   prior_scales:Union[Tensor,None]=None, prior_width:Union[Tensor,None]=None, outputscales:bool=True,
                   ker_kwargs:Union[dict, None]=None )-> Kernel:

    """ An utilitary to create and initialize covariance functions.

    Args:
        kernel: basis kernel type
        dim: dimension of the data (number of variables)
        decomp: instructions to create a composite kernel with subgroups of variables. Defaults to None
        Ex : decomp = [[0,1],[1,2]] --> k(x0,x1,x2) = k1(x0,x1) + k2(x1,x2)
        n_funcs: batch dimension (number of tasks or latent functions depending on the case), defaults to 1
        prior_scales: mean values of the prior for characteristic lengthscales. Defaults to None
        prior_width: deviation_to_mean ratio of the prior for characteristic lengthscales. Defaults to None
        outputscales: whether or not the full kernel has a learned scaling factor, i.e k(x) = a* k'(x). 
        If decomp is nontrivial, each subkernel is automatically granted an outputscale. Defaults to True
        ker_kwargs: additional arguments to pass to the underlying gp kernel. Defaults to None

    Returns:
        A gp-compatible kernel
    """
    if ker_kwargs is None:
        ker_kwargs = {}

    if decomp is None:
        decomp = [list(range(dim))]

    l_priors = [None] * len(decomp)
    if prior_scales is not None:
        if prior_width is None:
            raise ValueError('A prior width should be provided if a prior mean is')
        if type(prior_scales) is not list:  # 2 possible formats: an array with one length per variable, or a list with one array per kernel
            prior_scales = [prior_scales[idx_list] for idx_list in decomp]
        if type(prior_width) is not list:   # 2 possible formats: an array with one length per variable, or a list with one array per kernel
            prior_width = [prior_width[idx_list] for idx_list in decomp]

        for i_ker, idx_list in enumerate(decomp):
            if len(idx_list) > 1:
                l_priors[i_ker] = gp.priors.MultivariateNormalPrior(loc=prior_scales[i_ker],
                                            covariance_matrix=torch.diag_embed(prior_scales[i_ker]*prior_width[i_ker]))
            else:
                l_priors[i_ker] = gp.priors.NormalPrior(loc=prior_scales[i_ker],
                                                              scale=prior_scales[i_ker]*prior_width[i_ker])

    kernels_args = [{'ard_num_dims': len(idx_list), 'active_dims': idx_list, 'lengthscale_prior': l_priors[i_ker],
                         'batch_shape': torch.Size([n_funcs])} for i_ker, idx_list in enumerate(decomp)]

    kernels = []
    for i_ker, ker_args in enumerate(kernels_args):
        ker = kernel(**ker_args, **ker_kwargs)
        kernels.append(ker)

    if len(decomp) > 1 :
        covar_module = gp.kernels.ScaleKernel(kernels[0], batch_shape=torch.Size([n_funcs]))
        for ker in kernels[1:]:
            covar_module += gp.kernels.ScaleKernel(ker, batch_shape=torch.Size([n_funcs]))
    else:
        if outputscales:
            covar_module = gp.kernels.ScaleKernel(kernels[0], batch_shape=torch.Size([n_funcs]))
        else:
            covar_module = kernels[0]

    if prior_scales is not None and kernels[0].has_lengthscale:
        try:
            if len(decomp) > 1 :
                for i_ker in range(len(kernels)):
                        covar_module.kernels[i_ker].base_kernel.lengthscale = l_priors[i_ker].mean
            elif outputscales:
                covar_module.base_kernel.lengthscale = l_priors[0].mean
            else:
                covar_module.lengthscale = l_priors[0].mean
        except:
            raise ValueError('Provided prior scales were of the wrong shape')

    return covar_module

def init_lmc_coefficients( train_y: Tensor, n_latents: int, QR_form:bool=False):
    n_data, n_tasks = train_y.shape
    # if n_data >= n_tasks:
    #     SVD = TruncatedSVD(n_components=n_latents)
    #     y_transformed = SVD.fit_transform(train_y.cpu().numpy().T) / np.sqrt(n_data - 1)
    ## we use numpy-based factorizations over torch-based ones for stability reasons
    if n_data >= n_latents:
        U, S, Vt = randomized_svd(train_y.cpu().numpy().T, n_components=n_latents, random_state=0)
        U, S = torch.as_tensor(U, device=train_y.device, dtype=train_y.dtype), torch.as_tensor(S, device=train_y.device, dtype=train_y.dtype)
    else:
        Q, R = np.linalg.qr(train_y.cpu().numpy().T, mode='complete')
        S = 1e-3 * torch.ones(n_latents, device=train_y.device, dtype=train_y.dtype)
        S[:n_data] = torch.as_tensor(np.diag(R).copy(), device=train_y.device, dtype=train_y.dtype)
        U = torch.as_tensor(Q[:,:n_latents], device=train_y.device, dtype=train_y.dtype)
    if QR_form:
        return U, S
    else:
        y_transformed = U * S / np.sqrt(n_data - 1)
    return y_transformed.T

##----------------------------------------------------------------------------------------------------------------------

## Parametrizations

class ScalarParam(torch.nn.Module):
    """
    Torch parametrization for a scalar matrix.
    """
    def __init__( self, bounds: List[float] = [1e-16, 1e16]):
        super().__init__()
        self.bounds = bounds

    def forward( self, X :Tensor)-> Tensor:
        return torch.ones_like(X) * torch.clamp(X.mean(), *self.bounds)
    def right_inverse( self, A :Tensor)-> Tensor:
        return A

class PositiveDiagonalParam(torch.nn.Module):
    """
    Torch parametrization for a positive diagonal matrix.
    """
    def forward( self, X: Tensor)-> Tensor:
        return torch.diag_embed(torch.exp(torch.diag(X)))
    def right_inverse( self, A: Tensor)-> Tensor:
        return torch.diag_embed(torch.log(torch.diag(A)))

class UpperTriangularParam(torch.nn.Module):
    """
    Torch parametrization for an upper triangular matrix.
    """
    def forward( self, X: Tensor)-> Tensor:
        upper =  X.triu()
        upper[range(len(upper)), range(len(upper))] = torch.exp(upper[range(len(upper)), range(len(upper))])
        return upper
    def right_inverse( self, A: Tensor)-> Tensor: 
        res = A
        res[range(len(res)), range(len(res))] = torch.log(res[range(len(res)), range(len(res))])
        return res

class LowerTriangularParam(torch.nn.Module):
    """
    Torch parametrization for a Cholesky factor matrix (lower triangular with positive diagonal).
    """
    def __init__(self, bounds: List[float] = [1e-16, 1e16]):
        super().__init__()
        self.bounds = bounds

    def forward( self, X: Tensor )-> Tensor:
        lower = X.tril()
        # lower[range(len(lower)), range(len(lower))] = torch.exp(lower[range(len(lower)), range(len(lower))])
        lower[range(len(lower)), range(len(lower))] = torch.exp(torch.clamp(lower[range(len(lower)), range(len(lower))], *self.bounds))
        return lower
    def right_inverse( self, A: Tensor)-> Tensor:
        res = A
        res[range(len(res)), range(len(res))] = torch.log(res[range(len(res)), range(len(res))])
        return res

##----------------------------------------------------------------------------------------------------------------------

## GP models

class ExactGPModel(gp.models.ExactGP):
    """
    Standard exact GP model. Can handle independant multitasking via batch dimensions
    """
    def __init__( self, train_x:Tensor, train_y:Tensor, likelihood:Likelihood,
                  n_tasks: int = 1, prior_scales:Union[Tensor, None]=None,
                  prior_width:Union[Tensor, None]=None, mean_type:Mean=gp.means.ConstantMean,
                  decomp:Union[List[List[int]], None]=None, outputscales:bool=False,
                  kernel_type:Kernel=gp.kernels.RBFKernel,
                  ker_kwargs:Union[dict,None]=None,
                  n_inducing_points:Union[int,None]=None,
                  **kwargs ):
        """
        Args:
            train_x: training input data
            train_y: training data labels
            n_latents: number of latent processes
            n_tasks: number of output tasks
            prior_scales: Prior mean for characteristic lengthscales of the kernel. Defaults to None.
            prior_width: Prior deviation-to-mean ratio for characteristic lengthscales of the kernel. Defaults to None.
            mean_type: gp mean function for the outputs. Defaults to gp.means.ConstantMean.
            kernel_type: . gp kernel function for latent processes. Defaults to gp.kernels.RBFKernel.
            decomp: instructions to create a composite kernel with subgroups of variables. Ex : decomp = [[0,1],[1,2]] --> k(x0,x1,x2) = k1(x0,x1) + k2(x1,x2). Defaults to None.
            outputscales: whether to endow the kernel with a learned scaling factor, k(.) = a*k_base(.). Defaults to True
            ker_kwargs: Additional arguments to pass to the gp kernel function. Defaults to None.
            n_inducing_points: if an integer is provided, the model will use the sparse GP approximation of Titsias (2009).
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        if ker_kwargs is None:
            ker_kwargs = {}
        self.dim = train_x.shape[1]
        self.n_tasks = n_tasks
        self.batch_lik = isinstance(likelihood, gp.likelihoods.GaussianLikelihood)
        self.mean_module = mean_type(input_size=self.dim, batch_shape=torch.Size([n_tasks]))
        self.covar_module = handle_covar_(kernel_type, dim=self.dim, decomp=decomp, prior_scales=prior_scales,
                                          prior_width=prior_width, outputscales=outputscales,
                                          n_funcs=n_tasks, ker_kwargs=ker_kwargs)
        if n_inducing_points is not None:
            self.covar_module = gp.kernels.InducingPointKernel(self.covar_module, torch.randn(n_inducing_points, self.dim), likelihood)


    def forward( self, x:Tensor )-> Union[gp.distributions.MultivariateNormal, gp.distributions.MultitaskMultivariateNormal]:
        """
        Defines the computation performed at every call.
        Args:
            x: Data to evaluate the model at

        Returns:
            Prior distribution of the model output at the input locations. Can be a multitask multivariate normal if batch dimension is >1, or a multivariate normal otherwise
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if not self.batch_lik and self.n_tasks > 1 : # for the batch case, but not the projected model inheritance
            return gp.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gp.distributions.MultivariateNormal(mean_x, covar_x))
        else:
            return gp.distributions.MultivariateNormal(mean_x, covar_x)


    def lscales( self, unpacked:bool=True )-> Union[List[Tensor], Tensor]:  # returned format : n_kernels x n_dims
        """
        Displays the learned characteric lengthscales of the kernel(s).
        Args:
            unpacked: whether to unpack the output list and trim useless dimensions of the tensor. 
            Applies only if the model kernel is not composite. Defaults to True

        Returns:
            A list of tensors representing the learned characteristic lengthscales of each subkernel and each task (or a single tensor if the kernel is non-composite and unpacked=True).
            Each one has shape n_tasks x n_dims (n_dim number of dimensions of the subkernel)
        """
        if hasattr(self.covar_module, 'kernels'):
            n_kernels = len(self.covar_module.kernels)
            ref_kernel = self.covar_module.kernels[0]
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            scales = [reduce(getattr, attr_name.split('.'), ker).squeeze() for ker in self.covar_module.kernels]
        else:
            n_kernels = 1
            ref_kernel = self.covar_module
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            scales = reduce(getattr, attr_name.split('.'), self.covar_module).squeeze()

        return [scales] if (n_kernels==1 and not unpacked) else scales

    def outputscale( self, unpacked:bool=False)-> Tensor:
        """
        Displays the outputscale(s) of the kernel(s).
        Args:
            unpacked: whether to trim useless dimensions of the tensor. Defaults to False

        Returns:
            A tensors representing the learned outputscales of each subkernel and each task (shape n_tasks x n_kernels)
        """
        n_kernels = len(self.covar_module.kernels) if hasattr(self.covar_module, 'kernels') else 1
        n_funcs = self.n_latents if hasattr(self, 'n_latents') else self.n_tasks  ## to distinguish between the projected and batch-exact cases
        res = torch.zeros((n_funcs, n_kernels))
        if n_kernels > 1 :
            for i_ker in range(n_kernels):
                res[:, i_ker] = self.covar_module.kernels[i_ker].outputscale.data.squeeze()
        else:
            res[:,0] = self.covar_module.outputscale.data.squeeze()
        return res.squeeze() if (n_kernels==1 and unpacked) else res
    
    def kernel_cond( self ):
        K_plus = self.prediction_strategy.lik_train_train_covar.evaluate_kernel().to_dense()
        return torch.linalg.cond(K_plus)
    
    def compute_loo(self, output=None, complex_mean=False, eps=1e-6):
        train_x, train_y = self.train_inputs[0], self.train_targets
        likelihood = self.likelihood
        if self.n_tasks > 1:
            sigma2, yminusmu = torch.zeros_like(train_y), torch.zeros_like(train_y)
            # K = likelihood(output).to_data_independent_dist().lazy_covariance_matrix
            # Kbatch = output.lazy_covariance_matrix.base_linear_op.detach().to_dense() # to be removed later. A bug in gpytorch makes this necessary
            if hasattr(output.lazy_covariance_matrix, 'base_linear_op'):
                Kbatch = output.lazy_covariance_matrix.base_linear_op.evaluate() # to be removed later. A bug in gpytorch makes this necessary
            else:
                Kbatch = output.lazy_covariance_matrix.evaluate()
            global_noise = likelihood.noise.squeeze().data if hasattr(likelihood, 'noise') else 0
            m = self.mean_module(train_x).reshape(*train_y.shape)
            targets = (train_y - m).T
            # K = likelihood(output).lazy_covariance_matrix
            # identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
            # L = K.cholesky(upper=False)
            # sigma2 = 1.0 / L._cholesky_solve(identity[None,:], upper=False).diagonal(dim1=-1, dim2=-2)
            # yminusmu = L._cholesky_solve(targets.unsqueeze(-1), upper=False).squeeze(-1) * sigma2
            # sigma2, yminusmu = sigma2.detach().T, yminusmu.detach().T
            for i in range(self.n_tasks):
                K = Kbatch[i]
                noise = global_noise + likelihood.task_noises.squeeze().data[i]
                identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
                # identity_op = IdentityLinearOperator(diag_shape=K.shape[0], dtype=K.dtype, device=K.device)
                K += noise * identity
                while eps < 1.: # this is a hack to avoid numerical instability
                    try:
                        L = K.cholesky(upper=False)
                        break
                    except:
                        eps *= 10
                        K += eps * identity
                        warnings.warn('Cholesky decomposition failed. Increasing jitter to {}'.format(eps))
                L = torch.linalg.cholesky(K, upper=False)
                sigma2[:,i] = 1.0 / torch.cholesky_solve(identity[None,:], L, upper=False).diagonal(dim1=-1, dim2=-2)
                yminusmu[:,i] = torch.cholesky_solve(targets[i].unsqueeze(-1), L, upper=False).squeeze(-1) * sigma2[:,i]
            sigma2, yminusmu = sigma2.detach(), yminusmu.detach()

        else: # single-output case
            m, K, noise_it = self.mean_module(train_x), output.lazy_covariance_matrix, self.likelihood.noise.data
            m = m.reshape(*train_y.shape)
            identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
            noise_val = max(noise_it, eps)
            K += noise_val * identity
            with gp.settings.cholesky_max_tries(3):
                if complex_mean:
                    if not hasattr(self.mean_module, 'basis_matrix'):
                        raise ValueError('A complex mean treatment was required, but the model mean function doesn\'t allow it !')
                    else: # This has not been thoroughly tested yet
                        K_factors = K.cholesky(upper=False)
                        K_inv = K_factors._cholesky_solve(identity, upper=False).squeeze()
                        H = self.mean_module.basis_matrix(train_x)
                        M = torch.mm(torch.mm(H.T, K_inv), H)
                        M_factors = torch.linalg.cholesky(M + eps, upper=False)  # Now this is a Torch method, not a gp one (M is not a lazy matrix anymore)
                        identity_bis = torch.eye(*M.shape[-2:], dtype=M.dtype, device=M.device)
                        M_inv = torch.cholesky_solve(identity_bis, M_factors, upper=False)
                        K_minus = K_inv - K_inv @ H @ M_inv @ H.T @ K_inv
                        sigma2 = 1.0 / K_minus.diagonal(dim1=-1, dim2=-2)
                        yminusmu = K_minus @ train_y * sigma2
                else:
                    L = K.cholesky(upper=False)
                    sigma2 = 1.0 / L._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2)
                    yminusmu = L._cholesky_solve((train_y - m).unsqueeze(-1), upper=False).squeeze(-1) * sigma2

        return sigma2, yminusmu

class MultitaskGPModel(ExactGPModel):
    """
    A multitask GP model with exact GP treatment. This class encompasses both ICM and naive LMC models.
    """
    def __init__( self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood,
                  n_tasks: int, n_latents: int, model_type:str='ICM', init_lmc_coeffs:bool=True,
                  fix_diagonal:bool=False, **kwargs):
        """Initialization of the model. Note that the optional arguments of the ExactGPModel also apply here thanks to the inheritance.

        Args:
            train_x: Input data
            train_y: Input labels
            likelihood: Likelihood of the model
            n_tasks: number of tasks
            n_latents: number of latent functions
            model_type: choice between 'ICM' and 'LMC' (see the reference paper). Defaults to "ICM"
            init_lmc_coeffs: if True, initializes LMC coefficients with SVD of the training labels ; else inializes with samples from standard normal distributions. Defaults to True
            fix_diagonal: for ICM only. If True, fixes the learned diagonal term of the task covariance matrix, accounting for task-specific (non-shared) latent processes. Defaults to False
        """

        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood, n_tasks=1, outputscales=False, **kwargs) # we build upon a single-task GP, created by calling parent class

        self.mean_module = gp.means.MultitaskMean(self.mean_module, num_tasks=n_tasks)
        
        if model_type=='ICM':
            self.covar_module = gp.kernels.MultitaskKernel(self.covar_module, num_tasks=n_tasks, rank=n_latents)
        elif model_type=='LMC':
            self.covar_module = gp.kernels.LCMKernel(base_kernels=[copy.deepcopy(self.covar_module) for i in range(n_latents)],
                                                           num_tasks=n_tasks, rank=1)

        if init_lmc_coeffs:
            lmc_coeffs = init_lmc_coefficients(train_y, n_latents)
            if model_type=='ICM':
                # this parameter has already been initialized with random values at the instantiation of the variational strategy, so registering it anew is facultative
                self.covar_module.task_covar_module.register_parameter(name='covar_factor', parameter=torch.nn.Parameter(lmc_coeffs))
            elif model_type=='LMC':
                for i in range(n_latents):
                    # this parameter has already been initialized with random values at the instantiation of the variational strategy, so registering it anew is facultative
                    self.covar_module.covar_module_list[i].task_covar_module.covar_factor = torch.nn.Parameter(lmc_coeffs[:,i].unsqueeze(-1))
            else:
                raise ValueError('Wrong specified model type, should be ICM or LMC')

        if fix_diagonal:
            if model_type=='ICM':
                self.covar_module.task_covar_module.register_parameter(name='raw_var',
                                            parameter=torch.nn.Parameter(-10*torch.ones(n_tasks, device=train_y.device),
                                            requires_grad=False))
            elif model_type=='LMC':
                for i in range(len(self.covar_module.covar_module_list)):
                    self.covar_module.covar_module_list[i].task_covar_module.register_parameter(name='raw_var',
                                                parameter=torch.nn.Parameter(-10*torch.ones(n_tasks, device=train_y.device),
                                                requires_grad=False))
                
        self.n_tasks, self.n_latents, self.model_type = n_tasks, n_latents, model_type

    def lmc_coefficients( self )-> Tensor:
        """

        Returns:
            tensor of shape n_latents x n_tasks representing the LMC/ICM coefficients of the model.
        """
        if self.model_type=='LMC':
            res = torch.zeros((self.n_latents, self.n_tasks))
            for i in range(self.n_latents):
                res[i] = self.covar_module.covar_module_list[i].task_covar_module.covar_factor.data.squeeze()
        else:
            res = self.covar_module.task_covar_module.covar_factor.data.squeeze().T
        return res

    def lscales( self, unpacked:bool=True)-> Union[List[Tensor], Tensor] :
        """
        Displays the learned characteric lengthscales of the kernel(s).
        Args:
            unpacked: whether to unpack the output list and trim useless dimensions of the tensor. 
            Applies only if the model kernel is not composite. Defaults to True

        Returns:
            A list of tensors representing the learned characteristic lengthscales of each subkernel and each task (or a single tensor if the kernel is non-composite and unpacked=True).
            Each one has shape n_latents x n_dims (n_dim number of dimensions of the subkernel)
        """
        if self.model_type=='LMC':
            data_covar = self.covar_module.covar_module_list[0].data_covar_module
        else:
            data_covar = self.covar_module.data_covar_module

        if hasattr(data_covar, 'kernels'):
            n_kernels = len(data_covar.kernels)
            ref_kernel = data_covar.kernels[0]
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            if self.model_type=='ICM':
                scales = [reduce(getattr, attr_name.split('.'), ker).squeeze().repeat(self.n_latents, 1) for ker in data_covar.kernels]
            else:
                ref_scales = [reduce(getattr, attr_name.split('.'), ker).squeeze() for ker in data_covar.kernels]
                ker_dims = [len(scales) if scales.ndim > 0 else 1 for scales in ref_scales]
                scales = [torch.zeros((self.n_latents, ker_dims[i])) for i in range(n_kernels)]
        else:
            n_kernels = 1
            ref_kernel = data_covar
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            if self.model_type=='ICM':
                scales = reduce(getattr, attr_name.split('.'), data_covar).squeeze().repeat(self.n_latents, 1)
            else:
                ref_scales = reduce(getattr, attr_name.split('.'), data_covar).squeeze()
                ker_dim = len(ref_scales) if ref_scales.ndim > 0 else 1
                scales = torch.zeros((self.n_latents, ker_dim))

        if self.model_type=='LMC':
            for i in range(self.n_latents):
                if n_kernels > 1:
                    for j in range(n_kernels):
                        scales[j][i,:] = reduce(getattr, attr_name.split('.'), self.covar_module.covar_module_list[i].data_covar_module.kernels[j]).squeeze()
                else:
                    scales[i,:] = reduce(getattr, attr_name.split('.'), self.covar_module.covar_module_list[i].data_covar_module).squeeze()
      
        return [scales] if (n_kernels==1 and not unpacked) else scales    
        
    def outputscale( self, unpacked:bool=False)-> Tensor:
        """
        Displays the outputscale(s) of the kernel(s).
        Args:
            unpacked: whether to trim useless dimensions of the tensor. Defaults to False

        Returns:
            A tensor representing the learned outputscales of each subkernel and each task (shape n_latents x n_kernels)
        """
        if self.model_type=='LMC':
            data_covar = self.covar_module.covar_module_list[0].data_covar_module
        else:
            data_covar = self.covar_module.data_covar_module

        n_kernels = len(data_covar.kernels) if hasattr(data_covar, 'kernels') else 1
        res = torch.zeros((self.n_latents, n_kernels))
        if n_kernels > 1:
            for i_ker in range(n_kernels):
                if self.model_type=='LMC':
                    for i_lat in range(self.n_latents):
                        res[i_lat, i_ker] = self.covar_module.covar_module_list[i_lat].data_covar_module.kernels[i_ker].outputscale.data.squeeze()
                else:
                    res[:, i_ker] = data_covar.kernels[i_ker].outputscale.data.squeeze()
        else:
            if self.model_type == 'LMC':
                for i_lat in range(self.n_latents):
                    res[i_lat, 0] = self.covar_module.covar_module_list[i_lat].data_covar_module.outputscale.data.squeeze()
            else:
                res[:, 0] = data_covar.outputscale.data.squeeze()

        return res.squeeze() if (n_kernels==1 and unpacked) else res

    def forward( self, x ):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def compute_var(self, x):
        """
        Computes the variance of the model at input locations.
        Args:
            x: input data to evaluate the model at

        Returns:
            A tensor of the variances of the model at input locations.
        """
        if self.model_type!='ICM':
            raise ValueError('This method is only available for ICM models')
        
        linop = self.covar_module.forward(x,x) + self.likelihood._shaped_noise_covar((len(x), len(x)), add_noise=True)
        first_term = linop.diagonal(dim1=-2, dim2=-1).reshape((len(x), self.n_tasks))

        x_train = self.train_inputs[0]
        ker_op = self.covar_module.forward(x_train,x_train)
        noise_op = self.likelihood._shaped_noise_covar((len(x_train), len(x_train)), add_noise=True)
        i_task = 1 if isinstance(ker_op.linear_ops[1], PsdSumLinearOperator) else 0

        data_ker = ker_op.linear_ops[(i_task + 1) % 2]
        k_evals, k_evecs = data_ker._symeig(eigenvectors=True)

        noise_inv_root = noise_op.linear_ops[i_task].root_inv_decomposition()
        C = ker_op.linear_ops[i_task]
        C_tilde = noise_inv_root.matmul(C.matmul(noise_inv_root))
        C_evals, C_evecs = C_tilde._symeig(eigenvectors=True)
        C_hat = C.matmul(noise_inv_root).matmul(C_evecs).evaluate().squeeze()
        C_square = C_hat**2

        S = torch.kron(k_evals, C_evals) + 1.0
        if x.is_cuda:
            free_mem = torch.cuda.mem_get_info()[0]
        else:
            free_mem = psutil.virtual_memory()[1]

        num_bytes = x.element_size()
        batch_size = int(free_mem / (16 * len(x_train) * self.n_tasks**2 * num_bytes))
        n = x.shape[0]  # Total number of samples
        second_term_results = []  # List to store the results
        for i in range(0, n, batch_size):
            x_batch = x[i:i+batch_size]
            k_hat = self.covar_module.data_covar_module(x_batch, x_train).matmul(k_evecs).evaluate().squeeze()
            k_square = k_hat**2
            second_term = torch.kron(k_square, C_square) @ S.pow(-1).squeeze()
            second_term_results.append(second_term.reshape((len(x_batch), self.n_tasks)))

        # Convert the list of results to a tensor
        second_term = torch.cat(second_term_results)
        return torch.clamp(first_term - second_term, min=1e-6)

    def compute_loo(self, output=None):
        train_x, train_y = self.train_inputs[0], self.train_targets        
        if output is None:
            output = self.forward(train_x)
        m, K = self.mean_module(train_x), self.likelihood(output).lazy_covariance_matrix
        m = m.reshape(*train_y.shape)
        targets = torch.reshape(train_y - m, (np.prod(train_y.shape).astype(int),1))
        with gp.settings.cholesky_max_tries(6):
            L = K.cholesky(upper=False)
            identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
            sigma2 = 1.0 / L._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2)
            yminusmu = L._cholesky_solve(targets, upper=False).squeeze(-1) * sigma2
        sigma2 = torch.reshape(sigma2.detach(), train_y.shape)
        yminusmu = torch.reshape(yminusmu.detach(), train_y.shape)
        return sigma2, yminusmu


class CustomLMCVariationalStrategy(gp.variational.LMCVariationalStrategy):  # allows to put means on tasks instead of latent processes
    """
    This small overlay to the native LMCVariationalStrategy of gp allows to put deterministic means on tasks rather than latent processes.
    """
    def __init__(self, mean_module:gp.means.mean, *args, **kwargs):
        """

        Args:
            mean_module: The already-generated, batched, many-tasks mean function to impose on output tasks.
        """
        super().__init__(*args, **kwargs)
        self.output_mean_module = mean_module

    def __call__(self, x:Tensor, task_indices=None, prior=False, **kwargs)-> gp.distributions.MultitaskMultivariateNormal:
        """

        Args:
            x:Input data to evaluate model at.

        Returns:
            The posterior distribution of the model at input locations.
        """
        multitask_dist = super().__call__(x, task_indices=None, prior=False, **kwargs)
        tasks_means = self.output_mean_module(x)
        return multitask_dist.__class__(multitask_dist.mean + tasks_means.T, multitask_dist.lazy_covariance_matrix)


class VariationalMultitaskGPModel(gp.models.ApproximateGP):
    """
    A standard variational LMC model using gp functionalities.
    """
    def __init__( self, train_x:Tensor, n_latents:int, n_tasks:int, train_ind_ratio:float=1.5, seed:int=0,
                  init_lmc_coeffs:bool=False, train_y:Union[Tensor,None]=None, prior_scales:Tensor=None, prior_width:Tensor=None,
                  mean_type:Mean=gp.means.ConstantMean, kernel_type:Kernel=gp.kernels.RBFKernel,
                  outputscales:bool=False, 
                  decomp:Union[List[List[int]],None]=None,
                  distrib:gp.variational._VariationalDistribution=gp.variational.CholeskyVariationalDistribution, 
                  var_strat:gp.variational._VariationalStrategy=gp.variational.VariationalStrategy,
                  ker_kwargs:Union[dict,None]=None, **kwargs):
        """
        Args:
            train_x: training input data
            n_latents: number of latent processes
            n_tasks: number of output tasks
            train_ind_ratio: ratio between the number of training points and this of inducing points. Defaults to 1.5.
            seed: Random seed for inducing points generation. Defaults to 0.
            init_lmc_coeffs: Whether to initialize LMC coefficients with the SVD of the training labels. Defaults to False.
            train_y: training data labels, used only for the SVD initialization of LMC coefficients. Defaults to None.
            prior_scales: Prior mean for characteristic lengthscales of the kernel. Defaults to None.
            prior_width: Prior deviation-to-mean ratio for characteristic lengthscales of the kernel. Defaults to None.
            mean_type: gp mean function for the outputs. Defaults to gp.means.ConstantMean.
            kernel_type: . gp kernel function for latent processes. Defaults to gp.kernels.RBFKernel.
            decomp: instructions to create a composite kernel with subgroups of variables. Ex : decomp = [[0,1],[1,2]] --> k(x0,x1,x2) = k1(x0,x1) + k2(x1,x2). Defaults to None.
            distrib: gp variational distribution for inducing values (see gp documentation). Defaults to gp.variational.CholeskyVariationalDistribution.
            var_strat: gp variational strategy (see gp documentation). Defaults to gp.variational.VariationalStrategy.
            ker_kwargs: Additional arguments to pass to the gp kernel function. Defaults to None.
        """

        if ker_kwargs is None:
            ker_kwargs = {}
        self.dim = train_x.shape[1]
        if train_y is not None and train_y.shape[1]!=n_tasks:
            n_tasks = train_y.shape[1]
            warnings.warn('Number of tasks in the training labels does not match the specified number of tasks. Defaulting to the number of tasks in the training labels.')

        if float(train_ind_ratio) == 1.:
            warnings.warn('Caution : inducing points not learned !')
            inducing_points = train_x
            learn_inducing_locations = False
            var_strat = gp.variational.UnwhitenedVariationalStrategy  #better compatibility in this case
            distrib = gp.variational.CholeskyVariationalDistribution  #better compatibility in this case
        else:
            learn_inducing_locations = True
            n_ind_points = int(np.floor(train_x.shape[0] / train_ind_ratio))
            sampler = qmc.LatinHypercube(d=self.dim, seed=seed)
            inducing_points = torch.as_tensor(2 * sampler.random(n=n_ind_points) - 1, dtype=train_x.dtype)
            #same inducing points for all latents here

        variational_distribution = distrib(inducing_points.size(-2), batch_shape=torch.Size([n_latents]))
        strategy = var_strat(self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_locations)
        output_mean_module = mean_type(input_size=self.dim, batch_shape=torch.Size([n_tasks]))

        variational_strategy = CustomLMCVariationalStrategy(
            output_mean_module,
            strategy,
            num_tasks=n_tasks,
            num_latents=n_latents,
            latent_dim=-1)

        super().__init__(variational_strategy)

        self.covar_module = handle_covar_(kernel_type, dim=self.dim, decomp=decomp, prior_scales=prior_scales,
                                            prior_width=prior_width, n_funcs=n_latents, ker_kwargs=ker_kwargs, outputscales=outputscales)
        self.mean_module = gp.means.ZeroMean(batch_shape=torch.Size([n_latents])) #in gp, latent processes can have non-zero means, which we wish to avoid

        self.n_tasks, self.n_latents, self.decomp = n_tasks, n_latents, decomp

        if init_lmc_coeffs and train_y is not None:
            lmc_coefficients = init_lmc_coefficients(train_y, n_latents=n_latents)
            if train_y.device.type=='cuda':
                lmc_coefficients = lmc_coefficients.cuda()
            # this parameter has already been initialized with random values at the instantiation of the variational strategy, so registering it anew is facultative
            self.variational_strategy.register_parameter("lmc_coefficients", torch.nn.Parameter(lmc_coefficients))  #shape n_latents x n_tasks

    def forward( self, x:Tensor )-> Tensor:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

    def lscales( self, unpacked:bool=True )-> Union[List[Tensor], Tensor]:  # returned shape : n_kernels x n_dims
        """
        Displays the learned characteric lengthscales of the kernel(s).
        Args:
            unpacked: whether to unpack the output list and trim useless dimensions of the tensor. Applies only if the model kernel is not composite

        Returns:
            A list of tensors representing the learned characteristic lengthscales of each subkernel and each task (or a single tensor if the kernel is non-composite and unpacked=True).
            Each one has shape n_latents x n_dims (n_dim number of dimensions of the subkernel)
        """    
        if hasattr(self.covar_module, 'kernels'):
            n_kernels = len(self.covar_module.kernels)
            ref_kernel = self.covar_module.kernels[0]
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            scales = [reduce(getattr, attr_name.split('.'), ker) for ker in self.covar_module.kernels]
        else:
            n_kernels = 1
            ref_kernel = self.covar_module
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            scales = reduce(getattr, attr_name.split('.'), self.covar_module)

        return [scales] if (n_kernels==1 and not unpacked) else scales
    
    def outputscale( self, unpacked:bool=False)-> Tensor:
        """
        Displays the outputscale(s) of the kernel(s).
        Args:
            unpacked: whether to trim useless dimensions of the tensor

        Returns:
            A tensors representing the learned outputscales of each subkernel and each task (shape n_latents x n_kernels)
        """
        n_kernels = len(self.covar_module.kernels) if hasattr(self.covar_module, 'kernels') else 1
        res = torch.zeros((self.n_latents, n_kernels))
        if n_kernels > 1:
            for i_ker in range(n_kernels):
                res[:, i_ker] = self.covar_module.kernels[i_ker].outputscale.data.squeeze()
        else:
            res[:, 0] = self.covar_module.outputscale.data.squeeze()
        return res.squeeze() if (n_kernels==1 and unpacked) else res
    
    def lmc_coefficients( self ):
        return self.variational_strategy.lmc_coefficients.data
    
    def compute_latent_distrib( self, x, prior=False, **kwargs):
        return self.base_variational_strategy(x, prior=prior, **kwargs)



## making the mixing matrix a separate class allows to call torch.nn.utils.parametrizations.orthogonal
## onto it during instanciation of a ProjectedGPModel
class LMCMixingMatrix(torch.nn.Module):
    """
    Class for the parametrized mixing matrix of projected models. Making it a separate class allows to call 
    torch.nn.utils.parametrizations.orthogonal onto it during instanciation of a ProjectedGPModel
    """
    def __init__( self, Q_plus:Tensor, R:Tensor, bulk:bool=True ):
        """

        Args:
            Q_plus: (augmented) orthonormal part of the mixing matrix. See the reference article for explanations
            R: upper triangular part of the mixing matrix. See the reference article for explanations
        """
        super().__init__()
        if Q_plus.shape[1]==Q_plus.shape[0]:
            self.mode = 'Q_plus'
        elif Q_plus.shape[1]==R.shape[0]:
            self.mode = 'Q'
        else:
            raise ValueError('Wrong dimensions for Q_plus : should be n_tasks x n_tasks or n_tasks x n_latents')
        
        self.n_latents = R.shape[0]
        self.n_tasks = Q_plus.shape[0]
        self._size = torch.Size([self.n_latents, self.n_tasks])
        self.bulk = bulk
        if bulk:
            if self.mode=='Q_plus':
                R_padded = torch.eye(self.n_tasks)
                R_padded[:self.n_latents, :self.n_latents] = R
                H = Q_plus @ R_padded
            else:
                H = Q_plus @ R
            self.register_parameter("H", torch.nn.Parameter(H, requires_grad=True))
        else:
            self.register_parameter("Q_plus", torch.nn.Parameter(Q_plus, requires_grad=True))
            self.register_parameter("R", torch.nn.Parameter(R, requires_grad=True))

    def Q( self ):
        if self.mode=='Q_plus':
            return self.Q_plus[:, :self.n_latents]
        else:
            return self.Q_plus

    def Q_orth( self ):
        return self.Q_plus[:, self.n_latents:]

    def QR(self):
        if self.bulk:
            Q_plus, R_padded = torch.linalg.qr(self.H)
            if self.mode=='Q_plus':
                Q = Q_plus[:, :self.n_latents]
                Q_orth = Q_plus[:, self.n_latents:]
                R = R_padded[:self.n_latents, :self.n_latents]
            else:
                Q, Q_orth, R = Q_plus, None, R_padded
        else:
            Q, Q_orth, R = self.Q(), self.Q_orth(), self.R
        return Q, R, Q_orth

    def forward( self ):
        if self.bulk:
            if self.mode == 'Q':
                return self.H.T
            else:
                return self.H[:,:self.n_latents].T
        else:
            return (self.Q() @ self.R).T #format : n_latents x n_tasks

    def size( self, int=None ):
        if int:
            return self._size[int]
        else:
            return self._size


class ProjectedGPModel(ExactGPModel):
    """
    The projected LMC from the reference article.
    """
    def __init__( self, train_x:Tensor, train_y:Tensor, n_tasks:int, n_latents:int, proj_likelihood:Union[None,Likelihood]=None, 
                  init_lmc_coeffs:bool=False, BDN:bool=True, diagonal_B:bool=False, scalar_B:bool=False, diagonal_R:bool=False,
                  mean_type:Mean=gp.means.ConstantMean, ortho_param='matrix_exp', bulk=True,
                  noise_thresh:float=-9., noise_init:float=1e-2, outputscales:bool=False, eps=1e-3, **kwargs):
        """Initialization of the model. Note that the optional arguments of the ExactGPModel also apply here thanks to the inheritance.
        
        Args:
            train_x: training input data
            train_y: training input labels
            n_tasks: number of output tasks
            n_latents: number of latent processes
            proj_likelihood: batched independant likelihood of size n_latents for latent processes. Defaults to None.
            init_lmc_coeffs: whether to initialize LMC coefficients with SVD of the training labels. Defaults to False.
            BDN: whether to enforce the Block Diagonal Noise approximation (see reference article), making for a block-diagonal task noise matrix. Defaults to True.
            diagonal_B: whether to parametrize a diagonal noise factor B_tilde (see reference article), a simplification which theoretically causes no loss of generality. Defaults to False.
            scalar_B: whether to parametrize a scalar noise factor B_tilde (see reference article). Defaults to False.
            diagonal_R: whether to parametrize a diagonal scale component for the mixing matrix (see reference article). Defaults to False.
            mean_type: gp mean function for task-level processes. Defaults to gp.means.ConstantMean.
        """
        if proj_likelihood is None or proj_likelihood.noise.shape[0] != n_latents:
            warnings.warn("In projected GP model the dimension of the likelihood is the number of latent processes. "
                  "Provided likelihood was the wrong shape or None, so it was replaced by a fresh one")

            proj_likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_latents]),
                                                    noise_constraint=gp.constraints.GreaterThan(np.exp(noise_thresh)))

        super().__init__(train_x, torch.zeros_like(train_y), proj_likelihood, n_tasks=n_latents, 
                         mean_type=gp.means.ZeroMean, outputscales=outputscales, **kwargs) # !! proj_likelihood will only be named likelihood in the model
        self.register_buffer('train_y', train_y)

        if mean_type is not gp.means.ZeroMean:
            raise ValueError('Projected GP model does not support non-zero output-wise means for now !')

        n_data, n_tasks = train_y.shape
        if init_lmc_coeffs:
            if scalar_B and BDN:
                Q_plus, R = init_lmc_coefficients(train_y, n_latents=n_latents, QR_form=True) # Q_plus has shape n_tasks x n_latents, R_padded has shape n_latents x n_latents        
            else:
                Q_plus, R_padded = init_lmc_coefficients(train_y, n_latents=n_tasks, QR_form=True) # Q_plus has shape n_tasks x n_tasks, R_padded has shape n_tasks x n_latents
                R = R_padded[:n_latents]
            # n_data, n_tasks = train_y.shape
            # if n_data > n_tasks:
            #     U, Sigma, VT = randomized_svd(train_y.cpu().numpy(), n_components=n_tasks, random_state=0)
            #     Q_plus, R = torch.as_tensor(VT.T, dtype=train_y.dtype), torch.as_tensor(np.diag(Sigma[:n_latents]) / np.sqrt(n_data - 1), dtype=train_y.dtype)
            # else:
            #     try:
            #         Q_plus, R_padded, Vt = torch.linalg.svd(train_y.T, full_matrices=True) 
            #         # we perform SVD rather than QR for the case where R must be diagonal (OILMM)
            #     except: # sklearn's randomized_svd is more stable than torch's svd in some cases
            #         U, Sigma, Vt = randomized_svd(train_y.cpu().numpy().T, n_components=n_data, random_state=0)
            #         Q_plus, R_padded = torch.as_tensor(U, dtype=train_y.dtype), torch.as_tensor(Sigma, dtype=train_y.dtype)
            #     if n_latents < n_data:
            #         R_padded = R_padded[:n_latents]
            #     else:
            #         R_padded, R_short = 1e-3*torch.ones(n_latents, dtype=train_y.dtype), R_padded
            #         R_padded[:n_data] = R_short
            #     R = torch.diag_embed(R_padded) / np.sqrt(n_data - 1)
        else:
            fake_coeffs = torch.randn(n_tasks, n_latents)
            Q_plus, R_padded, Vt = torch.linalg.svd(fake_coeffs) # Q_plus has shape n_tasks x n_tasks, R_padded has shape n_tasks x n_latents
            # we perform SVD rather than QR for the case where R must be diagonal (OILMM)
            R = R_padded[:n_latents]
            if scalar_B and BDN: # case of the PLMC_fast
                Q_plus = Q_plus[:,:n_latents]

        R = torch.diag_embed(R) / np.sqrt(n_data - 1)
        lmc_coefficients = LMCMixingMatrix(Q_plus, R, bulk=bulk)
        if not bulk:
            lmc_coefficients = torch.nn.utils.parametrizations.orthogonal(lmc_coefficients, name="Q_plus", orthogonal_map=ortho_param,
                                                                        use_trivialization=(ortho_param!='householder'))  # parametrizes Q_plus as orthogonal
            if diagonal_R:
                torch.nn.utils.parametrize.register_parametrization(lmc_coefficients, "R", PositiveDiagonalParam())
            else:
                torch.nn.utils.parametrize.register_parametrization(lmc_coefficients, "R", UpperTriangularParam())
        self.lmc_coefficients = lmc_coefficients

        if scalar_B:
            diagonal_B = True
            self.register_parameter("log_B_tilde", torch.nn.Parameter(np.log(noise_init) * torch.ones(n_tasks - n_latents)))
            torch.nn.utils.parametrize.register_parametrization(self, "log_B_tilde", ScalarParam(bounds=(noise_thresh, -noise_thresh)))
            if BDN:
                self.register_buffer('Y_squared_norm', (train_y**2).sum()) # case of the PLMC_fast (term for MLL computation)
        elif diagonal_B:
            self.register_parameter("log_B_tilde", torch.nn.Parameter(np.log(noise_init)*torch.ones(n_tasks - n_latents)))
            self.register_constraint("log_B_tilde", gp.constraints.GreaterThan(noise_thresh))
        else:
            self.register_parameter("B_tilde_inv_chol", torch.nn.Parameter(torch.diag_embed(np.log(1/noise_init)*torch.ones(n_tasks - n_latents))))
            torch.nn.utils.parametrize.register_parametrization(self, "B_tilde_inv_chol", LowerTriangularParam(bounds=(noise_thresh, -noise_thresh)))
        self.diagonal_B, self.scalar_B = diagonal_B, scalar_B

        if not BDN:
            self.register_parameter("M", torch.nn.Parameter(torch.zeros((n_latents, n_tasks - n_latents))))

        self.n_tasks = n_tasks
        self.n_latents = n_latents
        self.latent_dim = -1
        self.eps = eps


    def projected_noise( self )-> Tensor:
        """
        Returns a vector of size n_latents containing the modeled noises of latent processes. Its diagonal embedding is the matrix Sigma_P from the article. 
        """
        return self.likelihood.noise.squeeze(-1)
    
    # @lru_cache(maxsize=None) # caching projected data and projected matrix is appealing, but it messes with backpropagation. No workaround has been found yet
    def projection_matrix( self )-> Tensor:
        """
        Returns matrix T from the article of shape n_tasks x n_latents, such that YT is the "projected data" seen by latent processes 
        """
        Q, R, Q_orth = self.lmc_coefficients.QR()
        H_pinv = torch.linalg.solve_triangular(R.T, Q, upper=False, left=False)  # shape n_tasks x n_latents
        if hasattr(self, "M"):
            return H_pinv + Q_orth @ self.M.T * self.projected_noise()[None,:]
        else:
            return H_pinv

    def project_data( self, data ):
        Q, R, Q_orth = self.lmc_coefficients.QR()
        unscaled_proj = Q.T @ data.T
        Hpinv_times_Y = torch.linalg.solve_triangular(R, unscaled_proj, upper=True)  # shape n_latents x n_points ; opposite convention to most other quantities !!
        if hasattr(self, "M"):
            return Hpinv_times_Y + self.projected_noise()[:,None] * self.M @ Q_orth.T @ data.T
        else:
            return Hpinv_times_Y

    def full_likelihood( self ):
        Q, R, Q_orth = self.lmc_coefficients.QR()
        res = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_tasks, rank=self.n_tasks, has_global_noise=False)
        QR = Q @ R
        sigma_p = self.projected_noise()
        if sigma_p.is_cuda:
            res.cuda()
        if hasattr(self, "M"):
            if self.diagonal_B:
                B_tilde_root = torch.diag_embed(torch.exp(self.log_B_tilde / 2))
            else:
                B_tilde_root = torch.linalg.solve_triangular(self.B_tilde_inv_chol, 
                                            torch.eye(self.n_tasks - self.n_latents, device=self.B_tilde_inv_chol.device), upper=False).T
            B_tilde = B_tilde_root @ B_tilde_root.T
            B_term = Q_orth @ B_tilde @ Q_orth.T
            M_term = - QR @ (sigma_p[:,None] * self.M) @ B_tilde @ Q_orth.T
            Mt_term = M_term.T
            D_term_rotated = torch.diag_embed(sigma_p) + sigma_p[:,None] * self.M @ B_tilde @ self.M.T * sigma_p[None,:]
            D_term = QR @ D_term_rotated @ QR.T
        else:
            if self.scalar_B:
                if self.log_B_tilde.numel() > 0:
                    B_term = torch.exp(self.log_B_tilde[0]) * (torch.eye(self.n_tasks, device=self.log_B_tilde.device) - Q @ Q.T)
                else:
                    B_term = 0.
            else:
                if self.diagonal_B:
                    B_tilde_root = torch.diag_embed(torch.exp(self.log_B_tilde / 2))
                else:
                    B_tilde_root = torch.linalg.solve_triangular(self.B_tilde_inv_chol,
                        torch.eye(self.n_tasks - self.n_latents, device=self.B_tilde_inv_chol.device), upper=False).T
                B_term_root = Q_orth @ B_tilde_root
                B_term = B_term_root @ B_term_root.T
            M_term, Mt_term = 0., 0.
            D_term_root = QR * torch.sqrt(sigma_p)[None,:]
            D_term = D_term_root @ D_term_root.T

        Sigma = D_term + M_term + Mt_term + B_term
        # We use a while loop to ensure that the full noise covariance is positive definite.
        # We can deactivate gradient computation as loss computation does not involve the full likelihood
        with torch.no_grad(): 
            eps = 1e-6
            while eps < self.eps:
                try:
                    identity = torch.eye(self.n_tasks, dtype=res.task_noise_covar.dtype, device=res.task_noise_covar.device)
                    res.task_noise_covar_factor.data = torch.linalg.cholesky(Sigma + eps*identity)
                    break
                except:
                    eps *= 10
                    warnings.warn("Cholesky of the full noise covariance failed. Trying again with jitter {0} ...".format(eps))

        return res

    def B_tilde( self )-> Tensor:
        """
        Outputs the discarded noise factor B_tilde from the reference paper. 
        Returns:
            Discarded noise factor B_tilde (see reference paper), symmetric or even diagonal matrix of size (n_tasks - n_latents).
        """        
        if self.diagonal_B:
            return torch.diag_embed(torch.exp(self.log_B_tilde))
        else:
            L_inv = torch.linalg.solve_triangular(self.B_tilde_inv_chol, torch.eye(self.n_tasks - self.n_latents), upper=False)
            return L_inv.T @ L_inv

    def forward( self, x:Tensor )-> gp.distributions.MultivariateNormal:  # ! forward only returns values of the latent processes !
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

    def compute_latent_distrib( self, x:Tensor, **kwargs )-> gp.distributions.MultivariateNormal:
        """
        Outputs (distributional) posterior values of the latent processes at the input locations. This is the function which is called to compute
        the loss during training.
        Args:
            x: input data tensor

        Returns:
            A batched gp multivariate normal distribution representing latent processes values, which mean has shape n_latents x n_points.
        """
        proj_targets = self.project_data(self.train_y)
        super().set_train_data(inputs=self.train_inputs, targets=proj_targets, strict=False)
        batch_distrib = ExactGPModel.__call__(self, x, **kwargs)
        return batch_distrib  # shape n_latents x n_points
    
    def compute_loo(self, output=None):
        train_x, train_y = self.train_inputs[0], self.train_y
        if output is None:
            output = self.compute_latent_distrib(train_x)
        K = self.likelihood(output).lazy_covariance_matrix
        y_proj = self.project_data(train_y).T
        identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
        L = K.cholesky(upper=False)
        sigma2 = 1.0 / L._cholesky_solve(identity[None,:], upper=False).diagonal(dim1=-1, dim2=-2)
        yminusmu = L._cholesky_solve(y_proj.T.unsqueeze(-1), upper=False).squeeze(-1) * sigma2
        sigma2, yminusmu = sigma2.detach().T, yminusmu.detach().T
        return sigma2, yminusmu

    def __call__(self, x:Tensor, **kwargs)-> gp.distributions.MultitaskMultivariateNormal:
        """
        Outputs the full posterior distribution of the model at input locations. This is used to make predictions.
        Args:
            x: input data tensor

        Returns:
            A multitask multivariate gp normal distribution representing task processes values, which mean has shape n_points x n_tasks.
        """
        if self.training: # in training mode, we just compute the prior distribution of latent processes
            return super().__call__(x, **kwargs)
        
        super().set_train_data(inputs=self.train_inputs, targets=self.project_data(self.train_y), strict=False)
        latent_dist = ExactGPModel.__call__(self, x, **kwargs)

        num_batch = len(latent_dist.batch_shape)
        latent_dim = num_batch + self.latent_dim

        num_dim = num_batch + len(latent_dist.event_shape)
        lmc_coefficients = self.lmc_coefficients().expand(*latent_dist.batch_shape, self.lmc_coefficients.size(-1))

        # Mean: ... x N x n_tasks
        latent_mean = latent_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
        mean = latent_mean @ lmc_coefficients.permute(
            *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
        )

        # Covar: ... x (N x n_tasks) x (N x n_tasks)
        latent_covar = latent_dist.lazy_covariance_matrix
        lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
        latent_covar = to_linear_operator(latent_covar.evaluate())
        covar = KroneckerProductLinearOperator(latent_covar, lmc_factor).sum(latent_dim)
        covar = covar.add_jitter(self.eps)

        return gp.distributions.MultitaskMultivariateNormal(mean, covar)
    

class ProjectedLMCmll(gp.mlls.ExactMarginalLogLikelihood):
    """
    The loss function for the ProjectedGPModel. 
    """
    def __init__(self, latent_likelihood:Likelihood, model:ProjectedGPModel):
        """

        Args:
            latent_likelihood: the likelihood of a ProjectedGPModel (batched gaussian likelihood of size n_latents)
            model: any ProjectedGPModel.

        Raises:
            RuntimeError: rejects non-gaussian likelihoods.
        """        
        if not isinstance(latent_likelihood, gp.likelihoods.gaussian_likelihood._GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ProjectedLMCmll, self).__init__(latent_likelihood, model)
        self.previous_lat = None


    def forward(self, latent_function_dist:gp.distributions.Distribution, target:Tensor, inputs=None, *params):
        """
        Computes the value of the loss (MLL) given the model predictions and the observed values at training locations. 
        Args:
            latent_function_dist: gp batched gaussian distribution of size n_latents x n_points representing the values of latent processes.
            target: training labels Y of shape n_points x n_tasks

        Raises:
            RuntimeError: rejects non-gaussian latent distributions.

        Returns:
            The (scalar) value of the MLL loss for this model and data.
        """        
        if not isinstance(latent_function_dist, gp.distributions.multivariate_normal.MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        num_data = latent_function_dist.event_shape.numel()
        
        # project the targets
        proj_target = self.model.project_data(target) # shape n_latents x n_points

        # Get the log prob of the marginal distribution of latent processes
        latent_output = self.likelihood(latent_function_dist, *params) # shape n_latents x n_points
        latent_res = latent_output.log_prob(proj_target)
        latent_res = self._add_other_terms(latent_res, params).sum().div_(num_data)  # Scale by the amount of data we have

        # compute the part of likelihood lost by projection
        p, q = self.model.n_tasks, self.model.n_latents
        self.proj_term_list = [0]*3
        ## We store the additional terms in a list attribute in order to be able to plot them individually for testing
        Q, R, Q_orth = self.model.lmc_coefficients.QR()
        if not hasattr(self.model, 'M') and self.model.scalar_B:
            if self.model.log_B_tilde.numel() > 0:
                # log_B_tilde = torch.clamp(self.model.log_B_tilde, -9, 9)
                log_B_tilde = self.model.log_B_tilde
                B_tilde_inv_val = torch.exp(- log_B_tilde[0])
                log_B_tilde_root_diag = log_B_tilde / 2
                self.proj_term_list[1] = - 0.5 * B_tilde_inv_val * (self.model.Y_squared_norm - (target @ Q).pow(2).sum()).div_(num_data)
            else:
                self.proj_term_list[1] = 0.
                log_B_tilde_root_diag = torch.tensor([0.])
        else:
            if self.model.diagonal_B:
                log_B_tilde_root_diag = self.model.log_B_tilde / 2
                B_tilde_inv = torch.diag_embed(torch.exp(- self.model.log_B_tilde))
                rot_proj_target = target @ Q_orth
                discarded_noise_term = rot_proj_target @ B_tilde_inv @ rot_proj_target.T
            else:
                B_tilde_inv_root_diag = self.model.B_tilde_inv_chol[range(p-q), range(p-q)]
                log_B_tilde_root_diag = -torch.log(B_tilde_inv_root_diag)
                discarded_noise_root = target @ Q_orth @ self.model.B_tilde_inv_chol
                discarded_noise_term = discarded_noise_root @ discarded_noise_root.T
            self.proj_term_list[1] = - 0.5 * torch.trace(discarded_noise_term).div_(num_data)

        # All terms are implicitly or explicitly divided by the number of datapoints
        self.proj_term_list[0] = - 0.5 * 2 * torch.sum(log_B_tilde_root_diag) # factor 2 because of the use of a root
        if self.model.lmc_coefficients.bulk:
            self.proj_term_list[2] = - 0.5 * torch.log(R[range(q), range(q)]**2).sum()
        else:
            self.proj_term_list[2] = - 0.5 * 2 * self.model.lmc_coefficients.parametrizations.R.original[range(q), range(q)].sum()
        projection_term = sum(self.proj_term_list) - 0.5 * (p - q) * np.log(2*np.pi)

        res = latent_res + projection_term
        return res

class LazyLMCModel(gp.models.ExactGP):
    """
    A training-less LMC-like model.
    """
    def __init__( self, train_x:Tensor, train_y:Tensor, n_latents:int, mean_type:Mean=gp.means.ConstantMean, outputscales:bool=False, 
                  decomp:Union[List[List[int]],None]=None,
                  noise_val:float=1e-4,
                  ker_kwargs:Union[dict,None]=None, **kwargs):
        """
        Args:
            train_x: training input data
            train_y: training input labels
            n_latents: number of latent processes
            mean_type: gp mean function for task-level processes. Defaults to gp.means.ConstantMean.
        """
        proj_likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_latents]),
                                        noise_constraint=gp.constraints.GreaterThan(np.exp(noise_val))) #useless, only for inheritance
        super().__init__(train_x, torch.zeros_like(train_y), proj_likelihood, n_tasks=n_latents, 
                         mean_type=gp.means.ZeroMean, outputscales=outputscales, **kwargs) # !! proj_likelihood will only be named likelihood in the model
        self.register_buffer('train_y', train_y)

        n_data, n_tasks = train_y.shape

        self.n_tasks = n_tasks
        self.n_latents = n_latents
        self.latent_dim = -1
        self.eps = eps



