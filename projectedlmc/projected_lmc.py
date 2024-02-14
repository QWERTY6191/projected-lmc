import copy
import torch
import gpytorch
from gpytorch.means.mean import Mean
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
import numpy as np
from torch import Tensor
from scipy.stats import qmc
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator
from linear_operator.operators.dense_linear_operator import to_linear_operator
from functools import reduce #, lru_cache
from typing import Union, List
##----------------------------------------------------------------------------------------------------------------------
## Utilitaries

class PolynomialMean(Mean):
    def __init__( self, input_size: int, degree: int = 3, batch_shape: torch.Size = torch.Size(), bias: bool = True):
        """A mean function made of an order-d (univariate) polynomial in each variable

        Args:
            input_size: dimension of the data (number of variables)
            degree: degree of the polynomial, defaults to 3
            bat in case of several batched tasks
            bias: whether or not to include a bias term, defaults to True
        """
        super().__init__()
        self.degree = degree
        for i in range(1, degree+1):
            self.register_parameter(name="weights_{0}".format(i),
                                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

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
        ker_kwargs: additional arguments to pass to the underlying gpytorch kernel. Defaults to None

    Returns:
        A gpytorch-compatible kernel
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
                l_priors[i_ker] = gpytorch.priors.MultivariateNormalPrior(loc=prior_scales[i_ker],
                                            covariance_matrix=torch.diag_embed(prior_scales[i_ker]*prior_width[i_ker]))
            else:
                l_priors[i_ker] = gpytorch.priors.NormalPrior(loc=prior_scales[i_ker],
                                                              scale=prior_scales[i_ker]*prior_width[i_ker])

    kernels_args = [{'ard_num_dims': len(idx_list), 'active_dims': idx_list, 'lengthscale_prior': l_priors[i_ker],
                         'batch_shape': torch.Size([n_funcs])} for i_ker, idx_list in enumerate(decomp)]

    kernels = []
    for i_ker, ker_args in enumerate(kernels_args):
        ker = kernel(**ker_args, **ker_kwargs)
        kernels.append(ker)

    if len(decomp) > 1 :
        covar_module = gpytorch.kernels.ScaleKernel(kernels[0])
        for ker in kernels[1:]:
            covar_module += gpytorch.kernels.ScaleKernel(ker)
    else:
        if outputscales:
            covar_module = gpytorch.kernels.ScaleKernel(kernels[0], batch_shape=torch.Size([n_funcs]))
        else:
            covar_module = kernels[0]

    if prior_scales is not None:
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


class ScalarParam(torch.nn.Module):
    """
    Torch parametrization for a scalar matrix.
    """
    def forward( self, X :Tensor)-> Tensor:
        return torch.ones_like(X) * X.mean()
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
    def forward( self, X: Tensor )-> Tensor:
        lower = X.tril()
        lower[range(len(lower)), range(len(lower))] = torch.exp(lower[range(len(lower)), range(len(lower))])
        return lower
    def right_inverse( self, A: Tensor)-> Tensor:
        res = A
        res[range(len(res)), range(len(res))] = torch.log(res[range(len(res)), range(len(res))])
        return res

##----------------------------------------------------------------------------------------------------------------------

## GP models

class ExactGPModel(gpytorch.models.ExactGP):
    """
    Standard exact GP model. Can handle independant multitasking via batch dimensions
    """
    def __init__( self, train_x:Tensor, train_y:Tensor, likelihood:Likelihood,
                  n_tasks: int = 1, prior_scales:Union[Tensor, None]=None,
                  prior_width:Union[Tensor, None]=None, mean_type:Mean=gpytorch.means.ConstantMean,
                  decomp:Union[List[List[int]], None]=None, outputscales:bool=True,
                  kernel_type:Kernel=gpytorch.kernels.RBFKernel,
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
            mean_type: gpytorch mean function for the outputs. Defaults to gpytorch.means.ConstantMean.
            kernel_type: . gpytorch kernel function for latent processes. Defaults to gpytorch.kernels.RBFKernel.
            decomp: instructions to create a composite kernel with subgroups of variables. Ex : decomp = [[0,1],[1,2]] --> k(x0,x1,x2) = k1(x0,x1) + k2(x1,x2). Defaults to None.
            outputscales: whether to endow the kernel with a learned scaling factor, k(.) = a*k_base(.). Defaults to True
            ker_kwargs: Additional arguments to pass to the gpytorch kernel function. Defaults to None.
            n_inducing_points: if an integer is provided, the model will use the sparse GP approximation of Titsias (2009).
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        if ker_kwargs is None:
            ker_kwargs = {}
        self.dim = train_x.shape[1]
        self.n_tasks = n_tasks
        self.mean_module = mean_type(input_size=self.dim, batch_shape=torch.Size([n_tasks]))
        self.covar_module = handle_covar_(kernel_type, dim=self.dim, decomp=decomp, prior_scales=prior_scales,
                                          prior_width=prior_width, outputscales=outputscales,
                                          n_funcs=n_tasks, ker_kwargs=ker_kwargs)
        if n_inducing_points is not None:
            self.covar_module = gpytorch.kernels.InducingPointKernel(self.covar_module, torch.randn(n_inducing_points, self.dim), likelihood)


    def forward( self, x:Tensor )-> Union[gpytorch.distributions.MultivariateNormal, gpytorch.distributions.MultitaskMultivariateNormal]:
        """
        Defines the computation performed at every call.
        Args:
            x: Data to evaluate the model at

        Returns:
            Prior distribution of the model output at the input locations. Can be a multitask multivariate normal if batch dimension is >1, or a multivariate normal otherwise
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # if self.n_tasks > 1 and not hasattr(self, 'compute_latent_distrib'): # for the batch case, but not the projected model inheritance
        if False: # for the batch case, but not the projected model inheritance
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x))
        else:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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

        self.mean_module = gpytorch.means.MultitaskMean(self.mean_module, num_tasks=n_tasks)
        
        if model_type=='ICM':
            self.covar_module = gpytorch.kernels.MultitaskKernel(self.covar_module, num_tasks=n_tasks, rank=n_latents)
        elif model_type=='LMC':
            self.covar_module = gpytorch.kernels.LCMKernel(base_kernels=[copy.deepcopy(self.covar_module) for i in range(n_latents)],
                                                           num_tasks=n_tasks, rank=1)

        if init_lmc_coeffs:
            SVD = TruncatedSVD(n_components=n_latents)
            y_transformed = torch.as_tensor(SVD.fit_transform(train_y.cpu().T)) / np.sqrt(len(train_y) - 1) #shape n_tasks x n_latents
            if model_type=='ICM':
                self.covar_module.task_covar_module.register_parameter(name='covar_factor', parameter=torch.nn.Parameter(y_transformed))
            elif model_type=='LMC':
                for i in range(n_latents):
                    self.covar_module.covar_module_list[i].task_covar_module.covar_factor = torch.nn.Parameter(y_transformed[:,i].unsqueeze(-1))
            else:
                raise ValueError('Wrong specified model type, should be ICM or LMC')

        if fix_diagonal:
            if model_type=='ICM':
                self.covar_module.task_covar_module.register_parameter(name='raw_var',
                                            parameter=torch.nn.Parameter(-20*torch.ones(n_tasks, device=train_y.device),
                                            requires_grad=False))
            elif model_type=='LMC':
                for i in range(len(self.covar_module.covar_module_list)):
                    self.covar_module.covar_module_list[i].task_covar_module.register_parameter(name='raw_var',
                                                parameter=torch.nn.Parameter(-20*torch.ones(n_tasks, device=train_y.device),
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
            ref_scales = [reduce(getattr, attr_name.split('.'), ker).squeeze() for ker in data_covar.kernels]
            ker_dims = [len(scales) if scales.ndim > 0 else 1 for scales in ref_scales]
            res = [torch.zeros((self.n_latents, ker_dims[i])) for i in range(n_kernels)]
        else:
            n_kernels = 1
            ref_kernel = data_covar
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            ref_scales = reduce(getattr, attr_name.split('.'), data_covar).squeeze()
            ker_dim = len(ref_scales) if ref_scales.ndim > 0 else 1
            res = torch.zeros((self.n_latents, ker_dim))

        if self.model_type=='LMC':
            for i in range(self.n_latents):
                if n_kernels > 1:
                    for j in range(n_kernels):
                        res[j][i,:] = reduce(getattr, attr_name.split('.'), self.covar_module.covar_module_list[i].data_covar_module.kernels[j]).squeeze()
                else:
                    res[i,:] = reduce(getattr, attr_name.split('.'), self.covar_module.covar_module_list[i].data_covar_module).squeeze()
            return [res] if (n_kernels==1 and not unpacked) else res
        
        else:
            return [ref_scales] if (n_kernels == 1 and not unpacked) else ref_scales
        
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
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class CustomLMCVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):  # allows to put means on tasks instead of latent processes
    """
    This small overlay to the native LMCVariationalStrategy of gpytorch allows to put deterministic means on tasks rather than latent processes.
    """
    def __init__(self, mean_module:gpytorch.means.mean, *args, **kwargs):
        """

        Args:
            mean_module: The already-generated, batched, many-tasks mean function to impose on output tasks.
        """
        super().__init__(*args, **kwargs)
        self.output_mean_module = mean_module

    def __call__(self, x:Tensor, task_indices=None, prior=False, **kwargs)-> gpytorch.distributions.MultitaskMultivariateNormal:
        """

        Args:
            x:Input data to evaluate model at.

        Returns:
            The posterior distribution of the model at input locations.
        """
        multitask_dist = super().__call__(x, task_indices=None, prior=False, **kwargs)
        tasks_means = self.output_mean_module(x)
        return multitask_dist.__class__(multitask_dist.mean + tasks_means.T, multitask_dist.lazy_covariance_matrix)


class VariationalMultitaskGPModel(gpytorch.models.ApproximateGP):
    """
    A standard variational LMC model using gpytorch functionalities.
    """
    def __init__( self, train_x:Tensor, n_latents:int, n_tasks:int, train_ind_ratio:float=1.5, seed:int=0,
                  init_lmc_coeffs:bool=False, train_y:Union[Tensor,None]=None, prior_scales:Tensor=None, prior_width:Tensor=None,
                  mean_type:Mean=gpytorch.means.ConstantMean, kernel_type:Kernel=gpytorch.kernels.RBFKernel, 
                  decomp:Union[List[List[int]],None]=None,
                  distrib:gpytorch.variational._VariationalDistribution=gpytorch.variational.CholeskyVariationalDistribution, 
                  var_strat:gpytorch.variational._VariationalStrategy=gpytorch.variational.VariationalStrategy,
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
            mean_type: gpytorch mean function for the outputs. Defaults to gpytorch.means.ConstantMean.
            kernel_type: . gpytorch kernel function for latent processes. Defaults to gpytorch.kernels.RBFKernel.
            decomp: instructions to create a composite kernel with subgroups of variables. Ex : decomp = [[0,1],[1,2]] --> k(x0,x1,x2) = k1(x0,x1) + k2(x1,x2). Defaults to None.
            distrib: gpytorch variational distribution for inducing values (see gpytorch documentation). Defaults to gpytorch.variational.CholeskyVariationalDistribution.
            var_strat: gpytorch variational strategy (see gpytorch documentation). Defaults to gpytorch.variational.VariationalStrategy.
            ker_kwargs: Additional arguments to pass to the gpytorch kernel function. Defaults to None.
        """

        if ker_kwargs is None:
            ker_kwargs = {}
        self.dim = train_x.shape[1]

        if float(train_ind_ratio) == 1.:
            print('Warning : inducing points not learned !')
            inducing_points = train_x
            learn_inducing_locations = False
            var_strat = gpytorch.variational.UnwhitenedVariationalStrategy  #better compatibility in this case
            distrib = gpytorch.variational.CholeskyVariationalDistribution  #better compatibility in this case
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
                                            prior_width=prior_width, n_funcs=n_latents, ker_kwargs=ker_kwargs, outputscales=False)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_latents])) #in gpytorch, latent processes can have non-zero means, which we wish to avoid

        self.n_tasks, self.n_latents, self.decomp = n_tasks, n_latents, decomp

        if init_lmc_coeffs:
            SVD = TruncatedSVD(n_components=n_latents)
            y_transformed = SVD.fit_transform(train_y.cpu().numpy().T) / np.sqrt(len(train_y) - 1)
            lmc_coefficients = torch.as_tensor(y_transformed).T
            if train_y.device.type=='cuda':
                lmc_coefficients = lmc_coefficients.cuda()
            self.variational_strategy.register_parameter("lmc_coefficients", torch.nn.Parameter(lmc_coefficients))  #shape n_latents x n_tasks

    def forward( self, x:Tensor )-> Tensor:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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
        return self.lmc_coefficents.data


class LMCMixingMatrix(torch.nn.Module):
    """
    Class for the parametrized mixing matrix of projected models. Making it a separate class allows to call 
    torch.nn.utils.parametrizations.orthogonal onto it during instanciation of a ProjectedGPModel
    """
    def __init__( self, Q_plus:Tensor, R:Tensor, n_latents:int):
        """

        Args:
            Q_plus: (augmented) orthonormal part of the mixing matrix. See the reference article for explanations
            R: upper triangular part of the mixing matrix. See the reference article for explanations
            n_latents: number of latent processes
        """
        super().__init__()
        self.register_parameter("Q_plus", torch.nn.Parameter(Q_plus, requires_grad=True))
        self.register_parameter("R", torch.nn.Parameter(R, requires_grad=True))
        self.n_latents = n_latents
        self.n_tasks = self.Q_plus.shape[0]
        self._size = torch.Size([self.n_latents, self.n_tasks])

    def Q( self ):
        return self.Q_plus[:, :self.n_latents]

    def Q_orth( self ):
        return self.Q_plus[:, self.n_latents:]

    def forward( self ):
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
                  mean_type:Mean=gpytorch.means.ZeroMean, **kwargs):
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
            mean_type: gpytorch mean function for task-level processes. Defaults to gpytorch.means.ConstantMean.
        """
        if proj_likelihood is None or proj_likelihood.noise.shape[0] != n_latents:
            print("Warning : in projected GP model the dimension of the likelihood is the number of latent processes. "
                  "Provided likelihood was the wrong shape or None, so it was replaced by a fresh one")
            proj_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_latents]))

        super().__init__(train_x, torch.zeros_like(train_y), proj_likelihood, n_tasks=n_latents, 
                         mean_type=gpytorch.means.ZeroMean, outputscales=False, **kwargs) 
        # !! proj_likelihood will be named 'likelihood' in model attributes
        self.register_buffer('train_y', train_y)

        if mean_type is not gpytorch.means.ZeroMean:
            raise ValueError('Projected GP model does not support non-zero output-wise means for now !')

        if init_lmc_coeffs:
            n_data, n_tasks = train_y.shape
            if n_data > n_tasks:
                U, Sigma, VT = randomized_svd(train_y.cpu().numpy(), n_components=n_tasks, random_state=0)
                Q_plus, R = torch.as_tensor(VT.T), torch.as_tensor(np.diag(Sigma[:n_latents]) / np.sqrt(n_data - 1))
            else:
                try:
                    Q_plus, R_padded, Vt = torch.linalg.svd(train_y.T, full_matrices=True) 
                    # we perform SVD rather than QR for the case where R must be diagonal (OILMM)
                except: # sklearn's randomized_svd is more stable than torch's svd in some cases
                    U, Sigma, Vt = randomized_svd(train_y.cpu().numpy().T, n_components=n_data, random_state=0)
                    Q_plus, R_padded = torch.as_tensor(U), torch.as_tensor(Sigma)
                if n_latents < n_data:
                    R_padded = R_padded[:n_latents]
                else:
                    R_padded, R_short = 1e-3*torch.ones(n_latents), R_padded
                    R_padded[:n_data] = R_short
                R = torch.diag_embed(R_padded) / np.sqrt(n_data - 1)
        else:
            fake_coeffs = torch.randn(n_tasks, n_latents)
            Q_plus, R_padded, Vt = torch.linalg.svd(fake_coeffs)
            # we perform SVD rather than QR for the case where R must be diagonal (OILMM)
            R = torch.diag_embed(R_padded[:n_latents])

        lmc_coefficients = LMCMixingMatrix(Q_plus, R, n_latents)
        lmc_coefficients = torch.nn.utils.parametrizations.orthogonal(lmc_coefficients, name="Q_plus")  # parametrizes Q_plus as orthogonal
        if diagonal_R:
            torch.nn.utils.parametrize.register_parametrization(lmc_coefficients, "R", PositiveDiagonalParam())
        else:
            torch.nn.utils.parametrize.register_parametrization(lmc_coefficients, "R", UpperTriangularParam())
        self.lmc_coefficients = lmc_coefficients

        if scalar_B:
            diagonal_B = True
            self.register_parameter("log_B_tilde", torch.nn.Parameter(np.log(1e-2) * torch.ones(n_tasks - n_latents)))
            torch.nn.utils.parametrize.register_parametrization(self, "log_B_tilde", ScalarParam())
        elif diagonal_B:
            self.register_parameter("log_B_tilde", torch.nn.Parameter(np.log(1e-2)*torch.ones(n_tasks - n_latents)))
        else:
            self.register_parameter("B_tilde_inv_chol", torch.nn.Parameter(torch.diag_embed(np.log(1e2)*torch.ones(n_tasks - n_latents))))
            torch.nn.utils.parametrize.register_parametrization(self, "B_tilde_inv_chol", LowerTriangularParam())
        self.diagonal_B = diagonal_B

        if not BDN:
            self.register_parameter("M", torch.nn.Parameter(torch.zeros((n_latents, n_tasks - n_latents))))

        self.n_tasks = n_tasks
        self.n_latents = n_latents
        self.latent_dim = -1
        self.eps = 1e2 * gpytorch.variational.settings.cholesky_jitter.value(dtype=torch.float64) if hasattr(gpytorch.variational, "settings") else 1e-3


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
        H_pinv = torch.linalg.solve_triangular(self.lmc_coefficients.R.T, self.lmc_coefficients.Q, upper=False, left=False)  # shape n_tasks x n_latents
        if hasattr(self, "M"):
            return H_pinv + self.lmc_coefficients.Q_orth() @ self.M.T * self.projected_noise()[None,:]
        else:
            return H_pinv

    def project_data( self, data:Tensor )-> Tensor:
        """
        Takes some data Z as input and returns its "projected" counterpart ZT.
        Args:
            data: some array of shape ... x n_tasks.

        Returns:
            The array ZT, where Z is the input and T the projection matrix from the reference article.
        """
        unscaled_proj = self.lmc_coefficients.Q().T @ data.T
        Hpinv_times_Y = torch.linalg.solve_triangular(self.lmc_coefficients.R, unscaled_proj, upper=True)  # shape n_latents x n_points ; opposite convention to most other quantities !!
        if hasattr(self, "M"):
            return Hpinv_times_Y + self.projected_noise()[:,None] * self.M @ self.lmc_coefficients.Q_orth().T @ data.T
        else:
            return Hpinv_times_Y

    def full_likelihood( self )-> gpytorch.likelihoods.MultitaskGaussianLikelihood :
        """
        Returns the full cross-tasks likelihood of the model.
        Returns:
            A multitask gaussian likelihood (gpytorch object) with covariance matrix Sigma (see reference article) 
        """        
        res = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_tasks, rank=self.n_tasks, has_global_noise=False)
        Q, Q_orth, R = self.lmc_coefficients.Q(), self.lmc_coefficients.Q_orth(), self.lmc_coefficients.R
        sigma_p = self.projected_noise()
        if sigma_p.is_cuda:
            res.cuda()
        if self.diagonal_B:
            B_tilde_root = torch.diag_embed(torch.exp(self.log_B_tilde / 2))
        else:
            B_tilde_root = torch.linalg.solve_triangular(self.B_tilde_inv_chol, 
                                            torch.eye(self.n_tasks - self.n_latents, device=self.B_tilde_inv_chol.device), upper=False).T
        QR = Q @ R
        if hasattr(self, "M"):
            B_tilde = B_tilde_root @ B_tilde_root.T
            B_term = Q_orth @ B_tilde @ Q_orth.T
            M_term = - QR @ (sigma_p[:,None] * self.M) @ B_tilde @ Q_orth.T
            Mt_term = M_term.T
            D_term_rotated = torch.diag_embed(sigma_p) + sigma_p[:,None] * self.M @ B_tilde @ self.M.T * sigma_p[None,:]
            D_term = QR @ D_term_rotated @ QR.T
        else:
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
                    print("Cholesky of the full noise covariance failed. Trying again with jitter {0} ...".format(eps))

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

    def forward( self, x:Tensor )-> gpytorch.distributions.MultivariateNormal:  # ! forward only returns values of the latent processes !
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x:Tensor, **kwargs)-> gpytorch.distributions.MultitaskMultivariateNormal:
        """
        Outputs the full posterior distribution of the model at input locations. This is used to make predictions.
        Args:
            x: input data tensor

        Returns:
            A multitask multivariate gpytorch normal distribution representing task processes values, which mean has shape n_points x n_tasks.
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

        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)
    

class ProjectedLMCmll(gpytorch.mlls.ExactMarginalLogLikelihood):
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
        if not isinstance(latent_likelihood, gpytorch.likelihoods.gaussian_likelihood._GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ProjectedLMCmll, self).__init__(latent_likelihood, model)
        self.previous_lat = None


    def forward(self, latent_function_dist:gpytorch.distributions.Distribution, target:Tensor, inputs=None, *params):
        """
        Computes the value of the loss (MLL) given the model predictions and the observed values at training locations. 
        Args:
            latent_function_dist: gpytorch batched gaussian distribution of size n_latents x n_points representing the values of latent processes.
            target: training labels Y of shape n_points x n_tasks

        Raises:
            RuntimeError: rejects non-gaussian latent distributions.

        Returns:
            The (scalar) value of the MLL loss for this model and data.
        """        
        if not isinstance(latent_function_dist, gpytorch.distributions.multivariate_normal.MultivariateNormal):
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
        Q_plus = self.model.lmc_coefficients.Q_plus
        Q, Q_orth = Q_plus[:, :q], Q_plus[:, q:]
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

        ## We store the additional terms in a list attribute in order to be able to plot them individually for testing
        # All terms are implicitly or explicitly divided by the number of datapoints
        self.proj_term_list = [0]*3
        self.proj_term_list[0] = - 0.5 * 2 * torch.sum(log_B_tilde_root_diag) # factor 2 because of the use of a root
        self.proj_term_list[1] = - 0.5 * torch.trace(discarded_noise_term).div_(num_data)
        self.proj_term_list[2] = - 0.5 * 2 * torch.trace(torch.abs(self.model.lmc_coefficients.R)) # factor 2 because of the use of a root
        projection_term = sum(self.proj_term_list) - 0.5 * (p - q) * np.log(2*np.pi)

        res = latent_res + projection_term
        return res




