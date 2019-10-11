# BridgeSDEInference.jl

MCMC sampler for inference for diffusion processes with the use of Guided
Proposals using the package [Bridge.jl](https://github.com/mschauer/Bridge.jl).
Currently under development.


## Problem statements
### Partially observed diffusion
Consider a stochastic process ``X``, a solution to the stochastic differential
equation
```math
\begin{equation}\label{eq:mainSDE}
d X_t = b_{\theta}(t,X_t)dt + \sigma_{\theta}(t,X_t)dW_t,\quad t\in[0,T],\quad X_0=x_0,
\end{equation}
```
where ``b_{\theta}:R^d\to R^d`` is the drift function,
``\sigma_{\theta}:R^d\to R^{d\times d'}`` is the volatility coefficient
and ``W`` is a ``d'``-dimensional standard Brownian motion. We refer to ``X``
as a `diffusion`. ``\theta\in R^p`` is some unknown parameter. Suppose that a
linearly transformed ``X``, perturbed by Gaussian noise is observed at some
collection of time points, i.e. that
```math
\begin{equation}\label{eq:partObsData}
D:=\{V_{t_i}; i=1,\dots N\},
\end{equation}
```
is observed for some ``t_i, i=1,\dots N``, where
```math
\begin{equation}\label{eq:partialObservations}
V_{t_i}=L_i X_{t_i} + \xi_i,
\end{equation}
```
with ``L_i\in R^{d_i\times d}`` and independent ``\xi_i\sim Gsn(0,\Sigma_i)``,
``(i=1,\dots,d)``. Suppose further that ``\theta`` is equipped with some `prior`
distribution ``\pi(\theta)``. The aim is to estimate the `posterior`
distribution over ``\theta``, given the observation set `D`:
```math
\begin{equation}\label{eq:posterior}
\pi(\theta|D)\propto \pi(\theta)\pi(D|\theta).
\end{equation}
```

### First passage time observations
Consider a stochastic differential equation (\ref{eq:mainSDE}) and suppose
that the first coordinate of the drift term ``b_{\theta}^{[1]}(t,x):R^d\to R``
is linear in ``x``, whereas the first row of the volatility coefficient is
identically equal to a zero vector ``\sigma_{\theta}^{[1,1:d']}=0``. Suppose
further that instead of partial observation scheme as in
(\ref{eq:partObsData}), the process ``X`` is observed at a collection of
stopping times:
```math
D:=\{\tau^\star_i, i=1,\dots,N\},
```
where ``\tau^\star_i``'s are defined by
```math
\begin{align*}
\tau_{\star,0}&:=0\\
\tau^\star_{i+1}&:=\inf\{t\geq \tau_{\star,i}: X_t^{[1]}\geq l^\star_{i+1}\},\quad i=0,\dots\\
\tau_{\star,i}&:=\inf\{t\geq \tau^\star_{i}: X_t^{[1]}\leq l_{\star,i}\},\quad i=1,\dots,
\end{align*}
```
for some known constants ``l^\star_i``, ``l_{\star,i}``, ``i=1,\dots``. Note
that ``\tau^\star_i``'s are the first passage times of the first coordinate
process to some thresholds ``l^\star_i``, whereas ``\tau_{\star,i}`` are the
(latent) renewal times. The aim is to find a posterior (\ref{eq:posterior})
from such first passage time data.

### Mixed-effects models
Consider ``M`` stochastic differential equations
```math
\begin{equation}
d X^{(i)}_t = b_{\theta,\eta_i}(t,X^{(i)}_t)dt + \sigma_{\theta,\eta_i}(t,X^{(i)}_t)dW^{(i)}_t,\quad t\in[0,T],\quad X^{(i)}_0=x^{(i)}_0,\quad i=1,\dots,M.
\end{equation}
```
where ``\theta\in R^{p}`` denotes the parameter that is shared among all of
``X^{(i)}``'s, whereas ``\eta_i\in R^{p_i}``, ``i=1,\dots,M`` are the
parameters specific to a given ``X^{(i)}``. Suppose that the observations
of the type (\ref{eq:partObsData}) are given for each trajectory ``X^{(i)}``
(let's denote a joint set with ``D^\star:=\cup_{i=1}^M D_i``). The aim is to
find the posterior:
```math
\pi(\theta,\{\eta_i,i=1,\dots M\}|D^{\star})\propto \pi(\theta)\prod_{i=1}^M\pi(\eta_i)\pi(D_i|\theta,\eta_i).
```

## Overview of the solutions in BridgeSDEInference.jl

The first two problems from [Overview of mathematical problems](@ref) are
addressed by the function
```@docs
mcmc
```
The third problem ([Mixed-effects models](@ref)) is addressed by the function
(TODO fix so that points to a function in `repeated.jl`)
```@docs
mcmc
```

Please see the [Tutorial](@ref) section to see how to appropriately initialise
`setup`, run the `mcmc` function and query the results.

## References
**These are only the references which describe the algorithms implemented in
this package**. Note in particular that the list of references which treat the
same problems as addressed by this package, but use methods which are not
based on `guided proposals` is **much, much longer**. We refer to the
bibliography sections of the papers listed below for references to other
approaches.


### Partial observations of a diffusion
* Guided proposals for diffusion bridges (no-noise setting):
    - Moritz Schauer, Frank van der Meulen, Harry van Zanten. *Guided proposals for simulating multi-dimensional diffusion bridges.* Bernoulli, 23(4A), 2017, pp. 2917–2950. [[Bernoulli](https://projecteuclid.org/euclid.bj/1494316837)], [[arXiv](https://arxiv.org/abs/1311.3606)].
* Bayesian inference with guided proposals for diffusions observed exactly and discretely in time:
  - Frank van der Meulen, Moritz Schauer. *Bayesian estimation of discretely observed multi-dimensional diffusion processes using guided proposals.* Electronic Journal of Statistics 11 (1), 2017. [[EJoS](https://projecteuclid.org/euclid.ejs/1495850628)], [[arXiv](https://arxiv.org/abs/1406.4704)].
* Bayesian inference with guided proposals for diffusions observed with noise (according to (\ref{eq:partialObservations})):
  - Frank van der Meulen, Moritz Schauer. *Bayesian estimation of incompletely observed diffusions.* Stochastics 90 (5), 2018, pp. 641–662. [[Stochastics](https://www.tandfonline.com/doi/full/10.1080/17442508.2017.1381097)], [[arXiv](https://arxiv.org/abs/1606.04082)].
* Simulation of hypo-elliptic diffusion bridges:
  - Joris Bierkens, Frank van der Meulen, Moritz Schauer. *Simulation of elliptic and hypo-elliptic conditional diffusions.* arXiv, 2018. [[arXiv](https://arxiv.org/abs/1810.01761)].
* Efficient schemes for computing all the necessary term for a fully generic implementation of guided proposals:
  - Frank van der Meulen, Moritz Schauer. *Continuous-discrete smoothing of diffusions.* arXiv, 2017. [[arXiv](https://arxiv.org/abs/1712.03807)].

### First passage time set_observations
None published
### Mixed-effect models
None published
