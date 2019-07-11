[back to README](../README.md)
# Parametrisations of FitzHugh-Nagumo model
There are `5` various parametrisations of the FitzHugh-Nagumo model implemented in this package. We explain them in turn below
## `:regular`
This the most commonly encountered parametrisation of the FitzHugh-Nagumo model. The target, two dimensional process `(Y,X)` solves the following stochastic differential equation:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20dY_t%26%3D%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%28Y_t-Y_t%5E3-X_t%20&plus;%20s%5Cright%29dt%5C%5C%20dX_t%26%3D%5Cleft%28%5Cgamma%20Y_t%20-%20X_t%20&plus;%5Cbeta%5Cright%29dt%20&plus;%20%5Csigma%20dW_t%2C%20%5Cend%7Balign*%7D)

The proposal is taken to be a guided proposal with auxiliary law ![equation](https://latex.codecogs.com/gif.latex?%5Cwidetilde%7BP%7D) induced by the linear diffusion obtained by linearising FitzHugh-Nagumo diffusion at an end-point:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20d%5Cwidetilde%7BY%7D_t%20%26%3D%20%5Cleft%5B%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%28%20%281-3y_T%5E2%29%5Cwidetilde%7BY%7D_t%20-%20%5Cwidetilde%7BX%7D_t%20%5Cright%20%29%20&plus;%20%5Cfrac%7Bs%7D%7B%5Cepsilon%7D%20&plus;%202%5Cfrac%7By_T%5E3%7D%7B%5Cepsilon%7D%5Cright%20%5Ddt%2C%20%5C%5C%20d%5Cwidetilde%7BX%7D_t%20%26%3D%20%5Cleft%5B%5Cgamma%5Cwidetilde%7BY%7D_t%20-%20%5Cwidetilde%7BX%7D_t&plus;%5Cbeta%20%5Cright%20%5Ddt%20&plus;%20%5Csigma%20dW_t%2C%20%5Cend%7Balign*%7D)

## `:simpleAlter`
The target stochastic differential equations is re-parametrised in such a way that the first coordinate is given by the integrated second coordinate:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20dY_t%20%26%3D%20%5Cdot%7BY%7D_tdt%5C%5C%20d%5Cdot%7BY%7D_t%20%26%3D%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%28%281-%5Cgamma%29Y_t%20-Y_t%5E3%20-%5Cepsilon%20%5Cdot%7BY%7D_t%20&plus;s-%5Cbeta%20&plus;%20%281-3Y_t%5E2%29%5Cdot%7BY%7D_t%20%5Cright%20%29dt%20&plus;%20%5Cfrac%7B%5Csigma%7D%7B%5Cepsilon%7DdW_t%20%5Cend%7Balign*%7D)

The auxiliary law ![equation](https://latex.codecogs.com/gif.latex?%5Cwidetilde%7BP%7D) is now induced by a pair: `(I,B)`, where `B` is a scaled Brownian motion and `I` is an integrated `B`:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20dI_t%26%3DB_tdt%2C%5C%5C%20dB_t%26%3D%5Cfrac%7B%5Csigma%7D%7B%5Cepsilon%7DdW_t.%20%5Cend%7Balign*%7D)

## `:complexAlter`
The stochastic differential equation solved by the target process is the same as in `:simpleAlter`. However, the auxliary law ![equation](https://latex.codecogs.com/gif.latex?%5Cwidetilde%7BP%7D) is induced by a two-dimensional diffusion, where the second coordinate is a linear diffusion obtained from linearising ![equation](https://latex.codecogs.com/gif.latex?%5Cdot%7BY%7D) at an end-point and the first coordinate is an integrated second coordinate. If only the first coordinate is observed the proposal takes a form:

<!---
\begin{align*}
d\widetilde{Y}_t &= \widetilde{X}_t dt,\\
d\widetilde{X}_t &= \frac{1}{\epsilon}\left[ \left( 1-\gamma-3y_T^2 \right )\widetilde{Y}_t +\left( 1-\epsilon-3y_T^2 \right )\widetilde{X}_t + \left(2y_T^3+s-\beta \right )\right ]dt + \frac{\sigma}{\epsilon}dW_t.
\end{align*}
--->
![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20d%5Cwidetilde%7BY%7D_t%20%26%3D%20%5Cwidetilde%7BX%7D_t%20dt%2C%5C%5C%20d%5Cwidetilde%7BX%7D_t%20%26%3D%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%5B%20%5Cleft%28%201-%5Cgamma-3y_T%5E2%20%5Cright%20%29%5Cwidetilde%7BY%7D_t%20&plus;%5Cleft%28%201-%5Cepsilon-3y_T%5E2%20%5Cright%20%29%5Cwidetilde%7BX%7D_t%20&plus;%20%5Cleft%282y_T%5E3&plus;s-%5Cbeta%20%5Cright%20%29%5Cright%20%5Ddt%20&plus;%20%5Cfrac%7B%5Csigma%7D%7B%5Cepsilon%7DdW_t.%20%5Cend%7Balign*%7D)

On the other hand, if both coordinates are observed, the proposal is given by:

<!---
\begin{align*}
d\widetilde{Y}_t &= \widetilde{X}_t dt,\\
d\widetilde{X}_t &= \frac{1}{\epsilon}\left[ \left( 1-\gamma-3y_T^2 - 6y_T\dot{y}_T \right )\widetilde{Y}_t +\left( 1-\epsilon -3y_T^2 \right )\widetilde{X}_t + \left(2y_T^3+s-\beta + 6y_T^2\dot{y}_T \right )\right ]dt + \frac{\sigma}{\epsilon}dW_t.
\end{align*}
--->
![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20d%5Cwidetilde%7BY%7D_t%20%26%3D%20%5Cwidetilde%7BX%7D_t%20dt%2C%5C%5C%20d%5Cwidetilde%7BX%7D_t%20%26%3D%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%5B%20%5Cleft%28%201-%5Cgamma-3y_T%5E2%20-%206y_T%5Cdot%7By%7D_T%20%5Cright%20%29%5Cwidetilde%7BY%7D_t%20&plus;%5Cleft%28%201-%5Cepsilon%20-3y_T%5E2%20%5Cright%20%29%5Cwidetilde%7BX%7D_t%20&plus;%20%5Cleft%282y_T%5E3&plus;s-%5Cbeta%20&plus;%206y_T%5E2%5Cdot%7By%7D_T%20%5Cright%20%29%5Cright%20%5Ddt%20&plus;%20%5Cfrac%7B%5Csigma%7D%7B%5Cepsilon%7DdW_t.%20%5Cend%7Balign*%7D)

## `:simpleConjug`
It is defined analogously to `:simpleAlter`, the only difference being that an additional step is taken of redefining the parameters:

![equation](https://latex.codecogs.com/gif.latex?s%5Cleftarrow%20%5Cfrac%7Bs%7D%7B%5Cepsilon%7D%2C%5Cquad%20%5Cbeta%5Cleftarrow%20%5Cfrac%7B%5Cbeta%7D%7B%5Cepsilon%7D%2C%5Cquad%20%5Csigma%5Cleftarrow%20%5Cfrac%7B%5Csigma%7D%7B%5Cepsilon%7D%2C%5Cquad%20%5Cgamma%5Cleftarrow%20%5Cfrac%7B%5Cgamma%7D%7B%5Cepsilon%7D%2C%5Cquad%20%5Cepsilon%5Cleftarrow%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%2C)

## `:complexConjug`
It is defined analogously to `:complexAlter`, the only difference being that an additional step is taken of redefining the parameters (just as it was done in `:simpleConjug` above).
