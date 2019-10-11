# Lorenz '96 system
```math
dX_t = b_\theta(X_t)dt + \Sigma dW_t,\quad t\in[0,T],\quad X_0=x_0,
```
where ``X`` and ``W`` are `d`-dimensional, ``\Sigma`` is a ``d`` by ``d`` diagonal matrix and ``b`` is given by:
```math
b^{[i]}_\theta(x):= (x^{[i+1]}-x^{[i-2]})x^{[i-1]}-x^{[i]}+\theta
```
where ``i\in\{1,\dots,d\}`` is a cycling  index.
