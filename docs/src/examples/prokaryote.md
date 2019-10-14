# Prokaryotic autoregulatory gene network
```math
dX_t = b(X_t)dt + \sigma(X_t) dW_t,
```
where
```math
b(x):=\left(\begin{matrix}
c_3 x^{[4]} - c_7 x^{[1]}\\
c_4 x^{[1]} + 2f(x) - c_8 x^{[3]}\\
g(x)-f(x)\\
g(x)
\end{matrix}\right)
```
with
```math
\begin{align*}
f(x)&:=c_6 x^{[3]}-0.5c_5x^{[2]}(x^{[2]}-1)\\
g(x)&:=c_2(K-x^{[4]})-c_1 x^{[3]}x^{[4]}
\end{align*}
```
and
```math
\sigma(x)
```
