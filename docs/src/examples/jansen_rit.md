# Jansen-Rit model
```math
\begin{equation*}
    \begin{aligned}
        d X_t &= \dot X_t d t  \\
        d Y_t &= \dot Y_t d t  \\
        d Z_t &= \dot Z_t d t \\
        d \dot X_t &=   \left[A a \left(\mu_x(t) + \mbox{Sigm}(Y_t - Z_t)\right) - 2a \dot X_t - a^2 X_t\right] d t + \sigma_x d W^{(1)}_t\\
        d \dot Y_t &=  \left[A a \left(\mu_y(t) + C_2\mbox{Sigm}(C_1 X_t)\right) - 2a \dot Y_t - a^2 Y_t\right] d t + \sigma_y d W^{(2)}_t\\
        d \dot Z_t &=  \left[B b \left(\mu_z(t) + C_4\mbox{Sigm}(C_3 X_t)\right) - 2b \dot Z_t - b^2 Z_t\right] d t + \sigma_z d W^{(3)}_t,
    \end{aligned}
\end{equation*}
```
with initial condition
```math
(X_0,Y_0,Z_0, \dot X_0, \dot Y_0, \dot Z_0)=(x_0,y_0,z_0, \dot x_0, \dot y_0, \dot z_0) \in R^6
```
where
```math
\mbox{Sigm}(x) := \frac{\nu_{max}}{1 + e^{r(v_0 - x)}},
```
and
```math
C_1 = C, \quad C_2 = 0.8C, \quad C_4 = C_3 = 0.25C.
```
