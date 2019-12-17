# [First passage times](@id FPT_header)
The package works readily in a setting where its `smooth` coordinates (coordinates whose volatility coefficients are identically equal to zero) are observed only at first-passage times to some fixed thresholds (these could be counted as first up-crossings or first down-crossings).

If we observe the first-passage time to level ``L`` of coordinate ``i`` at time
``t``, then we need to first pass it as a usual partial observation of
coordinate ``i`` at time ``t`` to take value ``L`` and additionally we need
to specify in a separate struct that this is the first passage time observation.
This latter specification is done by the function `set_observations!`
```@docs
set_observations!
```
The last argument (`fpt`) is supposed to be a vector whose each element
corresponds to a separate observation and gives additional information regarding
its first-passage time nature. If it is just a partial observation, then this
object can take value `nothing` (this is the default behaviour for all elements),
alternatively, we can define an instance of a struct `FPTInfo`
```@docs
FPTInfo
```
which indicates that first-passage times are observed, it says which coordinates
are observed, whether it is the up- or down-crossing, and whether the diffusion
is conditioned on having reached `reset` level specified as the fourth parameter
since the previous observation time (in which case `autoRenewed` is set to `false` and otherwise `true`).
