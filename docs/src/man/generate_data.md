# Generation of data
There are three short scripts for generating data; they are available [here](https://github.com/mmider/BridgeSDEInference.jl/blob/master/scripts/data_generation). They illustrate how data can be generated in the setting of partially observed diffusion processes, first passage time observations as well as repeated observations (the last will soon be changed to mixed-effect models). In all three the main workhorse routine is `simulate_segment`.
```@docs
simulate_segment
```
The FPT setting additionally uses `findCrossings`, defined in the same place.
```@docs
findCrossings
```
