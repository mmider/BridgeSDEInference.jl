macro timeit(f)
      start = time()
      out = eval(f)
      elapsed = time() - start
      print("Time elapsed: ", elapsed, "\n")
      (output = out, time = elapsed)
end

function std_aux_laws(AuxLaw, param, θ, obs, obs_time, idx=1:length(obs[1]))
      laws = [AuxLaw(param, θ₀..., t₀, u[idx], T, v[idx]) for (t₀,T,u,v)
              in zip(obs_time[1:end-1], obs_time[2:end], obs[1:end-1], obs[2:end])]
      display(laws[1])
      laws
end
