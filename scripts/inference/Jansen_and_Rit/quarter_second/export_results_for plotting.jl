#after running JR_quarter_second.jl
df1 = DataFrame(obs_time = obs_time, obs = [obs[i][1] for i in 1:length(obs))
CSV.write("./frank/df1.csv", df1)

time = repeat(out.time, length(out.paths))
function save_df(out)
    I = length(out.time)
    J = length(out.paths)
    x1 = zeros(J*I)
    x2 = zeros(J*I)
    x3 = zeros(J*I)
    x4 = zeros(J*I)
    x5 = zeros(J*I)
    x6 = zeros(J*I)
    iter = fill(0, J*I)
    k = 1
    for j in 1:J
        for i in 1:I
            x1[k], x2[k] = (out.paths[j][i][1], out.paths[j][i][2])
            x3[k], x4[k] = (out.paths[j][i][3], out.paths[j][i][4])
            x5[k], x6[k] = (out.paths[j][i][5], out.paths[j][i][6])
            iter[k] = j
            k = k + 1
        end
    end
    x1, x2, x3, x4, x5, x6, iter
end

x1, x2, x3, x4, x5, x6, iter = save_df(out)


df2 = DataFrame(time = time, iter = iter, x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6)
CSV.write("./frank/df2.csv", df2)


chain = chains.θ_chain
df3 = DataFrame(iter = 1:length(chain), par_a = [chain[i][2] for i in 1:length(chain)], par_b = [chain[i][4] for i in 1:length(chain)],
                par_C = [chain[i][5] for i in 1:length(chain)], par_muy = [chain[i][10] for i in 1:length(chain)],
                 par_sigmay = [chain[i][13] for i in 1:length(chain)])
CSV.write("./frank/df3.csv", df3)
