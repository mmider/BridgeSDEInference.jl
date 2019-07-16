using DataFrames
using CSV
using Gadfly, Colors

"""
    plotPaths(df; obs=[], obsTime=[], obsCoords=[])

Plot sampled paths stored in a dataframe. First two columns must be :idx and
:time. Additionally put observations from `obs` vector into a plot
"""
function plotPaths(df; obs=[], obsTime=[], obsCoords=[])
    coords = names(df)[3:end]
    plots = plot(df, x=:time, y=coords[1], color=:idx, Geom.line,
                 Scale.color_continuous(colormap=Scale.lab_gradient("#fceabb",
                                                                    "#a2acae",
                                                                    "#36729e")))
    if 1 in obsCoords
        q = layer(x=obsTime[1], y=obs[1], Geom.point,
                  Theme(default_color=color("seagreen")))
        append!(plots.layers, q)
    end

    for (i,c) in enumerate(coords[2:end])
        p = plot(df, x=:time, y=c, color=:idx, Geom.line,
                 Scale.color_continuous(colormap=Scale.lab_gradient("#fceabb",
                                                                    "#a2acae",
                                                                    "#36729e")))
        if i+1 in obsCoords
            q = layer(x=obsTime[i+1], y=obs[i+1], Geom.point,
                      Theme(default_color=color("seagreen")))
            append!(p.layers, q)
        end

        plots = vstack(plots, p)
    end
    draw(SVGJS("temp.js.svg", 40cm, 30cm), plots)
    display(plots)
end

"""
    plotChain(df; coords=:all)

Plot the MCMC chain
"""
function plotChain(df; coords=:all)
    if coords == :all
        coords = names(df)
    else
        coords = [names(df)[i] for i in coords]
    end
    m = length(df[coords[1]])
    plots = plot(df, y=coords[1], Geom.line)
    for c in coords[2:end]
        plots = vstack(plots, plot(df, y=c, Geom.line))
    end
    draw(SVGJS("temp.js.svg", 20cm, 10cm), plots)
    display(plots)
end
