using Mantis
using GLMakie
using DataFrames
using CSV

foldername = "."
ps = [2, 3, 4]
num_elements = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]

function plot_ax!(
    fig, ax, which_legend, x, y, label, slope1=nothing, slope2=nothing; marker=:circle, p=1, colours=nothing
)
    if isnothing(colours)
        scatterlines!(ax, x, y; label=label, color=colours2[p], marker=marker, markersize=10)
    else
        scatterlines!(ax, x, y; label=label, color=colours[p], marker=marker, markersize=10)
    end
    if !isnothing(slope1)
        C = y[end] / (x[end]^slope1)
        lines!(
            ax,
            x,
            C .* (x .^ slope1);
            label=L"O(N^{%$(slope1)})",
            linestyle=styles[abs(slope1)],
            color=:black,
        )
    end
    if !isnothing(slope2)
        C = y[end] / (x[end]^(slope2))
        lines!(
            ax,
            x,
            C .* (x .^ (slope2));
            label=L"O(N^{%$(slope2)})",
            linestyle=:dashdot,
            color=:black,
        )
    end

    return fig
end
pt = 4/3
figL2 = Figure(; fontsize = 17pt)
axwL2 = Axis(
    figL2[1, 1];
    xlabel=L"N = \sqrt{\text{num dofs}}",
    ylabel=L"||w_h - w_{exact}||_{L^2}",
    xscale=log10,
    yscale=log10,
)

figH1 = Figure(; fontsize = 17pt)
axwH1 = Axis(
    figH1[1, 1];
    xlabel=L"h",
    ylabel=L"||\text{var} - \nabla w_{exact}||_{L^2}",
    xscale=log10,
    yscale=log10,
)

figjump = Figure(; fontsize = 17pt)
axwjump = Axis(
    figjump[1, 1];
    xlabel=L"N = \sqrt{\text{num dofs}}",
    ylabel=L"||\nabla w_h \cdot \hat{n}|_R - \nabla w_{h}\cdot \hat{n}|_L ||_{L^{\infty}}",
    xscale=log10,
    yscale=log10,
)

figH2 = Figure(; fontsize = 17pt)
axwH2 = Axis(
    figH2[1, 1];
    xlabel=L"N = \sqrt{\text{num dofs}}",
    ylabel=L"||w_h - w_{exact}||_{\Delta}",
    xscale=log10,
    yscale=log10,
)

const colours = (:red, :blue, :green, :purple, :orange, :brown, :pink, :gray, :olive, :cyan)
const markers = (:circle, :rect, :diamond, :utriangle, :dtriangle, :rtriangle, :ltriangle)
const markers2 = (:circle, :circle, :xcross, :star4)
const markers2t = (:rect, :vline, :diamond, :utriangle)
const markers2C1 = (:diamond, :diamond, :diamond, :pentagon)
const colours2 = (:tomato, :tomato, :red, :firebrick)
const colours2t = (:tomato, :turquoise, :turquoise1, :turquoise3)
const colours2C1 = (:limegreen, :limegreen, :limegreen, :green)
const styles = ((:dash, :loose), :dash, :dot, :dashdot, :dashdotdot, (:dash, :loose))

for p in ps
    r = p - 1
    filename = "TestCase4-nels$(num_elements)-p$(p)-r$(r).csv"
    df = DataFrame(CSV.File(joinpath(foldername, filename)))
    hs = df[!, Symbol("h")]
    errors_L2 = df[!, Symbol("errors_w_L2_p$p")]
    errors_jump = df[!, Symbol("errors_w_jump_p$p")]
    errors_H1 = df[!, Symbol("errors_w_H1_p$p")]
    errors_gradw = errors_H1 .- errors_L2
    errors_H2 = df[!, Symbol("errors_w_H2_p$p")]
    errors_theta = df[!, Symbol("errors_theta_p$p")]
    num_dofs = df[!, Symbol("num_dofs_w_p$p")]
    num_dofs_theta = df[!, Symbol("num_dofs_theta_p$p")]

    plot_ax!(
        figL2,
        axwL2,
        (1, 2),
        sqrt.(num_dofs),
        errors_L2,
        L"p=%$(p)",
        -(p + 1);
        marker=markers2[p],
        p=p,
    )

    plot_ax!(
        figH1,
        axwH1,
        (1, 2),
        hs,
        errors_gradw,
        L"\nabla w_h,\ p=%$(p)",
        (p);
        marker=markers2[p],
        p=p,
    )
    plot_ax!(
        figH1,
        axwH1,
        (1, 2),
        hs,
        errors_theta,
        L"\theta_h,\ p=%$(p)",
        marker=markers2t[p],
        p=p,
        colours=colours2t
    )

    plot_ax!(
        figjump,
        axwjump,
        (1, 2),
        sqrt.(num_dofs),
        errors_jump,
        L"p=%$(p)",
        -p;
        marker=markers2[p],
        p=p,
    )

    plot_ax!(
        figH2,
        axwH2,
        (1, 2),
        sqrt.(num_dofs),
        errors_H2,
        L"p=%$(p)",
        -(p - 1);
        marker=markers2[p],
        p=p,
    )
end

Legend(figL2[1, 2], axwL2)
Legend(figH1[1, 2], axwH1, unique=true)
Legend(figjump[1, 2], axwjump)
Legend(figH2[1, 2], axwH2)
