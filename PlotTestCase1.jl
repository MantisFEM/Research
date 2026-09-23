using Mantis
using GLMakie
using DataFrames
using CSV

foldername = "."
ps = [3] # [5]
num_elements = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]

function plot_reference_results(ax, which, p)
    if which == "L2" && p == 3
        x_ap = sqrt.([49, 169, 625, 2401, 9409])
        y_ap = [3.75e-01, 9.64e-02, 2.19e-02, 5.36e-03, 1.33e-03]
        scatterlines!(
            ax,
            x_ap,
            y_ap;
            label=L"\text{FEM A\&P}",
            color=colours[2],
            marker=markers[2],
            markersize=10,
        )
        C = y_ap[4] / (x_ap[4]^(-2))
        lines!(ax, x_ap, C .* (x_ap .^ (-2)); linestyle=:dot, color=:black, label=L"O(N^{-2})")

        x_mixed = sqrt.([20, 80, 320, 1280, 5120])
        y_mixed = [1.24e-01, 1.21e-02, 7.56e-04, 4.95e-05, 3.13e-06]
        scatterlines!(
            ax,
            x_mixed,
            y_mixed;
            label=L"\text{FEM Mixed}",
            color=colours[4],
            marker=markers[4],
            markersize=10,
        )
        C = y_mixed[4] / (x_mixed[4]^(-4))
        lines!(ax, x_mixed, C .* (x_mixed .^ (-4)); linestyle=:dash, color=:black)
    elseif which == "L2" && p == 5
        x_ap = sqrt.([121, 441, 1681, 6561, 25921])
        y_ap = [2.15e-02, 2.72e-04, 2.95e-06, 3.27e-08, 4.23e-10]
        scatterlines!(
            ax,
            x_ap,
            y_ap;
            label=L"\text{FEM A\&P}",
            color=colours[2],
            marker=markers[2],
            markersize=10,
        )
        C = y_ap[4] / (x_ap[4]^(-6))
        lines!(ax, x_ap, C .* (x_ap .^ (-6)); linestyle=:dash, color=:black)

        x_mixed = sqrt.([42, 168, 672, 2688, 10752])
        y_mixed = [2.80e-02, 9.08e-04, 1.27e-05, 2.08e-07, 3.28e-09]
        scatterlines!(
            ax,
            x_mixed,
            y_mixed;
            label=L"\text{FEM Mixed}",
            color=colours[4],
            marker=markers[4],
            markersize=10,
        )
        C = y_mixed[4] / (x_mixed[4]^(-6))
        lines!(
            ax, x_mixed, C .* (x_mixed .^ (-6)); linestyle=:dash, color=:black
        )


    elseif which == "jump" && p == 3
        x_mixed = sqrt.([20, 80, 320, 1280, 5120])
        y_mixed = [2.58e+00, 1.73e+00, 3.41e-01, 4.73e-02, 6.07e-03] .* 2.0
        scatterlines!(
            ax,
            x_mixed,
            y_mixed;
            label=L"\text{FEM Mixed}",
            color=colours[4],
            marker=markers[4],
            markersize=10,
        )
        C = y_mixed[4] / (x_mixed[4]^(-3))
        lines!(
            axwjump, x_mixed, C .* (x_mixed .^ (-3)); linestyle=:dash, color=:black
        )
    elseif which == "jump" && p == 5
        x_mixed = sqrt.([42, 168, 672, 2688, 10752])
        y_mixed = [4.53e+00, 4.73e-01, 1.98e-02, 6.63e-04, 2.11e-05] .* 2.0
        scatterlines!(
            ax,
            x_mixed,
            y_mixed;
            label=L"\text{FEM Mixed}",
            color=colours[4],
            marker=markers[4],
            markersize=10,
        )
        C = y_mixed[4] / (x_mixed[4]^(-5))
        lines!(
            axwjump, x_mixed, C .* (x_mixed .^ (-5)); linestyle=:dash, color=:black
        )


    elseif which == "H2" && p == 3
        x_ap = sqrt.([49, 169, 625, 2401, 9409])
        y_ap = [1.41e+01, 6.49e+00, 3.08e+00, 1.52e+00, 7.56e-01]
        scatterlines!(
            ax,
            x_ap,
            y_ap;
            label=L"\text{FEM A\&P}",
            color=colours[2],
            marker=markers[2],
            markersize=10,
        )
        C = y_ap[4] / (x_ap[4]^(-1))
        lines!(
            ax,
            x_ap,
            C .* (x_ap .^ (-1));
            linestyle=:dot,
            color=:black,
            label=L"O(N^{-1})",
        )

        x_mixed = sqrt.([20, 80, 320, 1280, 5120])
        y_mixed = [1.31e+01, 6.26e+00, 2.00e+00, 5.33e-01, 1.35e-01]
        scatterlines!(
            ax,
            x_mixed,
            y_mixed;
            label=L"\text{FEM Mixed}",
            color=colours[4],
            marker=markers[4],
            markersize=10,
        )
        C = y_mixed[4] / (x_mixed[4]^(-2))
        lines!(
            axwH2, x_mixed, C .* (x_mixed .^ (-2)); linestyle=:dash, color=:black
        )
    elseif which == "H2" && p == 5
        x_ap = sqrt.([121, 441, 1681, 6561, 25921])
        y_ap = [3.25e+00, 2.26e-01, 1.34e-02, 7.61e-04, 4.50e-05]
        scatterlines!(
            ax,
            x_ap,
            y_ap;
            label=L"\text{FEM A\&P}",
            color=colours[2],
            marker=markers[2],
            markersize=10,
        )
        C = y_ap[4] / (x_ap[4]^(-4))
        lines!(axwH2, x_ap, C .* (x_ap .^ (-4)); linestyle=:dash, color=:black)

        x_mixed = sqrt.([42, 168, 672, 2688, 10752])
        y_mixed = [1.21e+01, 1.99e+00, 1.18e-01, 7.81e-03, 4.96e-04]
        scatterlines!(
            ax,
            x_mixed,
            y_mixed;
            label=L"\text{FEM Mixed}",
            color=colours[4],
            marker=markers[4],
            markersize=10,
        )
        C = y_mixed[4] / (x_mixed[4]^(-4))
        lines!(
            axwH2, x_mixed, C .* (x_mixed .^ (-4)); linestyle=:dash, color=:black
        )
    end
end

function plot_ax!(
    fig, ax, which_legend, x, y, label, slope1, slope2=nothing; marker=:circle, p=1, colours=nothing
)
    scatterlines!(ax, x, y; label=label, color=:red, marker=marker, markersize=10)
    C = y[end] / (x[end]^slope1)
    lines!(
        ax,
        x,
        C .* (x .^ slope1);
        label=L"O(N^{%$(slope1)})",
        linestyle=:dash,
        color=:black,
    )
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
    ylabel=L"||\text{var} - dw_{exact}||_{L^2}",
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
const markers2 = (:circle, :circle, :xcross, :star4, :cross)
const markers2t = (:rect, :vline, :diamond, :utriangle, :rect)
const markers2C1 = (:diamond, :diamond, :diamond, :pentagon)
const colours2 = (:tomato, :tomato, :red, :firebrick)
const colours2t = (:tomato, :turquoise, :turquoise1, :turquoise3)
const colours2C1 = (:limegreen, :limegreen, :limegreen, :green)
const styles = ((:dash, :loose), :dash, :dot, :dashdot, :dashdotdot, (:dot, :loose), (:dashdot, :loose))

for p in ps
    r = p - 1
    filename = "TestCase1-nels$(num_elements)-p$(p)-r$(r).csv"
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

    plot_reference_results(axwL2, "L2", p)
    plot_ax!(
        figL2,
        axwL2,
        (1, 2),
        sqrt.(num_dofs),
        errors_L2,
        L"p=%$(p)",
        -(p + 1);
        marker=:circle,
        p=p,
    )

    plot_ax!(
        figH1,
        axwH1,
        (1, 2),
        hs,
        errors_gradw,
        L"\nabla w_h, p=%$(p)",
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
        L"\theta, p=%$(p)",
        (p + 1);
        marker=markers2t[p],
        p=p,
        colours=colours2t
    )

    plot_reference_results(axwjump, "jump", p)
    plot_ax!(
        figjump,
        axwjump,
        (1, 2),
        sqrt.(num_dofs),
        errors_jump,
        L"p=%$(p)",
        -p;
        marker=:circle,
        p=p,
    )

    plot_reference_results(axwH2, "H2", p)
    plot_ax!(
        figH2,
        axwH2,
        (1, 2),
        sqrt.(num_dofs),
        errors_H2,
        L"p=%$(p)",
        -(p-1);
        marker=:circle,
        p=p,
    )
end

Legend(figL2[1, 2], axwL2)
Legend(figH1[1, 2], axwH1)
Legend(figjump[1, 2], axwjump)
Legend(figH2[1, 2], axwH2)
