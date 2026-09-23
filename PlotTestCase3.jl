using Mantis
using GLMakie
using DataFrames
using CSV

foldername = "."
ps = [2, 3, 4]
num_elements = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]

function plot_reference_data(ax, which, p)
    if p != 2
        return nothing
    end
    x_ac1 = sqrt.([48, 128, 408, 1448, 5448, 21128, 83208])

    if which == "L2"
        y_ac1 = [
            0.242937080572694,
            0.0638546878916581,
            0.0176368312872777,
            0.00474054487847352,
            0.00126320992536974,
            0.000335867836089744,
            8.92012839645476e-05,
        ]
        scatterlines!(
            ax,
            x_ac1,
            y_ac1;
            label=L"\text{Almost}\ C^1,\ p=2",
            color=colours2AC1[2],
            marker=markers2AC1[2],
            markersize=10,
        )
        C = y_ac1[4] / (x_ac1[4]^(-2))
        lines!(ax, x_ac1, C .* (x_ac1 .^ (-2)); linestyle=:dot, color=:black, label=L"O(N^{-2})")
    elseif which == "H1"
        hs_ac1 = [
            0.470173346377668,
            0.245325739517968,
            0.1251043465989,
            0.0631456978678055,
            0.0318516530631328,
            0.0160377873775608,
            0.00805643849324001,
        ]
        gradw_ac1 = [
            1.09126079876331,
            0.288461371923283,
            0.0797068650659393,
            0.0215769688867088,
            0.00581712271089407,
            0.00157200091974978,
            0.00042614268298479,
        ]
        scatterlines!(
            ax,
            hs_ac1,
            gradw_ac1;
            label=L"\nabla w_h\ \text{Almost.}\ C^1,\ p=2",
            color=colours2AC1[2],
            marker=markers2AC1[2],
            markersize=10,
        )
    elseif which == "jump"
        jump_ac1 = [
            7.83666708187963,
            3.78771809867396,
            1.93355883677652,
            0.988899803936026,
            0.50490560224412,
            0.257455742011498,
            0.131155650797735,
        ]
        scatterlines!(
            ax,
            x_ac1,
            jump_ac1;
            label=L"\text{Almost.}\ C^1,\ p=%$(p)",
            color=colours2AC1[2],
            marker=markers2AC1[2],
            markersize=10,
        )
        rate = -1
        C = jump_ac1[end] / ((x_ac1)[end]^(rate))
        lines!(
            axwjump,
            x_ac1,
            C .* (x_ac1 .^ (rate));
            label=L"O(N^{%$(rate)})",
            linestyle=styles[1],
            color=:black,
        )
    elseif which == "H2"
        yH2_ac1 = [
            0.356337461349162,
            0.214674985985322,
            0.117667546214677,
            0.0615580129598565,
            0.0313388896498537,
            0.0157919794628403,
            0.00792428871326978,
        ]
        scatterlines!(
            ax,
            x_ac1,
            yH2_ac1;
            label=L"\text{Almost.}\ C^1,\ p=%$(p)",
            color=colours2AC1[2],
            marker=markers2AC1[2],
            markersize=10,
        )
        rate = 1
        C = yH2_ac1[end] / ((x_ac1)[end]^(-rate))
        lines!(
            ax,
            x_ac1,
            C .* (x_ac1 .^ (-rate));
            linestyle=styles[p],
            color=:black,
        )
    end
end

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
const markers2AC1 = (:diamond, 'a', :diamond, :pentagon)
const colours2 = (:tomato, :tomato, :red, :firebrick)
const colours2t = (:tomato, :turquoise, :turquoise1, :turquoise3)
const colours2C1 = (:limegreen, :limegreen, :limegreen, :green)
const colours2AC1 = (:gold, :gold, :darkgoldenrod1, :darkgoldenrod3)
const styles = ((:dash, :loose), :dash, :dot, :dashdot, :dashdotdot, (:dash, :loose))

for p in ps
    r = p - 1
    q = p + 1
    q2 = 3 * p + 1
    filename = "TestCase3-nels$(num_elements)-p$(p)-r$(r).csv"
    df = DataFrame(CSV.File(joinpath(foldername, filename)))
    hs = df[!, Symbol("h")]
    errors_L2 = df[!, Symbol("errors_w_L2_p$p")]
    errors_jump = df[!, Symbol("errors_w_jump_p$p")]
    errors_H1 = df[!, Symbol("errors_w_H1_p$p")]
    errors_gradw = errors_H1 .- errors_L2
    errors_H2 = df[!, Symbol("errors_w_H2_p$p")]
    errors_theta = df[!, Symbol("errors_theta_p$p")]
    num_dofs = df[!, Symbol("num_dofs_w_p$p")]

    plot_reference_data(axwL2, "L2", p)
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

    plot_reference_data(axwH1, "H1", p)
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

    plot_reference_data(axwjump, "jump", p)
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

    plot_reference_data(axwH2, "H2", p)
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
