
function plotbasis(
    space::Forms.FormSpace{1},
    num_plot_points_per_element=25;
    title="",
    xlabel=L"x",
    ylabel=L"b_i(x)",
)
    fig = Figure()
    ax = Axis(fig[1, 1]; title=title, xlabel=xlabel, ylabel=ylabel)

    geometry = Forms.get_geometry(space)

    n_elements = Geometry.get_num_elements(geometry)
    xi = Points.CartesianPoints((LinRange(0.0, 1.0, num_plot_points_per_element),))
    BFF = Forms.FormField(space, space.label)

    dim_V = Forms.get_num_basis(space)
    colors = [:blue, :green, :red, :purple, :orange, :black, :pink, :brown]
    color_idx = 1
    for basis_idx in 1:dim_V
        BFF.coefficients[basis_idx] = 1.0
        if basis_idx > 1
            BFF.coefficients[basis_idx - 1] = 0.0
        end

        color_idx += 1
        if color_idx > length(colors)
            color_idx = 1
        end
        color_i = colors[color_idx]

        for element_idx in 1:n_elements
            form_eval, _ = Forms.evaluate(BFF, element_idx, xi)
            x = Geometry.evaluate(geometry, element_idx, xi)

            lines!(
                ax, x[:], form_eval[1]; color=color_i, label=L"%$(BFF.label)_{%$basis_idx}"
            )

            scatter!(ax, x[:][[1, end]], [0.0, 0.0]; color=:tomato)
        end
    end
    fig[1, 2] = Legend(fig, ax; marge=true, unique=true)

    return fig
end

function plot_solution(
    fields::T,
    num_plot_points_per_element=25;
    title=L"Solution",
    xlabel=L"x",
    ylabel=L"\phi(x)",
) where {n_fields, T <: NTuple{n_fields, Forms.AbstractFormField{1}}}
    fig = Figure()
    ax = Axis(fig[1, 1]; title=title, xlabel=xlabel, ylabel=ylabel)

    geometry = Forms.get_geometry(fields[1])

    n_elements = Geometry.get_num_elements(geometry)
    xi = Points.CartesianPoints((LinRange(0.0, 1.0, num_plot_points_per_element),))

    colors = [:blue, :green, :red, :purple, :orange, :black, :pink, :brown]
    for field_id in eachindex(fields)
        field = fields[field_id]
        color_i = colors[field_id]
        for element_idx in 1:n_elements
            form_eval, _ = Forms.evaluate(field, element_idx, xi)
            x = Geometry.evaluate(geometry, element_idx, xi)

            lines!(ax, x[:], form_eval[1]; color=color_i, label=L"%$(field.label)")
        end
    end
    fig[1, 2] = Legend(fig, ax; marge=true, unique=true)

    return fig
end
