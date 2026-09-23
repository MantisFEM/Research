############################################################################################
#                                         Warning!                                         #
############################################################################################
# This is an experimental feature which is highly inefficient and should not be relied on.
# Moreover, this assumes a CartesianGeometry and will not work for other Geometries.
############################################################################################
#                                         Warning!                                         #
############################################################################################

############################################################################################
#                                        Structure                                         #
############################################################################################
struct PartialDerivative{manifold_dim, form_rank, expression_rank, G, F} <:
       AbstractForm{manifold_dim, form_rank, expression_rank, G}
    form::F
    orders::NTuple{manifold_dim, Int}
    label::String

    function PartialDerivative(
        form::F, orders::NTuple{manifold_dim, Int}
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        F <: AbstractForm{manifold_dim, form_rank, expression_rank, G},
    }
        if form_rank != 0
            throw(
                ArgumentError(
                    "Only 0-forms are currently supported. " *
                    "Form rank $(form_rank) was given.",
                ),
            )
        end

        if any(i -> i < 0, orders)
            throw(
                ArgumentError(
                    "Only positive derivative orders are valid. " *
                    "Orders $(orders) were given.",
                ),
            )
        end

        return new{manifold_dim, form_rank, expression_rank, G, F}(
            form, orders, "∂(" * get_label(form) * ")"
        )
    end
end

const ∂ = PartialDerivative

############################################################################################
#                                         Getters                                          #
############################################################################################

get_form(partial_der::PartialDerivative) = partial_der.form
get_geometry(partial_der::PartialDerivative) = get_geometry(get_form(partial_der))
get_num_basis(partial_der::PartialDerivative) = get_num_basis(get_form(partial_der))
get_orders(partial_der::PartialDerivative) = partial_der.orders

############################################################################################
#                                     Abstract method                                      #
############################################################################################

function evaluate(
    form::PartialDerivative{manifold_dim, form_rank, expression_rank, G, FS},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank, G, FS <: AbstractFormSpace}
    # Form evaluation
    form_space = get_form(form)
    num_components = binomial(manifold_dim, form_rank)
    partial_orders = get_orders(form)
    nderivatives = sum(partial_orders)
    form_eval, form_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_id, xi, nderivatives
    )
    der_idx = FunctionSpaces.get_derivative_idx([partial_orders...])
    partial_der_eval = [
        form_eval[nderivatives + 1][der_idx][component] for component in 1:num_components
    ]
    partial_der_eval = _add_geometric_scaling!(
        partial_der_eval, form_space, element_id, xi, partial_orders
    )

    return partial_der_eval, form_indices
end

function evaluate(
    form::PartialDerivative{manifold_dim, form_rank, expression_rank, G, FS},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank, G, FS <: AbstractFormField}
    # Form evaluation
    form_field = get_form(form)
    num_components = binomial(manifold_dim, form_rank)
    partial_orders = get_orders(form)
    nderivatives = sum(partial_orders)
    fs_eval, fs_indices = _evaluate_form_in_canonical_coordinates(
        get_form_space(form_field), element_id, xi, nderivatives
    )
    der_idx = FunctionSpaces.get_derivative_idx([partial_orders...])
    fs_partial_der_eval = [
        fs_eval[nderivatives + 1][der_idx][component] for component in 1:num_components
    ]
    fs_partial_der_eval = _add_geometric_scaling!(
        fs_partial_der_eval, get_form_space(form_field), element_id, xi, partial_orders
    )

    partial_der_eval = Vector{Vector{Float64}}(undef, num_components)
    form_field_coefficients = get_coefficients(form_field)
    for i in eachindex(partial_der_eval)
        partial_der_eval[i] =
            fs_partial_der_eval[i] * form_field_coefficients[fs_indices[1]]
    end

    return partial_der_eval, [[1]]
end

function _add_geometric_scaling!(
    partial_der_eval,
    form_space::AbstractFormSpace{manifold_dim},
    element_id,
    xi,
    partial_orders,
) where {manifold_dim}
    jacobian = Geometry.jacobian(get_geometry(form_space), element_id, xi)
    for i in 1:manifold_dim
        for ord_id in CartesianIndices(partial_der_eval[1])
            point, _ = Tuple(ord_id)
            partial_der_eval[1][ord_id] /= jacobian[point][i, i]^partial_orders[i]
        end
    end

    return partial_der_eval
end
