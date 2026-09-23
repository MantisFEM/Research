############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    CoDifferential{manifold_dim, form_rank, expression_rank, G, F} <:
    AbstractForm{manifold_dim, form_rank, expression_rank, G}

Represents the codifferential of an `AbstractForm`.

# Fields
- `form::AbstractForm{manifold_dim, form_rank, expression_rank, G}`: The form to
    which the codifferential is applied.
- `label::String`: The codifferential label. This is a concatenation of `"d*"` with the
    label of `form`.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: The form rank of the codifferential. If the form rank of `form` is `k`
    then `form_rank` is `k-1`.
- `expression_rank`: Rank of the expression. Expressions without basis forms have rank 0,
    with one single set of basis forms have rank 1, with two sets of basis forms have rank
    2. Higher ranks are not possible.
- `G <: Geometry.AbstractGeometry{manifold_dim}`: Type of the underlying geometry.
- `F <: Forms.AbstractForm{manifold_dim, form_rank+1, expression_rank, G}`: The
    type of `form`.

# Inner Constructors
- `CoDifferential(form::F)`: General constructor.
"""
struct Laplacian{manifold_dim, form_rank, expression_rank, G, F} <:
       AbstractForm{manifold_dim, form_rank, expression_rank, G}
    form::F
    label::String

    function Laplacian(
        form::F
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        F <: AbstractForm{manifold_dim, form_rank, expression_rank, G},
    }
        if form_rank != 0
            throw(ArgumentError("""\
                Tried to compute the laplacian of a $form_rank form. We can only compute \
                the Laplacian of 0-forms. \
                """))
        end

        return new{manifold_dim, form_rank, expression_rank, G, F}(
            form, "Δ(" * get_label(form) * ")"
        )
    end
end

const Δ = Laplacian

get_form(form::Laplacian) = form.form
get_geometry(form::Laplacian) = get_geometry(get_form(form))

function evaluate(
    form::Laplacian{manifold_dim}, element_id::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    return _evaluate_laplacian(get_form(form), element_id, xi)
end

function _evaluate_laplacian(
    form::AbstractForm{manifold_dim}, ::Int, ::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    throw(ArgumentError("Method not implement for type $(typeof(form))."))
end

############################################################################################
#                                        Form Field                                        #
############################################################################################

function _evaluate_laplacian(
    form::FormField{manifold_dim, form_rank, G, FS},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {
    manifold_dim,
    form_rank,
    G <: Geometry.AbstractGeometry{manifold_dim},
    FS <: AbstractFormSpace{manifold_dim, form_rank, G},
}
    d_form_basis_eval, form_basis_indices = _evaluate_laplacian(
        form.form_space, element_id, xi
    )

    # This is equal to binomial(manifold_dim, form_rank + 1).
    n_derivative_components = size(d_form_basis_eval, 1)

    d_form_eval = Vector{Vector{Float64}}(undef, n_derivative_components)

    for derivative_form_component_idx in 1:n_derivative_components
        d_form_eval[derivative_form_component_idx] =
            d_form_basis_eval[derivative_form_component_idx] *
            form.coefficients[form_basis_indices[1]]
    end

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow
    # multi-indexed expressions, like wedges.
    return d_form_eval, [[1]]
end

############################################################################################
#                                        Form Space                                        #
############################################################################################

# Currently only implemented for 2D 0-forms.
function _evaluate_laplacian(
    form_space::FS, element_id::Int, xi::Points.AbstractPoints{2}
) where {G <: Geometry.AbstractGeometry{2}, FS <: AbstractFormSpace{2, 0, G}}
    return _evaluate_codifferential(ExteriorDerivative(form_space), element_id, xi)
end
# function _evaluate_laplacian(
#     form_space::FS, element_id::Int, xi::NTuple{2, AbstractVector{Float64}}
# ) where {G <: Geometry.AbstractGeometry{2}, FS <: AbstractFormSpace{2, 0, G}}
#     manifold_dim = 2
#     n_coderivative_form_components = 1
#     n_basis_functions = Forms.get_num_basis(form_space, element_id)
#     n_evaluation_points = prod(size.(xi, 1))

#     # Preallocate memory for output array
#     codiff_eval = [
#         zeros(Float64, n_evaluation_points, n_basis_functions) for
#         _ in 1:n_coderivative_form_components
#     ]

#     # Evaluate derivatives of the basis functions. We need derivatives up to order 2.
#     # fem_evals, form_basis_indices = _evaluate_form_in_canonical_coordinates(
#     #     form_space, element_id, xi, 2
#     # )
#     fem_evals, form_basis_indices = FunctionSpaces.evaluate(
#         form_space.fem_space, element_id, xi, 2
#     ) # Applied to zero forms, so the pullback to canonical doesn't change anything.

#     # Compute the metric terms, including derivative of the metric.
#     J = Geometry.jacobian(get_geometry(form_space), element_id, xi)
#     inv_g, g, sqrt_g = Geometry.inv_metric(get_geometry(form_space), element_id, xi)
#     H_x, H_y = Geometry.hessian(get_geometry(form_space), element_id, xi)

#     # Adjugate of the metric tensor
#     # adj_g = [LinearAlgebra.det(g[i,:,:]) * inv_g[i,:,:] for i in 1:n_evaluation_points]

#     # Derivative of the metric tensor
#     dgdu = [zeros(Float64, manifold_dim, manifold_dim) for _ in 1:n_evaluation_points]
#     dgdv = [zeros(Float64, manifold_dim, manifold_dim) for _ in 1:n_evaluation_points]
#     for i in eachindex(dgdu, dgdv)
#         # dg11
#         dgdu[i][1,1] = 2.0 * J[i,1,1] * H_x[i][1,1] + 2.0 * J[i,1,2] * H_x[i][1,2]
#         dgdv[i][1,1] = 2.0 * J[i,1,1] * H_x[i][2,1] + 2.0 * J[i,1,2] * H_x[i][2,2]

#         # dg12 and dg21
#         dgdu[i][1,2] = J[i,2,1] * H_x[i][1,1] + J[i,1,1] * H_y[i][1,1] +
#                        J[i,2,2] * H_x[i][1,2] + J[i,1,2] * H_y[i][1,2]
#         dgdu[i][2,1] = dgdu[i][1,2]
#         dgdv[i][1,2] = J[i,2,1] * H_x[i][2,1] + J[i,1,1] * H_y[i][2,1] +
#                        J[i,2,2] * H_x[i][2,2] + J[i,1,2] * H_y[i][2,2]
#         dgdv[i][2,1] = dgdv[i][1,2]

#         # dg22
#         dgdu[i][2,2] = 2.0 * J[i,2,1] * H_y[i][1,1] + 2.0 * J[i,2,2] * H_y[i][1,2]
#         dgdv[i][2,2] = 2.0 * J[i,2,1] * H_y[i][2,1] + 2.0 * J[i,2,2] * H_y[i][2,2]
#     end

#     # Per coordinate direction:
#     # d(inv_g)/du = -inv_g * dg/du * inv_g
#     dinv_g_du = [-inv_g[i,:,:] * dgdu[i][:,:] * inv_g[i,:,:] for i in 1:n_evaluation_points]
#     dinv_g_dv = [-inv_g[i,:,:] * dgdv[i][:,:] * inv_g[i,:,:] for i in 1:n_evaluation_points]
#     # d(sqrt_g)/du = 0.5 * sqrt_g * tr(dg/du * inv_g)
#     dsqrt_g_du = [0.5 * sqrt_g[i] * LinearAlgebra.tr(dgdu[i][:,:] * inv_g[i,:,:]) for i in 1:n_evaluation_points]
#     dsqrt_g_dv = [0.5 * sqrt_g[i] * LinearAlgebra.tr(dgdv[i][:,:] * inv_g[i,:,:]) for i in 1:n_evaluation_points]

#     # Compute the laplacian, which looks like the coderivative of the exterior derivative.
#     # α^1 = α¹ du + α² dv
#     # α¹ = d(β⁰) = ∂ᵤ β⁰ du + ∂ᵥ β⁰ dv
#     idx_du = FunctionSpaces.get_derivative_idx([1, 0])
#     idx_dv = FunctionSpaces.get_derivative_idx([0, 1])
#     idx_duu = FunctionSpaces.get_derivative_idx([2, 0])
#     idx_dvv = FunctionSpaces.get_derivative_idx([0, 2])
#     idx_duv = FunctionSpaces.get_derivative_idx([1, 1])
#     for i in 1:n_evaluation_points
#         codiff_eval[1][i, :] .=
#             fem_evals[3][idx_duu][1][i, :] .* inv_g[i,1,1] .+  # ∂u α¹ * g¹¹
#             fem_evals[2][idx_du][1][i, :] .* dinv_g_du[i][1,1] .+  # α¹ * ∂u g¹¹
#             fem_evals[3][idx_duv][1][i, :] .* inv_g[i,1,2] .+  # ∂u α² * g¹²
#             fem_evals[2][idx_dv][1][i, :] .* dinv_g_du[i][1,2] .+  # α² * ∂u g¹²
#             fem_evals[3][idx_duv][1][i, :] .* inv_g[i,2,1] .+  # ∂v α¹ * g²¹
#             fem_evals[2][idx_du][1][i, :] .* dinv_g_dv[i][2,1] .+  # α¹ * ∂v g²¹
#             fem_evals[3][idx_dvv][1][i, :] .* inv_g[i,2,2] .+  # ∂v α² * g²²
#             fem_evals[2][idx_dv][1][i, :] .* dinv_g_dv[i][2,2] .+  # α² * ∂v g²²
#             (1.0 / sqrt_g[i]) .* (
#                 .+ fem_evals[2][idx_du][1][i, :] .* inv_g[i,1,1] .* dsqrt_g_du[i]  # α¹ * g¹¹ * ∂u sqrt(g)
#                 .+ fem_evals[2][idx_dv][1][i, :] .* inv_g[i,1,2] .* dsqrt_g_du[i]  # α² * g¹² * ∂u sqrt(g)
#                 .+ fem_evals[2][idx_du][1][i, :] .* inv_g[i,2,1] .* dsqrt_g_dv[i]  # α¹ * g²¹ * ∂v sqrt(g)
#                 .+ fem_evals[2][idx_dv][1][i, :] .* inv_g[i,2,2] .* dsqrt_g_dv[i]  # α² * g²² * ∂v sqrt(g)
#             )
#     end

#     return codiff_eval, [form_basis_indices]#form_basis_indices#
# end
