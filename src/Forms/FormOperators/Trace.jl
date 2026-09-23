############################################################################################
#                                        Structure                                         #
############################################################################################

struct Trace{manifold_dim, form_rank, expression_rank, G, F, GT} <:
       AbstractForm{manifold_dim, form_rank, expression_rank, G}
    form::F
    label::String
    trace_geometry::GT
    edge::Int
    c::Float64
    component::Int

    function Trace(
        form::F, trace_geometry::GT, edge::Int, c=0.0, component=1
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        GT,
        G <: Geometry.AbstractGeometry{manifold_dim},
        F <: AbstractForm{manifold_dim, form_rank, expression_rank, G},
    }
        if form_rank == manifold_dim
            throw(ArgumentError("""\
                Tried to compute the trace of a volume form. The manifold \
                dimension is $(manifold_dim) and the form rank is $(form_rank). \
                """))
        end

        return new{manifold_dim - 1, form_rank, expression_rank, G, F, GT}(
            form, "d(" * get_label(form) * ")", trace_geometry, edge, c, component
        )
    end
end

"""
    tr

Symbolic wrapper for the trace operator. See [`Trace`](@ref) for the details.
"""
const tr = Trace

############################################################################################
#                                         Getters                                          #
############################################################################################
get_form(form::Trace) = form.form

function get_form_space_tree(form::Trace)
    return get_form_space_tree(get_form(form))
end

# get_geometry(form::Trace) = get_geometry(get_form(form))
get_geometry(form::Trace) = form.trace_geometry

function get_interior_element(form::Trace, trace_element_id, patch_id)
    # This is from trace element_id to interior element_id.
    elements_per_dim = Int(
        sqrt(
            Geometry.get_num_elements(get_geometry(get_form(form))) /
            Geometry.get_num_patches(get_geometry(get_form(form))),
        ),
    )
    # patch_id, patch_local_element_id = Geometry.get_patch_and_local_element_id(
    #     get_geometry(form), element_id
    # )
    if patch_id == 1
        edge = form.edge
    else
        edge = form.edge + 2
    end
    if edge == 2
        interior_element_id = Int(elements_per_dim * trace_element_id) # should be exact.
    elseif edge == 4
        interior_element_id = Int(elements_per_dim * (trace_element_id - 1)) + 1 # should be exact.
    else
        error("What edge? ", edge)
    end

    # We have to return the global element_id, so that it can be used in an evaluate. So add
    # the number of elements of all previous patches.
    global_element_id_on_patch = interior_element_id
    for i in 1:(patch_id - 1)
        global_element_id_on_patch += Geometry.get_num_elements(
            get_geometry(get_form(form)), i
        )
    end
    # println("element_ids")
    # println(element_id)
    # println(trace_element_id)
    return global_element_id_on_patch
end

"""
    evaluate(
        form::Trace{manifold_dim},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim}

Computes the trace at the element given by `element_id`, and canonical points `xi`.

# Arguments
- `form::Trace{manifold_dim}`: The exterior derivative structure.
- `element_id::Int`: The element identifier.
- `xi::Points.AbstractPoints{manifold_dim}`: The set of canonical points.

# Returns
- `::Vector{Array{Float64, expression_rank + 1}}`: The evaluated exterior derivative. The
    number of entries in the `Vector` is `binomial(manifold_dim, form_rank)`. The size
    of the `Array` is `(num_eval_points, num_basis)`, where `num_eval_points =
    Points.get_num_points(xi)` and `num_basis` is the number of basis functions used to represent
    the `form` on `element_id` ― for `expression_rank = 0` the inner `Array` is equivalent
    to a `Vector`.
"""
function evaluate(
    form::Trace{manifold_dim}, element_id::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    return _evaluate_trace(get_form(form), element_id, xi, form)
end

############################################################################################
#                                     Abstract method                                      #
############################################################################################

function _evaluate_trace(
    form::AbstractForm{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    edge::Int=2,
) where {manifold_dim}
    throw(ArgumentError("Method not implement for type $(typeof(form))."))
end

############################################################################################
#                                        Form Field                                        #
############################################################################################

function _evaluate_trace(
    form::FormField{manifold_dim, form_rank, G, FS},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    edge::Int=2,
) where {
    manifold_dim,
    form_rank,
    G <: Geometry.AbstractGeometry{manifold_dim},
    FS <: AbstractFormSpace{manifold_dim, form_rank, G},
}
    d_form_basis_eval, form_basis_indices = _evaluate_trace(
        get_form_space(form), element_id, xi
    )

    # This is equal to binomial(manifold_dim, form_rank + 1).
    n_derivative_components = size(d_form_basis_eval, 1)

    d_form_eval = Vector{Vector{Float64}}(undef, n_derivative_components)

    for derivative_form_component_idx in 1:n_derivative_components
        d_form_eval[derivative_form_component_idx] =
            d_form_basis_eval[derivative_form_component_idx] *
            form.coefficients[form_basis_indices[1]]
    end

    return d_form_eval, [[1]]
end

############################################################################################
#                                        Form Space                                        #
############################################################################################

function create_interior_xi(xi, manifold_dim_points, manifold_dim, edge)
    if manifold_dim_points == manifold_dim
        xi_interior = xi
    elseif manifold_dim_points == manifold_dim - 1
        # We take the trace of the form, so we have the 1D points but need them in 2D.
        cps = Points.get_constituent_points(xi)
        if edge == 1
            xi_interior = Points.CartesianPoints((cps[1], [0.0]))
        elseif edge == 2
            xi_interior = Points.CartesianPoints(([1.0], cps[1]))
        elseif edge == 3
            xi_interior = Points.CartesianPoints((cps[1], [1.0]))
        elseif edge == 4
            xi_interior = Points.CartesianPoints(([0.0], cps[1]))
        else
            error("Out of edges?")
        end
        # elseif manifold_dim_points == manifold_dim + 1
        #     cps = Points.get_constituent_points(xi)
        #     if edge == 1
        #         xi_interior = Points.CartesianPoints(cps)
        #     end
    else
        error("What happened?")
    end
    return xi_interior
end

function _evaluate_trace(
    form_space::AbstractFormSpace{manifold_dim, form_rank, G},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim_points},
    trace,
) where {manifold_dim, form_rank, G, manifold_dim_points}
    evals1 = zeros(2, 2)
    evals2 = zeros(2, 2)
    inds1 = zeros(Int, 2)
    inds2 = zeros(Int, 2)
    for interior_patch_id in 1:Geometry.get_num_patches(get_geometry(form_space))
        interior_element_id = get_interior_element(trace, element_id, interior_patch_id)
        # interior_patch_id = Geometry.get_patch_and_local_element_id(
        #     get_geometry(form_space), element_id
        # )[1]
        if interior_patch_id == 1
            edge = trace.edge
        else
            edge = trace.edge + 2
        end
        xi_interior = create_interior_xi(xi, manifold_dim_points, manifold_dim, edge)

        trace_eval, form_basis_indices = _evaluate_form_in_canonical_coordinates(
            form_space, interior_element_id, xi_interior, 0
        )

        # v = Geometry.evaluate(
        #     Geometry.get_base_geometry(trace.trace_geometry), element_id, xi
        # )
        J = Geometry.jacobian(
            Forms.get_geometry(form_space), interior_element_id, xi_interior
        )
        inv_g, g, sqrt_g = Geometry.inv_metric(
            Forms.get_geometry(form_space), interior_element_id, xi_interior
        )

        factor = J .* inv_g # == pushforward(sharp())

        # Glueing data
        # alpha = LinearAlgebra.det.(J)
        # tau = [sqrt(J[p][1, 2]^2 + J[p][2, 2]^2) for p in eachindex(J)] # length of tangent vector
        # t0 = [J[p][:, 2] / tau[p] for p in eachindex(J, tau)]
        # beta = [
        #     (J[p][1, 1] * t0[p][1] + J[p][2, 1] * t0[p][2]) / tau[p] for
        #     p in eachindex(J, t0, tau)
        # ]

        # factor .*= (1.0 ./ alpha)

        if interior_patch_id == 1
            # Left patch
            # evals1 = transpose(
            #     reduce(
            #         hcat,
            #         (1.0 ./ alpha[p]) .* trace_eval[1][1][1][p, :] -
            #         (beta[p] ./ alpha[p]) .* trace_eval[1][1][2][p, :] for p in eachindex(J)
            #     ),
            # )
            evals1a = transpose(
                reduce(
                    hcat,
                    factor[p][trace.component, 1] .* trace_eval[1][1][1][p, :] for
                    p in eachindex(J)
                ),
            )
            evals1b = transpose(
                reduce(
                    hcat,
                    factor[p][trace.component, 2] .* trace_eval[1][1][2][p, :] for
                    p in eachindex(J)
                ),
            )
            evals1 = hcat(evals1a, evals1b)
            # println("evals1")
            # display(evals1)
            # inds1 = form_basis_indices[1]
            inds1 = vcat(form_basis_indices[1], form_basis_indices[1])
            # println("inds1")
            # display(inds1)
        elseif interior_patch_id == 2
            # Right patch
            # evals2 = transpose(
            #     reduce(
            #         hcat,
            #         -(1.0 ./ alpha[p]) .* trace_eval[1][1][1][p, :] +
            #         (beta[p] ./ alpha[p]) .* trace_eval[1][1][2][p, :] for p in eachindex(J)
            #     ),
            # )
            evals2a = transpose(
                reduce(
                    hcat,
                    -factor[p][trace.component, 1] .* trace_eval[1][1][1][p, :] for
                    p in eachindex(J)
                ),
            )
            evals2b = transpose(
                reduce(
                    hcat,
                    -factor[p][trace.component, 2] .* trace_eval[1][1][2][p, :] for
                    p in eachindex(J)
                ),
            )
            evals2 = hcat(evals2a, evals2b)
            # println("evals2")
            # display(evals2)
            # inds2 = form_basis_indices[1]
            inds2 = vcat(form_basis_indices[1], form_basis_indices[1])
            # println("inds2")
            # display(inds2)
        else
            error("How many patches are there?")
        end
    end
    # println("evals12")
    # display([hcat(evals1, evals2)][1])
    # println("inds12")
    # display([vcat(inds1, inds2)][1])
    return [hcat(evals1, evals2)], [vcat(inds1, inds2)]
end

# function _evaluate_trace(
#     form_space::AbstractFormSpace{manifold_dim, form_rank, G},
#     element_id::Int,
#     xi::Points.AbstractPoints{manifold_dim_points},
#     trace,
# ) where {manifold_dim, form_rank, G, manifold_dim_points}
#     evals1 = zeros(2, 2)
#     evals2 = zeros(2, 2)
#     inds1 = zeros(Int, 2)
#     inds2 = zeros(Int, 2)
#     for interior_patch_id in 1:Geometry.get_num_patches(get_geometry(form_space))
#         interior_element_id = get_interior_element(trace, element_id, interior_patch_id)

#         if interior_patch_id == 1
#             edge = trace.edge
#         else
#             edge = trace.edge + 2
#         end
#         xi_interior = create_interior_xi(xi, manifold_dim_points, manifold_dim, edge)

#         trace_eval, form_basis_indices = _evaluate_form_in_canonical_coordinates(
#             form_space, interior_element_id, xi_interior, 0
#         )

#         if interior_patch_id == 1
#             # Left patch
#             evals1 = trace_eval[1][1][1]
#             inds1 = form_basis_indices[1]
#         elseif interior_patch_id == 2
#             # Right patch
#             evals2 = trace_eval[1][1][1]
#             inds2 = form_basis_indices[1]
#         else
#             error("How many patches are there?")
#         end
#     end
#     return [hcat(evals1, evals2)], [vcat(inds1, inds2)]
# end
