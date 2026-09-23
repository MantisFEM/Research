function _L2_norm_square(u, element_id, dΩ)
    integral = ∫(u ∧ ★(u), dΩ)

    return Forms.evaluate(integral, element_id)[1][1]
end

function L2_norm(u, dΩ)
    norm = 0.0
    inner_prod = ∫(u ∧ ★(u), dΩ)
    for el_id in 1:Forms.get_num_elements(u)
        norm += Forms.evaluate(inner_prod, el_id)[1][1]
    end

    return sqrt(norm)
end

function Linf_norm(u, dΩ)
    norm = 0.0
    for el_id in 1:Forms.get_num_elements(u)
        norm = max(
            norm,
            maximum(
                abs.(
                    Forms.evaluate(
                        u,
                        el_id,
                        Quadrature.get_nodes(
                            Quadrature.get_element_quadrature_rule(dΩ, el_id)
                        ),
                    )[1][1]
                ),
            ),
        )
    end

    return norm
end

function _compute_square_error_per_element(
    computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2"
) where {
    manifold_dim,
    form_rank,
    expression_rank_1,
    expression_rank_2,
    G <: Geometry.AbstractGeometry{manifold_dim},
    TF1 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_1, G},
    TF2 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_2, G},
    Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
}
    num_elements = Quadrature.get_num_base_elements(quad_rule)
    result = Vector{Float64}(undef, num_elements)

    for elem_id in 1:1:num_elements
        difference = computed_sol - exact_sol
        if norm == "L2"
            result[elem_id] = _L2_norm_square(difference, elem_id, quad_rule)
        elseif norm == "H1"
            throw(ArgumentError("Computing the H1 norm still needs to be updated."))
            # d_difference = Forms.ExteriorDerivative(difference)
            # result[elem_id] = sum(
            #     Forms.evaluate_inner_product(
            #         d_difference, d_difference, elem_id, quad_rule
            #     )[3],
            # )
        elseif norm == "Linf"
            result[elem_id] = maximum(
                abs.(
                    Forms.evaluate(difference, elem_id, Quadrature.get_nodes(quad_rule))[1][1]
                ),
            )
        else
            throw(
                ArgumentError(
                    "Unknown norm '$norm'. Only 'L2', 'Linf', and 'H1' are accepted inputs."
                ),
            )
        end
    end

    return result
end

function compute_error_per_element(
    computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2"
) where {
    manifold_dim,
    form_rank,
    expression_rank_1,
    expression_rank_2,
    G <: Geometry.AbstractGeometry{manifold_dim},
    TF1 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_1, G},
    TF2 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_2, G},
    Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
}
    partial_result = _compute_square_error_per_element(
        computed_sol, exact_sol, quad_rule, norm
    )
    if norm == "Linf"
        return partial_result
    elseif norm == "L2" || norm == "H1"
        return sqrt.(partial_result)
    else
        throw(
            ArgumentError(
                "Unknown norm '$norm'. Only 'L2', 'Linf', and 'H1' are accepted inputs."
            ),
        )
    end
end

function compute_error_total(
    computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2"
) where {
    manifold_dim,
    form_rank,
    expression_rank_1,
    expression_rank_2,
    G <: Geometry.AbstractGeometry{manifold_dim},
    TF1 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_1, G},
    TF2 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_2, G},
    Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
}
    partial_result = _compute_square_error_per_element(
        computed_sol, exact_sol, quad_rule, norm
    )
    if norm == "Linf"
        return maximum(partial_result)
    elseif norm == "L2" || norm == "H1"
        return sqrt(sum(partial_result))
    else
        throw(
            ArgumentError(
                "Unknown norm '$norm'. Only 'L2', 'Linf', and 'H1' are accepted inputs."
            ),
        )
    end
end

function compute_max_jump(α⁰)#, solve_patches=nothing)
    geom = Forms.get_geometry(α⁰)
    n_points_per_element = 25
    npatches = FunctionSpaces.get_num_patches(α⁰.form_space.fem_space)
    elems_per_dim = Int(sqrt(Geometry.get_num_elements(geom) / npatches))

    el_id_local = [0, 0]
    trace_all = (
        zeros(elems_per_dim * n_points_per_element),
        zeros(elems_per_dim * n_points_per_element),
    )
    trace_nor_all = (
        zeros(elems_per_dim * n_points_per_element),
        zeros(elems_per_dim * n_points_per_element),
    )

    for global_element_id in 1:1:Geometry.get_num_elements(geom)
        patch_id = FunctionSpaces.get_patch_id(α⁰.form_space.fem_space, global_element_id)

        element_id = global_element_id - (patch_id - 1) * (elems_per_dim^2)
        if npatches == 2 && (
            (element_id % elems_per_dim == 0 && patch_id == 1) ||
            (element_id % elems_per_dim == 1 && patch_id == 2)
        )
            # Left and right boundaries.
            # Compute trace and normal trace.

            # if !isnothing(solve_patches)
            #     sol = zeros(length(α⁰.coefficients))
            #     for (global_dof, local_dof_dict) in pairs(α⁰.form_space.fem_space[1].global_to_local_dof_dict)
            #         for (patch_i, local_dof) in pairs(local_dof_dict)

            #             if length(local_dof_dict) > 1
            #                 if patch_id == 1
            #                     if patch_i == 1
            #                         sol[global_dof] = solve_patches[1][local_dof]
            #                     end
            #                 else
            #                     if patch_i == 2
            #                         sol[global_dof] = solve_patches[2][local_dof]
            #                     end
            #                 end
            #             else
            #                 sol[global_dof] = solve_patches[patch_i][local_dof]
            #             end
            #         end
            #     end
            #     α⁰.coefficients .= sol
            # end

            el_id_local[patch_id] += 1

            y = LinRange(0.0, 1.0, n_points_per_element)

            if patch_id == 1
                # trace_eval_points = ([1.0], y)
                trace_eval_points = Points.CartesianPoints((LinRange(1.0, 1.0, 1), y))
            else
                # trace_eval_points = ([0.0], y)
                trace_eval_points = Points.CartesianPoints((LinRange(0.0, 0.0, 1), y))
            end

            jac = Geometry.jacobian(
                Forms.get_geometry(α⁰), global_element_id, trace_eval_points
            )
            t = [jac[p] * [0.0, 1.0] for p in eachindex(jac)]
            t_hat = [t[p] / sqrt(t[p][1]^2 + t[p][2]^2) for p in eachindex(t)]
            n_hat = [[t_hat[p][2], -t_hat[p][1]] for p in eachindex(t_hat)]

            # No need to flip the sign, since we are interested in the jump on the
            # interface of the function. Flipping the normal would result in + on one side
            # and - on the other side, which is not the jump.
            # if patch_id == 2
            #     n_hat = -n_hat
            # end

            eval_tr = Forms.evaluate(α⁰, global_element_id, trace_eval_points)[1][1]
            eval_tr_nor = hcat(
                reduce.(
                    +,
                    Forms.evaluate_sharp_pushforward(
                        Forms.d(α⁰), global_element_id, trace_eval_points
                    )[1],
                    dims=2,
                )...,
            )
            eval_tr_nor = [
                eval_tr_nor[p, 1] .* n_hat[p][1] .+ eval_tr_nor[p, 2] .* n_hat[p][2] for
                p in axes(eval_tr_nor, 1)
            ]

            trace_all[patch_id][((el_id_local[patch_id] - 1) * n_points_per_element + 1):(el_id_local[patch_id] * n_points_per_element)] .=
                eval_tr

            trace_nor_all[patch_id][((el_id_local[patch_id] - 1) * n_points_per_element + 1):(el_id_local[patch_id] * n_points_per_element)] .=
                eval_tr_nor
        end
    end

    l_inf_trace = maximum(abs.(trace_all[1] .- trace_all[2]))
    l_inf_trace_nor = maximum(abs.(trace_nor_all[1] .- trace_nor_all[2]))

    # return l_inf_trace, l_inf_trace_nor
    return l_inf_trace_nor
end

function compute_max_jump_1form(α⁰)
    geom = Forms.get_geometry(α⁰)
    n_points_per_element = 25
    npatches = FunctionSpaces.get_num_patches(Forms.get_fe_space(α⁰))
    elems_per_dim = Int(sqrt(Geometry.get_num_elements(geom) / npatches))

    if npatches == 2
        el_id_local = [0, 0]
        trace_all = (
            zeros(elems_per_dim * n_points_per_element),
            zeros(elems_per_dim * n_points_per_element),
        )
        trace_all_nor = (
            zeros(elems_per_dim * n_points_per_element),
            zeros(elems_per_dim * n_points_per_element),
        )
    elseif npatches == 4
        # left/right and top/bottom per patch
        temp = (
            zeros(elems_per_dim * n_points_per_element),
            zeros(elems_per_dim * n_points_per_element),
        )
        trace_all = (deepcopy(temp), deepcopy(temp), deepcopy(temp), deepcopy(temp))
        trace_all_nor = (deepcopy(temp), deepcopy(temp), deepcopy(temp), deepcopy(temp))
        el_id_local = [deepcopy([0, 0]) for i in 1:1:npatches]
    elseif npatches == 5
        # left/right and top/bottom per patch
        temp = (
            zeros(elems_per_dim * n_points_per_element),
            zeros(elems_per_dim * n_points_per_element),
        )
        trace_all = (
            deepcopy(temp), deepcopy(temp), deepcopy(temp), deepcopy(temp), deepcopy(temp)
        )
        trace_all_nor = (
            deepcopy(temp), deepcopy(temp), deepcopy(temp), deepcopy(temp), deepcopy(temp)
        )
        el_id_local = [deepcopy([0, 0]) for i in 1:1:npatches]
    end

    for global_element_id in 1:Geometry.get_num_elements(geom)
        patch_id = FunctionSpaces.get_patch_id(Forms.get_fe_space(α⁰), global_element_id)

        element_id = global_element_id - (patch_id - 1) * (elems_per_dim^2)
        if npatches == 2 && (
            (element_id % elems_per_dim == 0 && patch_id == 1) ||
            (element_id % elems_per_dim == 1 && patch_id == 2)
        )
            # Left and right boundaries.
            el_id_local[patch_id] += 1

            y = LinRange(0.0, 1.0, n_points_per_element)

            if patch_id == 1
                trace_eval_points = Points.CartesianPoints((LinRange(1.0, 1.0, 1), y))
            else
                trace_eval_points = Points.CartesianPoints((LinRange(0.0, 0.0, 1), y))
            end

            jac = Geometry.jacobian(
                Forms.get_geometry(α⁰), global_element_id, trace_eval_points
            )
            t = [jac[p] * [0.0, 1.0] for p in eachindex(jac)]
            t_hat = [t[p] / sqrt(t[p][1]^2 + t[p][2]^2) for p in eachindex(t)]
            n_hat = [[t_hat[p][2], -t_hat[p][1]] for p in eachindex(t_hat)]

            # No need to flip the sign, since we are interested in the jump on the
            # interface of the function. Flipping the normal would result in + on one side
            # and - on the other side, which is not the jump.
            # if patch_id == 2
            #     n_hat = -n_hat
            # end
            eval_tr = hcat(
                reduce.(
                    +,
                    Forms.evaluate_sharp_pushforward(
                        α⁰, global_element_id, trace_eval_points
                    )[1],
                    dims=2,
                )...,
            )
            eval_tr_tan = [
                eval_tr[p, 1] .* t_hat[p][1] .+ eval_tr[p, 2] .* t_hat[p][2] for
                p in axes(eval_tr, 1)
            ]
            eval_tr_nor = [
                eval_tr[p, 1] .* n_hat[p][1] .+ eval_tr[p, 2] .* n_hat[p][2] for
                p in axes(eval_tr, 1)
            ]

            trace_all[patch_id][((el_id_local[patch_id] - 1) * n_points_per_element + 1):(el_id_local[patch_id] * n_points_per_element)] .=
                eval_tr_tan
            trace_all_nor[patch_id][((el_id_local[patch_id] - 1) * n_points_per_element + 1):(el_id_local[patch_id] * n_points_per_element)] .=
                eval_tr_nor
        end

        if npatches == 4 && (
            (element_id % elems_per_dim == 0 && (patch_id == 1 || patch_id == 4)) ||
            (element_id % elems_per_dim == 1 && (patch_id == 2 || patch_id == 3))
        )
            # Left and right boundaries.

            el_id_local[patch_id][1] += 1

            y = LinRange(0.0, 1.0, n_points_per_element)

            if patch_id == 1 || patch_id == 4
                trace_eval_points = Points.CartesianPoints((LinRange(1.0, 1.0, 1), y))
            else
                trace_eval_points = Points.CartesianPoints((LinRange(0.0, 0.0, 1), y))
            end

            jac = Geometry.jacobian(
                Forms.get_geometry(α⁰), global_element_id, trace_eval_points
            )
            t = [jac[p] * [0.0, 1.0] for p in eachindex(jac)]
            t_hat = [t[p] / sqrt(t[p][1]^2 + t[p][2]^2) for p in eachindex(t)]
            n_hat = [[t_hat[p][2], -t_hat[p][1]] for p in eachindex(t_hat)]

            # No need to flip the sign, since we are interested in the jump on the
            # interface of the function. Flipping the normal would result in + on one side
            # and - on the other side, which is not the jump.
            # if patch_id == 2
            #     n_hat = -n_hat
            # end
            eval_tr = hcat(
                reduce.(
                    +,
                    Forms.evaluate_sharp_pushforward(
                        α⁰, global_element_id, trace_eval_points
                    )[1],
                    dims=2,
                )...,
            )
            eval_tr_nor = [
                eval_tr[p, 1] .* n_hat[p][1] .+ eval_tr[p, 2] .* n_hat[p][2] for
                p in axes(eval_tr, 1)
            ]

            trace_all_nor[patch_id][1][((el_id_local[patch_id][1] - 1) * n_points_per_element + 1):(el_id_local[patch_id][1] * n_points_per_element)] .=
                eval_tr_nor
        end

        if npatches == 4 && (
            (element_id <= elems_per_dim && (patch_id == 3 || patch_id == 4)) || (
                element_id > elems_per_dim^2 - elems_per_dim &&
                (patch_id == 1 || patch_id == 2)
            )
        )
            # Top and bottom boundaries.
            el_id_local[patch_id][2] += 1

            x = LinRange(0.0, 1.0, n_points_per_element)

            # The forms are defined on the whole of the domain, while their
            # traces are only on the boundary. We 'fake' the trace by only
            # evaluating the form on the boundary.
            if patch_id == 3 || patch_id == 4
                trace_eval_points = Points.CartesianPoints((x, LinRange(0.0, 0.0, 1)))
            else
                trace_eval_points = Points.CartesianPoints((x, LinRange(1.0, 1.0, 1)))
            end

            jac = Geometry.jacobian(
                Forms.get_geometry(α⁰), global_element_id, trace_eval_points
            )
            t = [jac[p] * [1.0, 0.0] for p in eachindex(jac)]
            t_hat = [t[p] / sqrt(t[p][1]^2 + t[p][2]^2) for p in eachindex(t)]
            n_hat = [[t_hat[p][2], -t_hat[p][1]] for p in eachindex(t_hat)]

            # No need to flip the sign, since we are interested in the jump on the
            # interface of the function. Flipping the normal would result in + on one side
            # and - on the other side, which is not the jump.
            # if patch_id == 2
            #     n_hat = -n_hat
            # end
            eval_tr = hcat(
                reduce.(
                    +,
                    Forms.evaluate_sharp_pushforward(
                        α⁰, global_element_id, trace_eval_points
                    )[1],
                    dims=2,
                )...,
            )
            eval_tr_nor = [
                eval_tr[p, 1] .* n_hat[p][1] .+ eval_tr[p, 2] .* n_hat[p][2] for
                p in axes(eval_tr, 1)
            ]

            trace_all_nor[patch_id][2][((el_id_local[patch_id][2] - 1) * n_points_per_element + 1):(el_id_local[patch_id][2] * n_points_per_element)] .=
                eval_tr_nor
        end
    end

    if npatches == 2
        l_inf_trace = maximum(abs.(trace_all[1] .- trace_all[2]))
        # println("\tL^∞ error of θ tangent: ", l_inf_trace)
        l_inf_trace_nor = maximum(abs.(trace_all_nor[1] .- trace_all_nor[2]))
    else
        l_inf_trace_nor = max(
            maximum(abs.(trace_all_nor[1][1] .- trace_all_nor[2][1])),
            maximum(abs.(trace_all_nor[4][1] .- trace_all_nor[3][1])),
            maximum(abs.(trace_all_nor[1][2] .- trace_all_nor[4][2])),
            maximum(abs.(trace_all_nor[2][2] .- trace_all_nor[3][2])),
        )
    end
    return l_inf_trace_nor
end

function edge_to_local_elements(edge_id, elements_per_dim)
    if edge_id == 1
        return 1:1:elements_per_dim
    elseif edge_id == 2
        return elements_per_dim:elements_per_dim:(elements_per_dim^2)
    elseif edge_id == 3
        return ((elements_per_dim^2) - elements_per_dim + 1):1:(elements_per_dim^2)
    elseif edge_id == 4
        return 1:elements_per_dim:((elements_per_dim^2) - elements_per_dim + 1)
    else
        error("panic.")
    end
end

function edge_to_tangent(edge_id)
    if edge_id == 1 || edge_id == 3
        return [1.0, 0.0]
    elseif edge_id == 2 || edge_id == 4
        return [0.0, 1.0]
    else
        error("panic.")
    end
end

function edge_to_xi(edge_id, num_points_per_element)
    if edge_id == 1
        return Points.CartesianPoints((
            LinRange(0.0, 1.0, num_points_per_element), LinRange(0.0, 0.0, 1)
        ))
    elseif edge_id == 2
        return Points.CartesianPoints((
            LinRange(1.0, 1.0, 1), LinRange(0.0, 1.0, num_points_per_element)
        ))
    elseif edge_id == 3
        return Points.CartesianPoints((
            LinRange(0.0, 1.0, num_points_per_element), LinRange(1.0, 1.0, 1)
        ))
    elseif edge_id == 4
        return Points.CartesianPoints((
            LinRange(0.0, 0.0, 1), LinRange(0.0, 1.0, num_points_per_element)
        ))
    else
        error("panic.")
    end
end

function compute_max_jump_1form(α⁰, connectivity)
    geom = Forms.get_geometry(α⁰)
    n_points_per_element = 25
    npatches = length(connectivity)
    elems_per_dim = Int(sqrt(Geometry.get_num_elements(geom) / npatches))

    max_jump = 0.0
    for patch_id in eachindex(connectivity)
        # Loop over all patches, always process the jump for low to higher number.
        for edge in eachindex(connectivity[patch_id])
            neighbour_patch_id, neighbour_edge = connectivity[patch_id][edge]
            if neighbour_patch_id != 0 && # Skip the edge if it has no neighbour
                neighbour_patch_id > patch_id # and only process neighbours with a larger patch_id
                # Loop over all elements on this edge, on both patches. The element ids that
                # we need depend on the edge_id that we're on.
                for (element_id, neighbour_element_id) in zip(
                    edge_to_local_elements(edge, elems_per_dim),
                    edge_to_local_elements(neighbour_edge, elems_per_dim),
                )
                    global_element_id = element_id
                    for i in 1:(patch_id - 1)
                        global_element_id += Geometry.get_num_elements(geom, i)
                    end
                    global_neighbour_element_id = neighbour_element_id
                    for i in 1:(neighbour_patch_id - 1)
                        global_neighbour_element_id += Geometry.get_num_elements(geom, i)
                    end
                    jac = Geometry.jacobian(
                        Forms.get_geometry(α⁰),
                        global_element_id,
                        edge_to_xi(edge, n_points_per_element),
                    )
                    t = [jac[p] * edge_to_tangent(edge) for p in eachindex(jac)]
                    t_hat = [t[p] / sqrt(t[p][1]^2 + t[p][2]^2) for p in eachindex(t)]
                    n_hat = [[t_hat[p][2], -t_hat[p][1]] for p in eachindex(t_hat)]
                    eval_tr = hcat(
                        reduce.(
                            +,
                            Forms.evaluate_sharp_pushforward(
                                α⁰,
                                global_element_id,
                                edge_to_xi(edge, n_points_per_element),
                            )[1],
                            dims=2,
                        )...,
                    )
                    # eval_tr_tan = [
                    #     eval_tr[p, 1] .* t_hat[p][1] .+ eval_tr[p, 2] .* t_hat[p][2] for
                    #     p in axes(eval_tr, 1)
                    # ]
                    eval_tr_nor = [
                        eval_tr[p, 1] .* n_hat[p][1] .+ eval_tr[p, 2] .* n_hat[p][2] for
                        p in axes(eval_tr, 1)
                    ]

                    jac = Geometry.jacobian(
                        Forms.get_geometry(α⁰),
                        global_neighbour_element_id,
                        edge_to_xi(neighbour_edge, n_points_per_element),
                    )
                    t = [jac[p] * edge_to_tangent(neighbour_edge) for p in eachindex(jac)]
                    t_hat = [t[p] / sqrt(t[p][1]^2 + t[p][2]^2) for p in eachindex(t)]
                    n_hat = [[t_hat[p][2], -t_hat[p][1]] for p in eachindex(t_hat)]
                    eval_tr_neighbour = hcat(
                        reduce.(
                            +,
                            Forms.evaluate_sharp_pushforward(
                                α⁰,
                                global_neighbour_element_id,
                                edge_to_xi(neighbour_edge, n_points_per_element),
                            )[1],
                            dims=2,
                        )...,
                    )
                    # eval_tr_neighbour_tan = [
                    #     eval_tr_neighbour[p, 1] .* t_hat[p][1] .+
                    #     eval_tr_neighbour[p, 2] .* t_hat[p][2] for
                    #     p in axes(eval_tr_neighbour, 1)
                    # ]
                    eval_tr_neighbour_nor = [
                        eval_tr_neighbour[p, 1] .* n_hat[p][1] .+
                        eval_tr_neighbour[p, 2] .* n_hat[p][2] for
                        p in axes(eval_tr_neighbour, 1)
                    ]

                    local_max_jump = maximum(abs.(eval_tr_nor .- eval_tr_neighbour_nor))

                    max_jump = max(max_jump, local_max_jump)
                end
            end
        end
    end

    return max_jump
end
