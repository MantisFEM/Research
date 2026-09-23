import LinearSolve as LS

function dirichlet_bc_indices_1_form_TH(space, clamped=false, stokes=false)
    if FunctionSpaces.get_num_patches(space.fem_space) == 1
        if clamped
            # Directly access the underlying TP space (in the DS space) to get the dof partition
            # which is easier than dealing with the additional indices from the DS space.
            if !stokes
                tangential_indices_1 = reduce(
                    vcat,
                    space.fem_space.component_spaces[1].dof_partition[1][[
                        1, 2, 3, 4, 6, 7, 8, 9
                    ]],
                )
                tangential_indices_2 = reduce(
                    vcat,
                    space.fem_space.component_spaces[2].dof_partition[1][[
                        1, 2, 3, 4, 6, 7, 8, 9
                    ]],
                )
            else
                tangential_indices_1 = reduce(
                    vcat, space.fem_space.component_spaces[1].dof_partition[1][[1, 2, 3, 4, 6, 7, 8, 9]]
                )
                tangential_indices_2 = reduce(
                    vcat, space.fem_space.component_spaces[2].dof_partition[1][[2, 4, 6, 8]]
                )
                stokes_dofs = vcat(
                    tangential_indices_1,
                    tangential_indices_2 .+
                    FunctionSpaces.get_num_basis(space.fem_space.component_spaces[1]),
                )
            end
        else
            # Directly access the underlying TP space (in the DS space) to get the dof partition
            # which is easier than dealing with the additional indices from the DS space.
            tangential_indices_1 = reduce(
                vcat,
                space.fem_space.component_spaces[1].dof_partition[1][[1, 2, 3, 7, 8, 9]],
            )
            tangential_indices_2 = reduce(
                vcat,
                space.fem_space.component_spaces[2].dof_partition[1][[1, 4, 7, 3, 6, 9]],
            )
        end
    elseif FunctionSpaces.get_num_patches(space.fem_space) == 2
        if clamped
            # Directly access the underlying MP C0 (!) space (in the DS space) to get the dof
            # partition which is easier than dealing with the additional indices from the DS
            # space.
            tangential_indices_1 = reduce(
                vcat,
                [
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[1][[
                            1, 2, 3, 4, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[2][[
                            1, 2, 3, 6, 7, 8, 9
                        ]],
                    ),
                ],
            )
            tangential_indices_2 = reduce(
                vcat,
                [
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[1][[
                            1, 2, 3, 4, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[2][[
                            1, 2, 3, 6, 7, 8, 9
                        ]],
                    ),
                ],
            )
        else
            # Directly access the underlying MP C0 (!) space (in the DS space) to get the dof
            # partition which is easier than dealing with the additional indices from the DS
            # space.
            tangential_indices_1 = reduce(
                vcat,
                [
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[1][[
                            1, 2, 3, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[2][[
                            1, 2, 3, 7, 8, 9
                        ]],
                    ),
                ],
            )
            tangential_indices_2 = reduce(
                vcat,
                [
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[1][[1, 4, 7]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[2][[3, 6, 9]],
                    ),
                ],
            )
        end
    elseif FunctionSpaces.get_num_patches(space.fem_space) == 3
        # Directly access the underlying MP C0 (!) space (in the DS space) to get the dof
        # partition which is easier than dealing with the additional indices from the DS
        # space.
        tangential_indices_1 = reduce(
            vcat,
            [
                reduce(
                    vcat,
                    space.fem_space.component_spaces[1].dof_partition[1][[
                        1, 2, 3, 7, 8, 9
                    ]],
                ),
                reduce(
                    vcat,
                    space.fem_space.component_spaces[1].dof_partition[2][[
                        1, 2, 3, 7, 8, 9
                    ]],
                ),
                reduce(
                    vcat,
                    space.fem_space.component_spaces[1].dof_partition[3][[
                        1, 2, 3, 7, 8, 9
                    ]],
                ),
            ],
        )
        tangential_indices_2 = reduce(
            vcat,
            [
                reduce(
                    vcat, space.fem_space.component_spaces[2].dof_partition[1][[1, 4, 7]]
                ),
                reduce(
                    vcat, space.fem_space.component_spaces[2].dof_partition[3][[3, 6, 9]]
                ),
            ],
        )
    elseif FunctionSpaces.get_num_patches(space.fem_space) == 4
        # Directly access the underlying MP C0 (!) space (in the DS space) to get the dof
        # partition which is easier than dealing with the additional indices from the DS
        # space.
        if clamped
            tangential_indices_1 = reduce(
                vcat,
                [
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[1][[
                            1, 2, 3, 4, 7
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[2][[
                            1, 2, 3, 6, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[3][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[4][[
                            1, 4, 7, 8, 9
                        ]],
                    ),
                ],
            )
            tangential_indices_2 = reduce(
                vcat,
                [
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[1][[
                            1, 2, 3, 4, 7
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[2][[
                            1, 2, 3, 6, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[3][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[4][[
                            1, 4, 7, 8, 9
                        ]],
                    ),
                ],
            )
        else
            error("Not implemented")
        end
    elseif FunctionSpaces.get_num_patches(space.fem_space) == 5
        # Directly access the underlying MP C0 (!) space (in the DS space) to get the dof
        # partition which is easier than dealing with the additional indices from the DS
        # space.
        if clamped
            tangential_indices_1 = reduce(
                vcat,
                [
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[1][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[2][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[3][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[4][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[1].dof_partition[5][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                ],
            )
            tangential_indices_2 = reduce(
                vcat,
                [
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[1][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[2][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[3][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[4][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                    reduce(
                        vcat,
                        space.fem_space.component_spaces[2].dof_partition[5][[
                            3, 6, 7, 8, 9
                        ]],
                    ),
                ],
            )
        else
            error("Not implemented")
        end
    else
        error(LazyString("Not implemented for more than 5 patches."))
    end

    if stokes
        return stokes_dofs
    end

    bc_basis_indices = vcat(
        tangential_indices_1,
        tangential_indices_2 .+
        FunctionSpaces.get_num_basis(space.fem_space.component_spaces[1]),
    )

    return vcat(bc_basis_indices)
end

function solve_Ainsworth_Parker_Taylor_Hood(
    Wp_H1::Forms.AbstractFormSpace{2, 0, G},
    Theta_space::Forms.AbstractFormSpace{2, 1, G},
    q_space::Forms.AbstractFormSpace{2, 2, G},
    verbose::Bool,
    dΩ,
    f⁰,
    zp_space=Wp_H1,
    save=false,
    dΩ2=nothing;
    clamped=false,
    bc_z_coeffs=nothing,
    bc_w_coeffs=nothing,
    bc_theta=nothing,
    stokes=false,
) where {G}
    # Step 1: Solve for zₚ from
    zp = step1(zp_space, f⁰, dΩ, verbose, bc_z_coeffs, dΩ2)

    # Step 2: Solve for θₚ₋₁ and rₚ₋₂ from
    # (div ψ, div θ) + (rot ψ, rₚ₋₂) = (ψ, grad zₚ) ∀ ψ ∈ Gp-1Gamma
    #                   (rot θₚ₋₁, s) = 0 ∀ s ∈ rot(Gp-1Gamma)
    my_one = Forms.ConstantFormSpace(Val(2), Forms.get_geometry(Wp_H1), "μ")

    # Tangential zero boundary conditions for Gp-1Gamma1 and Gp-1Gamma2
    bc_basis_indices = dirichlet_bc_indices_1_form_TH(Theta_space, clamped, stokes)
    if isnothing(bc_theta)
        bc_G = Dict(i => 0.0 for i in bc_basis_indices)
    else
        bc_G = Dict(i => bc_theta[i] for i in bc_basis_indices)
    end

    # solve for coefficients of solution
    if verbose
        println(
            "Solving for θ and q (num theta dofs = $(Forms.get_num_basis(Theta_space)), num q dofs = $(Forms.get_num_basis(q_space)))...",
        )
    end
    weak_form_inputs_theta = Assemblers.WeakFormInputs(
        (Theta_space, q_space, my_one), (zp,)
    )

    lhs_expressions_theta, rhs_expressions_theta = stokes_like_sys_stab(
        weak_form_inputs_theta, dΩ
    )
    weak_form_theta = Assemblers.WeakForm(
        lhs_expressions_theta, rhs_expressions_theta, weak_form_inputs_theta
    )

    # assemble all matrices
    A_theta, b_theta = Assemblers.assemble(weak_form_theta, bc_G; rhs_type=Vector{Float64})
    if verbose
        println("size(A_theta)=$(size(A_theta)).")
        if size(A_theta, 1) < 4000
            println("\ncond(A_theta)=$(LinearAlgebra.cond(Matrix(A_theta))).")
            println("rank(A_theta)=$(LinearAlgebra.rank(Matrix(A_theta))).")
        end
    end
    prob = LS.LinearProblem(A_theta, b_theta)
    linsolve = LS.init(prob, LS.QRFactorization())
    LS.solve!(linsolve)
    sol_theta_r = linsolve.u

    if verbose
        println("Done solving for θ and q.")
    end
    # create the form field from the solution coefficients
    θ, q, μ = Forms.build_form_fields((Theta_space, q_space, my_one), sol_theta_r)

    # Step 3: Solve for w
    w = step3(Wp_H1, θ, dΩ, verbose, bc_w_coeffs)

    return w, θ, zp, q
end

function zero_form_hodge_laplacian_min(
    inputs::Assemblers.AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    v⁰ = Assemblers.get_test_form(inputs)
    u⁰ = Assemblers.get_trial_form(inputs)
    f⁰ = Assemblers.get_forcing(inputs)
    A = ∫(d(v⁰) ∧ ★(d(u⁰)), dΩ)
    lhs_expression = ((A,),)
    b = -∫(v⁰ ∧ ★(f⁰), dΩ)
    rhs_expression = ((b,),)

    return lhs_expression, rhs_expression
end

function curl_free_theta(
    potential_space::Forms.AbstractFormSpace{2, 0, G},
    Theta_sol::Forms.AbstractFormField{2, 1, G},
    verbose::Bool,
    dΩ,
    save=false,
    tag="TaylorHood",
) where {G <: Geometry.AbstractGeometry{2}}
    # theta_space = Forms.get_form_space(Theta_sol)
    # # Step 1: Solve for zₚ from
    # # (grad zₚ, grad v) = (f, vₚ) for all v ∈ Wp_H1
    # my_one = Forms.ConstantFormSpace(Val(0), Forms.get_geometry(Theta_sol), "μ")

    weak_form_inputs_cft = Assemblers.WeakFormInputs((potential_space,), (δ(Theta_sol),))

    lhs_expressions_cft, rhs_expressions_cft = zero_form_hodge_laplacian_min(
        weak_form_inputs_cft, dΩ
    )
    bc_dirichlet_inds = dirichlet_bc_indices_0_form(potential_space)
    bc_dirichlet = Dict(i => 0.0 for i in bc_dirichlet_inds)

    weak_form_cft = Assemblers.WeakForm(
        lhs_expressions_cft, rhs_expressions_cft, weak_form_inputs_cft
    )

    # assemble all matrices
    A_cft, b_cft = Assemblers.assemble(weak_form_cft, bc_dirichlet)
    # solve for coefficients of solution
    if verbose
        println("Solving for cft...")
    end
    sol_cft = vec(A_cft \ b_cft)
    if verbose
        println("Done solving for cft.")
    end
    # create the form field from the solution coefficients
    potential = Forms.build_form_field(potential_space, sol_cft)
    cft = d(potential)

    if save
        output_filename_cft = "cft-sol-" * tag * ".vtu"
        output_file_cft = joinpath(output_data_folder, output_filename_cft)
        Mantis.Plot.plot(
            cft;
            vtk_filename=output_file_cft,
            n_subcells=1,
            degree=5,
            ascii=false,
            compress=false,
        )
    end

    return cft
end
