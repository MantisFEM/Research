
function dirichlet_bc_indices_1_form_BH(space, clamped)
    if FunctionSpaces.get_num_patches(space.fem_space) == 1
        if clamped
            # Directly access the underlying TP space (in the DS space) to get the dof partition
            # which is easier than dealing with the additional indices from the DS space.
            tangential_indices_1 = reduce(
                vcat, space.fem_space.component_spaces[1].dof_partition[1][[2, 4, 6, 8]]
            )
            tangential_indices_2 = reduce(
                vcat, space.fem_space.component_spaces[2].dof_partition[1][[2, 4, 6, 8]]
            )
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
                    vcat, space.fem_space.component_spaces[2].dof_partition[1][[1, 4, 7]]
                ),
                reduce(
                    vcat, space.fem_space.component_spaces[2].dof_partition[2][[3, 6, 9]]
                ),
            ],
        )
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
    else
        error(LazyString("Not implemented for more than 3 patches."))
    end

    bc_basis_indices = vcat(
        tangential_indices_1,
        tangential_indices_2 .+
        FunctionSpaces.get_num_basis(space.fem_space.component_spaces[1]),
    )

    return vcat(bc_basis_indices)
end

function solve_Ainsworth_Parker(
    Wp_H1::Forms.AbstractFormSpace{2, 0, G},
    Theta_space::Forms.AbstractFormSpace{2, 1, G},
    q_space::Forms.AbstractFormSpace{2, 2, G},
    verbose::Bool,
    dΩ,
    f⁰, #Forms.AnalyticalFormField(Val(0), forcing_function, geometry, "f⁰"),
    zp_space=Wp_H1,
    save=false;
    clamped=false,
) where {G}
    # Step 1: Solve for zₚ from
    zp = step1(zp_space, f⁰, dΩ, verbose)

    # Step 2: Solve for θₚ₋₁ and rₚ₋₂ from
    # (div ψ, div θ) + (rot ψ, rₚ₋₂) = (ψ, grad zₚ) ∀ ψ ∈ Gp-1Gamma
    #                   (rot θₚ₋₁, s) = 0 ∀ s ∈ rot(Gp-1Gamma)
    my_one = Forms.ConstantFormSpace(Val(2), Forms.get_geometry(Wp_H1), "μ")

    # Tangential zero boundary conditions for Gp-1Gamma1 and Gp-1Gamma2
    bc_basis_indices = dirichlet_bc_indices_1_form_BH(Theta_space, clamped)
    # println(bc_basis_indices)
    # println(Theta_space.fem_space.dof_partition)
    # println(Theta_space.fem_space.component_spaces[1].dof_partition)
    bc_G = Dict(i => 0.0 for i in bc_basis_indices)
    # bc_G = Forms.set_dirichlet_boundary_conditions(Theta_space, 0.0)

    # solve for coefficients of solution
    if verbose
        println(
            "Solving for θ and q (num theta dofs = $(Forms.get_num_basis(Theta_space)), num q dofs = $(Forms.get_num_basis(q_space)))...",
        )
    end
    weak_form_inputs_theta = Assemblers.WeakFormInputs(
        (Theta_space, q_space, my_one), (zp,)
    )

    lhs_expressions_theta, rhs_expressions_theta = stokes_like_sys(
        weak_form_inputs_theta, dΩ
    )
    weak_form_theta = Assemblers.WeakForm(
        lhs_expressions_theta, rhs_expressions_theta, weak_form_inputs_theta
    )

    # assemble all matrices
    A_theta, b_theta = Assemblers.assemble(weak_form_theta, bc_G)
    if verbose
        println("\ncond(A_theta)=$(LinearAlgebra.cond(Matrix(A_theta))).")
        println("rank(A_theta)=$(LinearAlgebra.rank(Matrix(A_theta))).")
        println("size(A_theta)=$(size(A_theta)).")
    end
    sol_theta_r = vec(A_theta \ b_theta)

    if verbose
        println("Done solving for θ and q.")
    end
    # create the form field from the solution coefficients
    θ, q, μ = Forms.build_form_fields((Theta_space, q_space, my_one), sol_theta_r)

    # Step 3: Solve for wₚ
    w = step3(Wp_H1, θ, dΩ, verbose)

    return w, θ, zp
end
