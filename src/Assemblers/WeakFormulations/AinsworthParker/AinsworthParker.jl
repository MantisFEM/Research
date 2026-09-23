function zp_sys(
    inputs::Assemblers.AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule, dΩ2=nothing
)
    v⁰ = Assemblers.get_test_form(inputs)
    zₚ = Assemblers.get_trial_form(inputs)
    f = Assemblers.get_forcing(inputs)

    Azp = ∫(d(v⁰) ∧ ★(d(zₚ)), dΩ)
    lhs_expression = ((Azp,),)

    if !isnothing(dΩ2)
        bzp = ∫(v⁰ ∧ ★(f), dΩ2)
    else
        bzp = ∫(v⁰ ∧ ★(f), dΩ)
    end
    rhs_expression = ((bzp,),)

    return lhs_expression, rhs_expression
end

function stokes_like_sys_stab(
    inputs::Assemblers.AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    ψ, q, ν = Assemblers.get_test_forms(inputs)
    θ, p, μ = Assemblers.get_trial_forms(inputs)
    fzp = Assemblers.get_forcing(inputs)

    A₁₁ = ∫(δ(ψ) ∧ ★(δ(θ)), dΩ) + ∫(d(ψ) ∧ ★(d(θ)), dΩ)  # (div ψ, div θ) + (rot ψ, rot θ)
    A₁₂ = ∫(d(ψ) ∧ ★(p), dΩ)  # (rot ψ, q)
    A₂₁ = ∫(q ∧ ★(d(θ)), dΩ)  # (t, rot θ)
    A₂₃ = ∫(q ∧ ★(μ), dΩ)  # (t, μ)
    A₃₂ = ∫(ν ∧ ★(p), dΩ)  # (ν, q)

    lhs_expression = ((A₁₁, A₁₂, 0), (A₂₁, 0, A₂₃), (0, A₃₂, 0))

    b₁ = ∫(ψ ∧ ★(d(fzp)), dΩ)  # (ψ, grad zₚ)

    rhs_expression = ((b₁,), (0,), (0,))

    return lhs_expression, rhs_expression
end

function stokes_like_sys(
    inputs::Assemblers.AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    ψ, q, ν = Assemblers.get_test_forms(inputs)
    θ, p, μ = Assemblers.get_trial_forms(inputs)
    fzp = Assemblers.get_forcing(inputs)

    A₁₁ = ∫(δ(ψ) ∧ ★(δ(θ)), dΩ)  # (div ψ, div θ)
    A₁₂ = ∫(d(ψ) ∧ ★(p), dΩ)  # (rot ψ, q)
    A₂₁ = ∫(q ∧ ★(d(θ)), dΩ)  # (t, rot θ)
    A₂₃ = ∫(q ∧ ★(μ), dΩ)  # (t, μ)
    A₃₂ = ∫(ν ∧ ★(p), dΩ)  # (ν, q)
    # A₁₁ = ∫(δ(ψ) ∧ ★(δ(θ)), dΩ)  # (div ψ, div θ)
    # A₁₂ = -∫(ψ ∧ ★(δ(p)), dΩ)  # (rot ψ, q)
    # A₂₁ = -∫(δ(q) ∧ ★(θ), dΩ)  # (t, rot θ)
    # A₂₃ = ∫(q ∧ ★(μ), dΩ)  # (t, μ)
    # A₃₂ = ∫(ν ∧ ★(p), dΩ)  # (ν, q)

    lhs_expression = ((A₁₁, A₁₂, 0), (A₂₁, 0, A₂₃), (0, A₃₂, 0))

    b₁ = ∫(ψ ∧ ★(d(fzp)), dΩ)  # (ψ, grad zₚ)

    rhs_expression = ((b₁,), (0,), (0,))

    return lhs_expression, rhs_expression
end

function w_sys(
    inputs::Assemblers.AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    v = Assemblers.get_test_form(inputs)
    w = Assemblers.get_trial_form(inputs)
    θ = Assemblers.get_forcing(inputs)

    Aw = ∫(d(v) ∧ ★(d(w)), dΩ)
    lhs_expression = ((Aw,),)

    bw = ∫(d(v) ∧ ★(θ), dΩ)
    rhs_expression = ((bw,),)

    return Assemblers.WeakForm(lhs_expression, rhs_expression, inputs)
end

function compute_rhs_diff(zp_space, f⁰, dΩ, f⁰_scaled, dΩ2)
    # Step 1: Solve for zₚ from
    # (grad zₚ, grad v) = (f, vₚ) for all v ∈ Wp_H1
    weak_form_inputs_zp = Assemblers.WeakFormInputs(zp_space, f⁰)
    weak_form_inputs_zp_scaled = Assemblers.WeakFormInputs(zp_space, f⁰_scaled)

    # Homogeneous dirichlet boundary conditions on zp.
    bc_zp_indices = dirichlet_bc_indices_0_form(zp_space)
    bc_zp = Dict(i => 0.0 for i in bc_zp_indices)

    # Gauss-Legendre
    lhs_expressions_zp_gl, rhs_expressions_zp_gl = zp_sys(weak_form_inputs_zp, dΩ)
    weak_form_zp_gl = Assemblers.WeakForm(
        lhs_expressions_zp_gl, rhs_expressions_zp_gl, weak_form_inputs_zp
    )
    _, b_zp_gl = Assemblers.assemble(weak_form_zp_gl, bc_zp)

    # Gauss-Jacobi
    lhs_expressions_zp_gj, rhs_expressions_zp_gj = zp_sys(weak_form_inputs_zp_scaled, dΩ2)
    weak_form_zp_gj = Assemblers.WeakForm(
        lhs_expressions_zp_gj, rhs_expressions_zp_gj, weak_form_inputs_zp_scaled
    )
    _, b_zp_gj = Assemblers.assemble(weak_form_zp_gj, bc_zp)


    return maximum(abs.(b_zp_gl - b_zp_gj))
end

function step1(zp_space, f⁰, dΩ, verbose, bc_z_coeffs=nothing, dΩ2=nothing)
    # Step 1: Solve for zₚ from
    # (grad zₚ, grad v) = (f, vₚ) for all v ∈ Wp_H1
    weak_form_inputs_zp = Assemblers.WeakFormInputs(zp_space, f⁰)

    # Homogeneous dirichlet boundary conditions on zp.
    bc_zp_indices = dirichlet_bc_indices_0_form(zp_space)
    if isnothing(bc_z_coeffs)
        bc_zp = Dict(i => 0.0 for i in bc_zp_indices)
    else
        bc_zp = Dict(i => bc_z_coeffs[i] for i in bc_zp_indices)
    end
    # bc_zp = Forms.set_dirichlet_boundary_conditions(zp_space, 0.0)

    lhs_expressions_zp, rhs_expressions_zp = zp_sys(weak_form_inputs_zp, dΩ, dΩ2)
    weak_form_zp = Assemblers.WeakForm(
        lhs_expressions_zp, rhs_expressions_zp, weak_form_inputs_zp
    )

    # assemble all matrices
    A_zp, b_zp = Assemblers.assemble(weak_form_zp, bc_zp)
    # solve for coefficients of solution
    if verbose
        println("Solving for zₚ...")
    end
    sol_zp = vec(A_zp \ b_zp)
    if verbose
        println("Done solving for zₚ.")
    end
    # create the form field from the solution coefficients
    return Forms.build_form_field(zp_space, sol_zp)
end

function step3(Wp_H1, θ, dΩ, verbose, bc_w_coeffs=nothing)
    # Step 3: Solve for wₚ from
    # (grad wₚ, grad v) = (θ, grad v) for all v ∈ Wp_H1
    weak_form_inputs_w = Assemblers.WeakFormInputs((Wp_H1,), (θ,))

    weak_form_w = w_sys(weak_form_inputs_w, dΩ)

    # Homogeneous dirichlet boundary conditions on w.
    bc_w_indices = dirichlet_bc_indices_0_form(Wp_H1)
    if isnothing(bc_w_coeffs)
        bc_w = Dict(i => 0.0 for i in bc_w_indices)
    else
        bc_w = Dict(i => bc_w_coeffs[i] for i in bc_w_indices)
    end
    # bc_w = Forms.set_dirichlet_boundary_conditions(Wp_H1, 0.0)

    # assemble all matrices
    A_w, b_w = Assemblers.assemble(weak_form_w, bc_w)
    if verbose
        println("\ncond(A_w)=$(LinearAlgebra.cond(Matrix(A_w))).")
        println("rank(A_w)=$(LinearAlgebra.rank(Matrix(A_w))).")
        println("size(A_w)=$(size(A_w)).")

        # solve for coefficients of solution
        println("Solving for w...")
    end
    sol_w = vec(A_w \ b_w)
    if verbose
        println("Done solving for w.")
    end
    # create the form field from the solution coefficients
    return Forms.build_form_field(Wp_H1, sol_w)
end

function dirichlet_bc_indices_0_form(space)
    if FunctionSpaces.get_num_patches(space.fem_space) == 1
        bc_zp_indices = reduce(
            vcat, space.fem_space.dof_partition[1][[1, 2, 3, 4, 6, 7, 8, 9]]
        )
    elseif FunctionSpaces.get_num_patches(space.fem_space) == 2
        bc_zp_indices = reduce(
            vcat,
            [
                reduce(vcat, space.fem_space.dof_partition[1][[1, 2, 3, 4, 7, 8, 9]]),
                reduce(vcat, space.fem_space.dof_partition[2][[1, 2, 3, 6, 7, 8, 9]]),
            ],
        )
    elseif FunctionSpaces.get_num_patches(space.fem_space) == 3
        bc_zp_indices = reduce(
            vcat,
            [
                reduce(vcat, space.fem_space.dof_partition[1][[1, 2, 3, 4, 7, 8, 9]]),
                reduce(vcat, space.fem_space.dof_partition[2][[1, 2, 3, 7, 8, 9]]),
                reduce(vcat, space.fem_space.dof_partition[3][[1, 2, 3, 6, 7, 8, 9]]),
            ],
        )
    elseif FunctionSpaces.get_num_patches(space.fem_space) == 4
        bc_zp_indices = reduce(
            vcat,
            [
                reduce(vcat, space.fem_space.dof_partition[1][[1, 2, 3, 4, 7]]),
                reduce(vcat, space.fem_space.dof_partition[2][[1, 2, 3, 6, 9]]),
                reduce(vcat, space.fem_space.dof_partition[3][[3, 6, 7, 8, 9]]),
                reduce(vcat, space.fem_space.dof_partition[4][[1, 4, 7, 8, 9]]),
            ],
        )
    elseif FunctionSpaces.get_num_patches(space.fem_space) == 5
        # Carpart
        bc_zp_indices = reduce(
            vcat,
            [
                reduce(vcat, space.fem_space.dof_partition[1][[3, 6, 7, 8, 9]]),
                reduce(vcat, space.fem_space.dof_partition[2][[3, 6, 7, 8, 9]]),
                reduce(vcat, space.fem_space.dof_partition[3][[3, 6, 7, 8, 9]]),
                reduce(vcat, space.fem_space.dof_partition[4][[3, 6, 7, 8, 9]]),
                reduce(vcat, space.fem_space.dof_partition[5][[3, 6, 7, 8, 9]]),
            ],
        )
    else
        error(LazyString("Not implemented for more than 5 patches."))
    end

    return bc_zp_indices
end

include("Biharmonic.jl")
include("TaylorHood.jl")
