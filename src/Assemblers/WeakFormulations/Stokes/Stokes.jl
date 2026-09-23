import LinearSolve as LS

function stokes_sys(
    inputs::Assemblers.AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    ψ, t, ν = Assemblers.get_test_forms(inputs)
    θ, q, μ = Assemblers.get_trial_forms(inputs)
    # ψ, t = Assemblers.get_test_forms(inputs)
    # θ, q = Assemblers.get_trial_forms(inputs)
    fzp = Assemblers.get_forcing(inputs)

    # A₁₁ = ∫(δ(ψ) ∧ ★(δ(θ)), dΩ)  # (div ψ, div θ)
    # A₁₂ = ∫(d(ψ) ∧ ★(q), dΩ)  # (curl ψ, q)
    # A₂₁ = ∫(t ∧ ★(d(θ)), dΩ)  # (t, curl θ)
    # A₂₃ = ∫(t ∧ ★(μ), dΩ)  # (t, μ)
    # A₃₂ = ∫(ν ∧ ★(q), dΩ)  # (ν, q)
    A₁₁ = ∫(d(ψ) ∧ ★(d(θ)), dΩ)  # (curl ψ, curl θ)
    A₁₂ = ∫(δ(ψ) ∧ ★(q), dΩ)  # (div ψ, q)
    A₂₁ = ∫(t ∧ ★(δ(θ)), dΩ)  # (t, div θ)
    A₂₃ = ∫(t ∧ ★(μ), dΩ)  # (t, μ)
    A₃₂ = ∫(ν ∧ ★(q), dΩ)  # (ν, q)
    # A₁₁ = ∫(ψ ∧ ★(θ), dΩ)  # (ψ, θ)
    # A₁₂ = ∫(d(ψ) ∧ ★(q), dΩ)  # (curl ψ, q)
    # A₂₁ = ∫(t ∧ ★(d(θ)), dΩ)  # (t, curl θ)
    # A₂₃ = ∫(t ∧ ★(μ), dΩ)  # (t, μ)
    # A₃₂ = ∫(ν ∧ ★(q), dΩ)  # (ν, q)
    # A₁₁ = ∫(ψ ∧ ★(θ), dΩ)  # (ψ, θ)
    # A₁₂ = ∫(δ(ψ) ∧ ★(q), dΩ)  # (div ψ, q)
    # A₂₁ = ∫(t ∧ ★(δ(θ)), dΩ)  # (t, div θ)
    # A₂₃ = ∫(t ∧ ★(μ), dΩ)  # (t, μ)
    # A₃₂ = ∫(ν ∧ ★(q), dΩ)  # (ν, q)

    lhs_expression = ((A₁₁, A₁₂, 0), (A₂₁, 0, A₂₃), (0, A₃₂, 0))
    # lhs_expression = ((A₁₁, A₁₂), (A₂₁, 0))

    b₁ = ∫(ψ ∧ ★(fzp), dΩ)  # (ψ, grad zₚ)

    rhs_expression = ((b₁,), (0,), (0,))
    # rhs_expression = ((b₁,), (0,))

    return lhs_expression, rhs_expression
end

function solve_Stokes_Taylor_Hood(
    Theta_space, q_space, verbose::Bool, dΩ, forcing_function, save=false; clamped=false
)
    # Step 2: Solve for θₚ₋₁ and rₚ₋₂ from
    # (div ψ, div θ) + (rot ψ, rₚ₋₂) = (ψ, grad zₚ) ∀ ψ ∈ Gp-1Gamma
    #                   (rot θₚ₋₁, s) = 0 ∀ s ∈ rot(Gp-1Gamma)
    my_one = Forms.ConstantFormSpace(Val(0), Forms.get_geometry(Theta_space), "μ")

    # Tangential zero boundary conditions for Gp-1Gamma1 and Gp-1Gamma2
    bc_basis_indices = dirichlet_bc_indices_1_form_TH(Theta_space, clamped)
    bc_G = Dict(i => 0.0 for i in bc_basis_indices)

    # solve for coefficients of solution
    if verbose
        println(
            "Solving for θ and q (num theta dofs = $(Forms.get_num_basis(Theta_space)), num q dofs = $(Forms.get_num_basis(q_space)))...",
        )
    end
    forcing = Forms.AnalyticalFormField(
        Val(1), forcing_function, Forms.get_geometry(Theta_space), "f⁰"
    )
    weak_form_inputs_theta = Assemblers.WeakFormInputs(
        (Theta_space, q_space, my_one), (forcing,)
    )
    # weak_form_inputs_theta = Assemblers.WeakFormInputs((Theta_space, q_space), (forcing,))

    lhs_expressions_theta, rhs_expressions_theta = stokes_sys(weak_form_inputs_theta, dΩ)
    weak_form_theta = Assemblers.WeakForm(
        lhs_expressions_theta, rhs_expressions_theta, weak_form_inputs_theta
    )

    # assemble all matrices
    A_theta, b_theta = Assemblers.assemble(weak_form_theta, bc_G; rhs_type=Vector{Float64})
    @show size(A_theta)
    if size(A_theta, 1) < 3000
        println(LinearAlgebra.cond(Matrix(A_theta)))
        println(LinearAlgebra.rank(Matrix(A_theta)))
    end
    if verbose
        println("\ncond(A_theta)=$(LinearAlgebra.cond(Matrix(A_theta))).")
        println("rank(A_theta)=$(LinearAlgebra.rank(Matrix(A_theta))).")
        println("size(A_theta)=$(size(A_theta)).")
    end
    # sol_theta_r = vec(A_theta \ b_theta)
    prob = LS.LinearProblem(A_theta, b_theta)
    linsolve = LS.init(prob)
    LS.solve!(linsolve)
    sol_theta_r = linsolve.u

    if verbose
        println("Done solving for θ and q.")
    end
    # create the form field from the solution coefficients
    θ, q, μ = Forms.build_form_fields((Theta_space, q_space, my_one), sol_theta_r)

    return θ, q, μ
end
