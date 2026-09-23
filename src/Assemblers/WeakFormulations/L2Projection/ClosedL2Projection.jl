############################################################################################
#                                     Global L2 projection                                 #
############################################################################################

"""
    closed_L2_projection(inputs::AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule)

Function to compute the L2 projection of a function onto a discrete form space with an
additional constraint that the resulting form is closed.

# Arguments
- `inputs::AbstractInputs`: The inputs for the weak form assembly, including test, trial and
    forcing terms.
- `dΩ::Quadrature.AbstractGlobalQuadratureRule`: The quadrature rule to use for the integral
    evaluation.

# Returns
- `lhs_expression<:NTuple{num_lhs_rows, NTuple{num_lhs_cols, AbstractRealValuedOperator}}`:
    The left-hand side of the weak form, which is a tuple of tuples contain all the blocks
    of the left-hand side matrix.
- `rhs_expression<:NTuple{num_rhs_rows, NTuple{num_rhs_cols, AbstractRealValuedOperator}}`:
    The right-hand side of the weak form, which is a tuple of tuples contain all the blocks
    of the right-hand side matrix.
"""
function closed_L2_projection(inputs::AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule)
    σ¹, τ¹ = Assemblers.get_test_forms(inputs)
    u¹, v¹ = Assemblers.get_trial_forms(inputs)
    f¹ = Assemblers.get_forcing(inputs)
    A_11 = ∫(σ¹ ∧ ★(u¹), dΩ)
    A_12 = ∫(d(σ¹) ∧ ★(d(v¹)), dΩ)
    A_21 = ∫(d(τ¹) ∧ ★(d(u¹)), dΩ)
    lhs_expressions = ((A_11, A_12), (A_21, 0))
    b_11 = ∫(σ¹ ∧ ★(f¹), dΩ)
    rhs_expressions = ((b_11,), (0,))

    return lhs_expressions, rhs_expressions
end

"""
    solve_L2_projection(Xᵏ, fₑ, dΩ)

Returns the solution of the weak form of the L2 projection.

# Arguments
- `Xᵏ`: The k-form space to use as trial and test space.
- `fₑ`: The forcing term to use for the right-hand side of the weak formulation.
- `dΩ`: The quadrature rule to use for the assembly.

# Returns
- `fₕ::FormField`: The projection of `fₑ` onto `Xᵏ`.
"""
function solve_closed_L2_projection(Xᵏ, fₑ, dΩ)
    weak_form_inputs = WeakFormInputs((Xᵏ, Xᵏ), (fₑ,))
    lhs_expressions, rhs_expressions = closed_L2_projection(weak_form_inputs, dΩ)
    weak_form = WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    A, b = assemble(weak_form)
    sol = vec(SparseArrays.qr(A) \ b)
    fₕ, λₕ = Forms.build_form_fields((Xᵏ, Xᵏ), sol; labels=("fₕ", "λₕ"))

    return fₕ
end
