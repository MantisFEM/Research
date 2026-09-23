"""
    BSplineSpace{F, TE, TI, TJ} <: AbstractFESpace{1, 1, 1}

Structure containing information about a univariate B-Spline function space defined on
`patch_1d::Mesh.Patch1D`, with given `polynomial_degree` and `regularity` per breakpoint.
Note that while the section spaces on each element are the same, they don't necessarily have
to be polynomials; they are just named `polynomials` for convention.

# Fields
- `knot_vector::KnotVector`: 1-dimensional knot vector.
- `extraction_op::ExtractionOperator`: Stores extraction coefficients and basis indices.
- `polynomials::F`: local section space F, named `polynomials` just for convention.
- `dof_partition::Vector{Vector{Vector{Int}}}`: Indices of boundary degrees of freedom.
"""
struct BSplineSpace{F, TE, TI, TJ} <: AbstractFESpace{1, 1, 1}
    knot_vector::KnotVector
    extraction_op::ExtractionOperator{1, TE, TI, TJ}
    polynomials::F
    dof_partition::Vector{Vector{Vector{Int}}}

    function BSplineSpace(
        patch_1d::Mesh.Patch1D,
        polynomials::F,
        regularity::Vector{Int},
        n_dofs_left::Int=1,
        n_dofs_right::Int=1,
    ) where {F <: AbstractCanonicalSpace}
        polynomial_degree = get_polynomial_degree(polynomials)

        if polynomial_degree < 0
            throw(ArgumentError("""\
                The polynomial degree must be greater than or equal to 0, but is \
                $polynomial_degree.\
                """))
        end

        num_breakpoints = size(patch_1d) + 1
        if num_breakpoints != length(regularity)
            throw(ArgumentError("""\
                The number of regularity conditions should be equal to the number of \
                breakpoints, but there are $(num_breakpoints) breakpoints and \
                $(length(regularity)) regularity conditions.\
                """))
        end

        for i in eachindex(regularity)
            if polynomial_degree <= regularity[i]
                throw(
                    ArgumentError("""\
                  The polynomial degree must be greater than the regularity, but the \
                  polynomial degree is $polynomial_degree and the regularity at index $i \
                  is $(regularity[i]).\
                  """)
                )
            end
        end

        for i in eachindex(regularity)
            if regularity[i] < -1
                throw(ArgumentError("""\
                    The minimum regularity is -1 (element-wise discontinuous), but the \
                    regularity at index $i is $(regularity[i]).\
                    """))
            end
        end

        if F <: AbstractLagrangePolynomials
            if maximum(regularity) > 0
                throw(
                    ArgumentError("""\
                  The regularity conditions for Lagrange polynomials must be -1 \
                  (discontinuous) or 0 (C^0 continuous). You have regularity conditions \
                  $(regularity), which has maximum $(maximum(regularity)).\
                  """)
                )
            end
        end

        knot_vector = create_knot_vector(
            patch_1d, polynomial_degree, regularity, "regularity"
        )
        extraction_op = extract_bspline_to_section_space(knot_vector, polynomials)
        bspline_dim = get_num_basis(extraction_op)

        dof_partition = Vector{Vector{Vector{Int}}}(undef, 1)
        dof_partition[1] = Vector{Vector{Int}}(undef, 3)
        # First, store the left dofs ...
        dof_partition[1][1] = collect(1:n_dofs_left)
        # ... then the interior dofs ...
        dof_partition[1][2] = collect((n_dofs_left + 1):(bspline_dim - n_dofs_right))
        # ... and then finally the right dofs.
        dof_partition[1][3] = collect((bspline_dim - n_dofs_right + 1):bspline_dim)

        return new{F, get_EIJ_types(extraction_op)...}(
            knot_vector, extraction_op, polynomials, dof_partition
        )
    end
end

# Helper functions with classical choices for defaults.
function BSplineSpace(
    patch_1d::Mesh.Patch1D, polynomial_degree::Int, regularity::Vector{Int}
)
    return BSplineSpace(patch_1d, Bernstein(polynomial_degree), regularity)
end
function BSplineSpace(
    patch_1d::Mesh.Patch1D, polynomials::AbstractCanonicalSpace, regularity::Int
)
    # Open knot vector (-1 regularity at the endpoints), given internal regularity.
    regularity = [-1; repeat([regularity], size(patch_1d) - 1); -1]
    return BSplineSpace(patch_1d, polynomials, regularity)
end
function BSplineSpace(patch_1d::Mesh.Patch1D, polynomial_degree::Int, regularity::Int)
    regularity = [-1; repeat([regularity], size(patch_1d) - 1); -1]
    return BSplineSpace(patch_1d, Bernstein(polynomial_degree), regularity)
end

"""
    get_knot_vector(space::BSplineSpace)

Returns the knot vector of the B-spline space `space`.

# Arguments
- `space::BSplineSpace`: The B-spline space.

# Returns
- `::KnotVector`: The knot vector of the B-spline space.
"""
function get_knot_vector(space::BSplineSpace)
    return space.knot_vector
end

"""
    get_polynomials(space::BSplineSpace)

Returns the reference Bernstein polynomials of `space`.

# Arguments
- `space::BSplineSpace`: A univariate B-Spline function space.

# Returns
- `::Bernstein`: Bernstein polynomials.
"""
function get_polynomials(space::BSplineSpace)
    return space.polynomials
end

function get_local_basis(
    space::BSplineSpace,
    element_id::Int,
    xi::Points.AbstractPoints{1},
    nderivatives::Int,
    component_id::Int=1,
)
    # The output of this function must correspond to the general evaluate function, so the
    # output must be a vector{vector{vector{Matrix{Float64}}}}. The output of the evaluate
    # on polynomials is a vector{vector{Matrix{Float64}}}, so we need to add an extra layer
    # of vectors to the output, corresponding to the component.
    section_space_eval = evaluate(get_polynomials(space), xi, nderivatives)
    ext_eval = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for i in 1:(nderivatives + 1)
        # The section spaces, which are CanonicalSpaces, are always 1D, so one derivative
        # per derivative order.
        ext_eval[i] = Vector{Vector{Matrix{Float64}}}(undef, 1)
        ext_eval[i][1] = [section_space_eval[i][1]]
    end

    return ext_eval
end

# Note that `elem_id` is an optional dummy argument for uniformity with other spaces (dummy
# because the degree is the same for all elements for B-splines).
function get_polynomial_degree(space::BSplineSpace, elem_id::Int=0)
    return get_polynomial_degree(get_polynomials(space))
end

"""
    get_patch(space::BSplineSpace)

Returns the patch of the univariate function space `space`.

# Arguments
- `space::BSplineSpace`: The B-Spline function space.

# Returns
- `::Mesh.Patch1D`: The patch of the B-Spline space.
"""
function get_patch(space::BSplineSpace)
    return get_knot_vector(space).patch_1d
end

"""
    get_multiplicity_vector(space::BSplineSpace)

Returns the multiplicities of the knot vector associated with the univariate function space
`space`.

# Arguments
- `space::BSplineSpace`: The B-Spline function space.

# Returns
- `::Vector{Int}`: The multiplicity of the knot vector associated with the B-Spline space.
"""
function get_multiplicity_vector(space::BSplineSpace)
    return get_knot_vector(space).multiplicity
end

"""
    get_regularity_vector(bspline::BSplineSpace)

Returns the regularities of the knot vector associated with the univariate function space `bspline`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.

# Returns
- `::Vector{Int}`: The regularity of the knot vector associated with the B-Spline space.
"""
function get_regularity_vector(bspline::BSplineSpace)
    return get_polynomial_degree(bspline) .- get_multiplicity_vector(bspline)
end

function get_num_elements(space::BSplineSpace)
    return size(get_knot_vector(space).patch_1d)
end

"""
    get_element_measure(space::BSplineSpace, element_id::Int)

Returns the size of the element specified by `element_id`.

# Arguments
- `space::BSplineSpace`: The B-Spline function space.
- `element_id::Int`: The id of the element.

# Returns
- `::Float64`: The size of the element.
"""
function get_element_measure(space::BSplineSpace, element_id::Int)
    return get_element_measure(get_knot_vector(space), element_id)
end

function get_element_lengths(space::BSplineSpace, element_id::Int)
    return get_element_measure(get_knot_vector(space), element_id)
end

"""
    get_element_vertices(space::BSplineSpace, element_id::Int)

Returns the vertices of the element specified by `element_id`.

# Arguments
- `space::BSplineSpace`: The B-Spline function space.
- `element_id::Int`: The id of the element.

# Returns
- `::NTuple{1, Vector{Float64}`: The vertices of the element.
"""
function get_element_vertices(space::BSplineSpace, element_id::Int)
    return Mesh.get_element_vertices(get_patch(space), element_id)
end

"""
    get_support(space::BSplineSpace, basis_id::Int)

Returns the elements where the B-spline given by `basis_id` is supported.

# Arguments
- `space::BSplineSpace`: The B-Spline function space.
- `basis_id::Int`: The id of the basis function.

# Returns
- `::Vector{Int}`: The support of the basis function.
"""
function get_support(space::BSplineSpace, basis_id::Int)
    first_element = convert_knot_to_breakpoint_idx(get_knot_vector(space), basis_id)
    last_element =
        convert_knot_to_breakpoint_idx(
            get_knot_vector(space), basis_id + get_knot_vector(space).polynomial_degree + 1
        ) - 1
    return collect(first_element:last_element)
end

function get_local_knot_vector(space::BSplineSpace, basis_idx::Int)
    knot_vector = get_knot_vector(space)
    deg = get_polynomial_degree(space)

    knot_cum_sum = cumsum(knot_vector.multiplicity)

    first_breakpoint_idx = convert_knot_to_breakpoint_idx(knot_vector, basis_idx)
    last_breakpoint_idx = convert_knot_to_breakpoint_idx(knot_vector, basis_idx + deg + 1)

    first_knot_mult = knot_cum_sum[first_breakpoint_idx] - basis_idx + 1
    last_knot_mult = basis_idx + deg + 1 - knot_cum_sum[last_breakpoint_idx - 1]

    breakpoints = get_patch(space).breakpoints[first_breakpoint_idx:last_breakpoint_idx]
    multiplicity = vcat(
        first_knot_mult,
        get_multiplicity_vector(space)[(first_breakpoint_idx + 1):(last_breakpoint_idx - 1)],
        last_knot_mult,
    )

    return KnotVector(Mesh.Patch1D(breakpoints), deg, multiplicity)
end

function get_max_local_dim(space::BSplineSpace)
    return get_knot_vector(space).polynomial_degree + 1
end

function get_greville_points(space::BSplineSpace)
    return get_greville_points(get_knot_vector(space))
end

function assemble_global_extraction_matrix(space::BSplineSpace)
    # Number of global basis functions
    num_global_basis = get_num_basis(space)
    # Number of elements
    nel = get_num_elements(space)
    # Number of local basis functions
    num_local_basis = (get_polynomial_degree(get_polynomials(space)) + 1) .* ones(Int, nel)
    num_local_basis_offset = cumsum([0; num_local_basis])
    # Initialize the global extraction matrix
    global_extraction_matrix = zeros(Float64, num_local_basis_offset[end], num_global_basis)

    # Loop over all elements
    for el_id in 1:nel
        # get extraction on this element
        extraction_coefficients = get_extraction_coefficients(space, el_id)
        global_basis_indices = get_basis_indices(space, el_id)
        # get local basis indices
        local_basis_indices =
            (num_local_basis_offset[el_id] + 1):num_local_basis_offset[el_id + 1]

        # Assemble the global extraction matrix
        global_extraction_matrix[local_basis_indices, global_basis_indices] =
            extraction_coefficients
    end

    return SparseArrays.sparse(global_extraction_matrix)
end

"""
    get_derivative_space(space::BSplineSpace)

Returns the derivative space of the B-spline space.

# Arguments
- `space::BSplineSpace`: The B-spline space.

# Returns
- `::BSplineSpace`: The derivative space.
"""
function get_derivative_space(space::BSplineSpace)
    # polynomial degree of derivative space
    p = get_polynomial_degree(space)
    dpolynomials = get_derivative_space(get_polynomials(space))

    # modified left and right dof-partitioning
    dof_partition = get_dof_partition(space)
    n_left = max(0, length(dof_partition[1][1]) - 1)
    n_right = max(0, length(dof_partition[1][3]) - 1)

    # regularity of derivative space
    dregularity = (p - 1) .- get_multiplicity_vector(space)
    for i in eachindex(dregularity)
        if dregularity[i] < -1
            dregularity[i] = -1
        end
    end

    return BSplineSpace(get_patch(space), dpolynomials, dregularity, n_left, n_right)
end

"""
    get_support_on_element(bspline::BSplineSpace, element_id::Int)

Get the indices of the basis functions that are supported on the given element.

# Arguments
- `bspline::BSplineSpace`: The b-spline space.
- `element_id::Int`: ID of the element.

# Returns
- `::Vector{Int}`: Indices of the basis function supported on the element.
"""
function get_support_on_element(bspline::BSplineSpace, element_id::Int)
    return get_basis_indices(bspline, element_id)
end

"""
    create_derivative_space(bspline::BSplineSpace{Bernstein})

Creates the BSplineSpace which contains the derivative of the given
b-spline. Note that this is NOT the Curry-Schoenberg space.

# Arguments
- `bspline::BSplineSpace{Bernstein}`: B-spline space to differentiate.

# Returns
- `::BSplineSpace`: Derivative b-spline space.
"""
function create_derivative_space(bspline::BSplineSpace)
    derivative_reg = get_regularity_vector(bspline) .- 1
    derivative_reg[1] = -1
    derivative_reg[end] = -1

    return BSplineSpace(
        get_patch(bspline), get_polynomial_degree(bspline) - 1, derivative_reg
    )
end

"""
    compute_derivative_coefficients(
        coeffs_bspline::Vector{Float64},
        bspline::BSplineSpace{Bernstein},
        bsplineder::BSplineSpace{Bernstein},
    )

Computes the coefficients for the derivative of the given b-spline. The
coefficients are for the space as constructed by
[create_derivative_space(bspline::BSplineSpace)](@ref)

# Arguments
- `coeffs_bspline::Vector{Float64}`: Coefficients of the b-spline to differentiate.
- `bspline::BSplineSpace{Bernstein}`: Space of the b-spline to differentiate.
- `bsplineder::BSplineSpace{Bernstein}`: Space of the b-spline derivatives.

# Returns
- `::Vector{Float64}`: Coefficients of the product b-spline.
"""
function compute_derivative_coefficients(
    coeffs_bspline::Vector{Float64}, bspline::BSplineSpace, bsplineder::BSplineSpace
)
    p = get_polynomial_degree(bspline)

    brk_der = Mesh.get_breakpoints(get_patch(bsplineder))
    m_der = get_multiplicity_vector(bsplineder)
    knt_der = reduce(vcat, repeat([brk_der[i]], m_der[i]) for i in eachindex(m_der))

    coeffs_derivative = zeros(get_num_basis(bsplineder))
    for i in eachindex(coeffs_derivative)
        coeffs_derivative[i] =
            p * (coeffs_bspline[i + 1] - coeffs_bspline[i]) / (knt_der[i + p] - knt_der[i])
    end

    return coeffs_derivative
end

"""
    _create_interpolation_matrix(bspline::BSplineSpace{Bernstein}, points::Vector{Float64})

Creates the interpolation matrix for the given B-spline space at the
given points.

# Arguments
- `bspline::BSplineSpac{Bernstein}e`: B-spline space.
- `points::Vector{Float64}`: Points at which to evaluate the B-spline.

# Returns
- `::Matrix{Float64}`: Interpolation matrix.
"""
function _create_interpolation_matrix(bspline::BSplineSpace, points::Vector{Float64})

    # Evaluate the splines at the given points.
    bspline_eval = zeros(length(points), get_num_basis(bspline))
    breakpoints = (get_knot_vector(bspline)).patch_1d.breakpoints
    for i in 1:1:get_num_elements(bspline)
        # Collect the points that are in the current element.
        points_elem = Float64[]
        points_indices = Int[]
        for j in 1:1:length(points)
            # Exclude the last breakpoint, unless it is the last point
            # in the domain.
            if points[j] >= breakpoints[i] && (
                points[j] < breakpoints[i + 1] ||
                (breakpoints[i + 1] == breakpoints[end] && points[j] == breakpoints[end])
            )
                push!(points_elem, points[j])
                push!(points_indices, j)
            end
        end

        # Map the points to the local element coordinates.
        @. points_elem =
            (points_elem - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i])

        # Evaluate the B-spline at the new points.
        evals, idxs = evaluate(bspline, i, Points.CartesianPoints((points_elem,)))
        bspline_eval[points_indices, idxs] += evals[1][1][1]
    end

    return bspline_eval
end

"""
    create_product_space(bspline1::BSplineSpace{Bernstein}, bspline2::BSplineSpace{Bernstein})

Creates the BSplineSpace which contains the result of the product of the
two given b-splines.

# Arguments
- `bspline1::BSplineSpace{Bernstein}`: B-spline space 1.
- `bspline2::BSplineSpace{Bernstein}`: B-spline space 2.

# Returns
- `::BSplineSpace{Bernstein}`: Product b-spline space.

# Notes and References
This function creates the product space based on the references noted
below. There are other options (such as the blossoming approach used in
[Kapl2017](@cite)) but these are not used here.

For the original algorithm, see [Morken1991](@cite). See also
[Vermeulen1992](@cite) for the same algorithm with a different notation.
"""
function create_product_space(bspline1::BSplineSpace, bspline2::BSplineSpace)
    # Note that we used the degree of the splines, while the reference
    # of the original algorithm use the order! (order = degree + 1)
    p_1 = get_polynomial_degree(bspline1)
    p_2 = get_polynomial_degree(bspline2)
    prod_p = p_1 + p_2

    brk_1 = Mesh.get_breakpoints(get_patch(bspline1))
    m_1 = get_multiplicity_vector(bspline1)
    brk_2 = Mesh.get_breakpoints(get_patch(bspline2))
    m_2 = get_multiplicity_vector(bspline2)

    prod_brk = sort!(unique(vcat(brk_1, brk_2)))
    prod_m = Vector{Int}(undef, length(prod_brk))
    for idx in eachindex(prod_m, prod_brk)
        idx_1 = findfirst(brk -> brk == prod_brk[idx], brk_1)
        idx_2 = findfirst(brk -> brk == prod_brk[idx], brk_2)

        if !isnothing(idx_1) && !isnothing(idx_2)
            prod_m[idx] = max(p_1 + m_2[idx_2], p_2 + m_1[idx_1])
        elseif !isnothing(idx_2)
            prod_m[idx] = p_1 + m_2[idx_2]
        elseif !isnothing(idx_1)
            prod_m[idx] = p_2 + m_1[idx_1]
        end
    end

    # Compute regularity vector.
    prod_r = prod_p .- prod_m

    patch1d = Mesh.Patch1D(prod_brk)

    return BSplineSpace(patch1d, prod_p, prod_r)
end

"""
    compute_product_coefficients(coeffs_bspline1::Vector{Float64}, bspline1::BSplineSpace{Bernstein}, coeffs_bspline2::Vector{Float64}, bspline2::BSplineSpace{Bernstein}, bsplineprod::BSplineSpace{Bernstein}, alg::String="interpolate")

Computes the coefficients for the product b-spline from the given
coefficients. The coefficients are for the space as constructed by
[create_product_space(bspline1::BSplineSpace, bspline2::BSplineSpace)](@ref).

# Arguments
- `coeffs_bspline1::Vector{Float64}`: Coefficients of the first b-spline.
- `bspline1::BSplineSpace{Bernstein}`: First b-spline space.
- `coeffs_bspline2::Vector{Float64}`: Coefficients of the second b-spline.
- `bspline2::BSplineSpace{Bernstein}`: Second b-spline space.
- `bsplineprod::BSplineSpace{Bernstein}`: Product b-spline space.
- `alg::String`: Algorithm to use. Options are `recursive` and `interpolate`.

# Returns
- `::Vector{Float64}`: Coefficients of the product b-spline.

# Notes and References
This function creates the product space based on the references noted
below. There are other options (such as the blossoming approach used in
[Kapl2017](@cite)) but these are not used here.

The algorithm `recursive` is recursive, so can become slow when many
coefficients need to be computed. For the original algorithm, see [Morken1991](@cite).

The algorithm `interpolate` interpolates the B-splines at the Greville
points of the product space and multiplies the result.
"""
function compute_product_coefficients(
    coeffs_bspline1::Vector{Float64},
    bspline1::BSplineSpace,
    coeffs_bspline2::Vector{Float64},
    bspline2::BSplineSpace,
    bsplineprod::BSplineSpace,
    alg::String="interpolate",
)
    if alg == "recursive"
        order_1 = get_polynomial_degree(bspline1) + 1
        order_2 = get_polynomial_degree(bspline2) + 1

        brk_1 = Mesh.get_breakpoints(get_patch(bspline1))
        m_1 = get_multiplicity_vector(bspline1)
        knt_1 = reduce(vcat, repeat([brk_1[i]], m_1[i]) for i in eachindex(m_1))

        brk_2 = Mesh.get_breakpoints(get_patch(bspline2))
        m_2 = get_multiplicity_vector(bspline2)
        knt_2 = reduce(vcat, repeat([brk_2[i]], m_2[i]) for i in eachindex(m_2))

        brk_prod = Mesh.get_breakpoints(get_patch(bsplineprod))
        m_prod = get_multiplicity_vector(bsplineprod)
        knt_prod = reduce(vcat, repeat([brk_prod[i]], m_prod[i]) for i in eachindex(m_prod))

        coeffs_prod = zeros(get_num_basis(bsplineprod))
        for h in eachindex(coeffs_prod)
            for (i, j) in
                Iterators.product(eachindex(coeffs_bspline1), eachindex(coeffs_bspline2))
                if coeffs_bspline1[i] != 0.0 && coeffs_bspline2[j] != 0.0
                    # Avoid computing coefficients that are zero. This
                    # saves a lot of time since no recursive
                    # computations are needed.
                    value =
                        _gamma_prod_coeffs(
                            h, i, j, order_1, order_2, knt_prod, knt_1, knt_2
                        ) *
                        coeffs_bspline1[i] *
                        coeffs_bspline2[j]
                    coeffs_prod[h] += value
                end
            end
        end

        return coeffs_prod

    elseif alg == "interpolate"
        # Get the Greville points of the product space.
        greville_points = get_greville_points(bsplineprod)[1]

        # Evaluate the first b-spline at the Greville points.
        bspl1 = _create_interpolation_matrix(bspline1, greville_points) * coeffs_bspline1

        # Evaluate the second b-spline at the Greville points.
        bspl2 = _create_interpolation_matrix(bspline2, greville_points) * coeffs_bspline2

        # Evaluate the product space at the Greville points.
        bsplineprod_eval = _create_interpolation_matrix(bsplineprod, greville_points)

        # Solve the interpolation problem.
        return bsplineprod_eval \ (bspl1 .* bspl2)

    else
        throw(ArgumentError("Algorithm $alg not supported."))
    end
end

"""
    _omega(i::Int, k::Int, t::Vector{Float64}, x::Float64)

Computes the omega function used in
[_gamma_prod_coeffs(i::Int, j1::Int, j2::Int, k1::Int, k2::Int, z::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64})](@ref).

# Arguments
- `i::Int`: Index for knot vector `t`.
- `k::Int`: Order of the spline.
- `t::Vector{Float64}`: Knot vector.
- `x::Float64`: Value to evaluate the omega function at.

# Returns
- `::Float64`: The requested value of omega.

# Notes and References
See equation 1.2 in [Morken1991](@cite) for the definition of omega.
This function uses the same notation.
"""
function _omega(i::Int, k::Int, t::Vector{Float64}, x::Float64)
    if t[i] < t[i + k - 1]
        return (x - t[i]) / (t[i + k - 1] - t[i])
    else
        return 0.0
    end
end

"""
    _alpha(j::Int, k::Int, i::Int, tau::Vector{Float64}, t::Vector{Float64})

Computes the alpha function used in
[_gamma_prod_coeffs(i::Int, j1::Int, j2::Int, k1::Int, k2::Int, z::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64})](@ref).

# Arguments
- `j::Int`: Index for knot vector `tau`.
- `k::Int`: Order of the spline.
- `i::Int`: Index for knot vector `t`.
- `tau::Vector{Float64}`: Local knot vector.
- `t::Vector{Float64}`: Global knot vector.

# Returns
- `::Float64`: The requested value of alpha.

# Notes and References
See equation 1.7 in [Morken1991](@cite) for the definition of alpha.
This function uses the same notation.
"""
function _alpha(j::Int, k::Int, i::Int, tau::Vector{Float64}, t::Vector{Float64})
    if k == 1
        if t[i] >= tau[j] && t[i] < tau[j + 1]
            return 1.0
        else
            return 0.0
        end
    else
        return _omega(j, k, tau, t[i + k - 1]) * _alpha(j, k - 1, i, tau, t) +
               (1.0 - _omega(j + 1, k, tau, t[i + k - 1])) * _alpha(j + 1, k - 1, i, tau, t)
    end
end

"""
    _gamma_prod_coeffs(
        i::Int,
        j1::Int,
        j2::Int,
        k1::Int,
        k2::Int,
        z::Vector{Float64},
        x::Vector{Float64},
        y::Vector{Float64},
    )

Computes the gamma function used in
[compute_product_coefficients(coeffs_bspline1::Vector{Float64}, bspline1::BSplineSpace, coeffs_bspline2::Vector{Float64}, bspline2::BSplineSpace, bsplineprod::BSplineSpace)](@ref).

# Arguments
- `i::Int`: Index for the coefficients of the product space.
- `j1::Int`: Index for the coefficients of the first spline space.
- `j2::Int`: Index for the coefficients of the second spline space.
- `k1::Int`: Order of the first spline space.
- `k2::Int`: Order of the second spline space.
- `z::Vector{Float64}`: Knot vector of the product space.
- `x::Vector{Float64}`: Knot vector of the first spline space.
- `y::Vector{Float64}`: Knot vector of the second spline space.

# Returns
- `::Float64`: The requested value of gamma.

# Notes and References
The algorithm for computing the coefficients is recursive, so can become
slow when many coefficients need to be computed.

For the original algorithm, see Proposition 4.1 in [Morken1991](@cite).
See also [Vermeulen1992](@cite). This function uses a notational similar
to the one of the notation of [Morken1991](@cite), with the knot
sequences as indicated in [Vermeulen1992](@cite).
"""
function _gamma_prod_coeffs(
    i::Int,
    j1::Int,
    j2::Int,
    k1::Int,
    k2::Int,
    z::Vector{Float64},
    x::Vector{Float64},
    y::Vector{Float64},
)
    # Compute the new order of the product space for this Gamma. This
    # line is important for the correctness of the algorithm. Yet, it is
    # not clearly mentioned in the original paper.
    k = k1 + k2 - 1

    if k1 == 1
        return _alpha(j1, 1, i, x, z) * _alpha(j2, k2, i, y, z)
    elseif k2 == 1
        return _alpha(j1, k1, i, x, z) * _alpha(j2, 1, i, y, z)
    else
        # Check the omega values to avoid unnecessary calculations.
        if _omega(j1, k1, x, z[i + k - 1]) == 0.0 &&
            _omega(j1 + 1, k1, x, z[i + k - 1]) == 1.0
            f1 = 0.0
        elseif _omega(j1, k1, x, z[i + k - 1]) == 0.0
            f1 =
                (1.0 - _omega(j1 + 1, k1, x, z[i + k - 1])) *
                _gamma_prod_coeffs(i, j1 + 1, j2, k1 - 1, k2, z, x, y)
        elseif _omega(j1 + 1, k1, x, z[i + k - 1]) == 1.0
            f1 =
                _omega(j1, k1, x, z[i + k - 1]) *
                _gamma_prod_coeffs(i, j1, j2, k1 - 1, k2, z, x, y)
        else
            f1 =
                _omega(j1, k1, x, z[i + k - 1]) *
                _gamma_prod_coeffs(i, j1, j2, k1 - 1, k2, z, x, y) +
                (1.0 - _omega(j1 + 1, k1, x, z[i + k - 1])) *
                _gamma_prod_coeffs(i, j1 + 1, j2, k1 - 1, k2, z, x, y)
        end

        # Check the omega values to avoid unnecessary calculations.
        if _omega(j2, k2, y, z[i + k - 1]) == 0.0 &&
            _omega(j2 + 1, k2, y, z[i + k - 1]) == 1.0
            f2 = 0.0
        elseif _omega(j2, k2, y, z[i + k - 1]) == 0.0
            f2 =
                (1.0 - _omega(j2 + 1, k2, y, z[i + k - 1])) *
                _gamma_prod_coeffs(i, j1, j2 + 1, k1, k2 - 1, z, x, y)
        elseif _omega(j2 + 1, k2, y, z[i + k - 1]) == 1.0
            f2 =
                _omega(j2, k2, y, z[i + k - 1]) *
                _gamma_prod_coeffs(i, j1, j2, k1, k2 - 1, z, x, y)
        else
            f2 =
                _omega(j2, k2, y, z[i + k - 1]) *
                _gamma_prod_coeffs(i, j1, j2, k1, k2 - 1, z, x, y) +
                (1.0 - _omega(j2 + 1, k2, y, z[i + k - 1])) *
                _gamma_prod_coeffs(i, j1, j2 + 1, k1, k2 - 1, z, x, y)
        end

        return ((k1 - 1.0) * f1 + (k2 - 1.0) * f2) / (k - 1.0)
    end
end
