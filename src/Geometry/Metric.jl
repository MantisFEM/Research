
"""
    metric(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Returns the metric and its determinant.

# Arguments
- `::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.

# Returns
- `g::Vector{SMatrix{manifold_dim, manifold_dim}}`: Metric tensor per evaluation point.
- `sqrt_g::Vector{Float64}`: Square-root of the determinant of the metric per evaluation
    point.
"""
function metric(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    # The metric tensor gᵢⱼ is
    #   gᵢⱼ := ∑ₖᵐ ∂Φᵏ\∂ξᵢ ⋅ ∂Φᵏ\∂ξⱼ
    # Since the Jacobian matrix J is given by
    #   Jᵢⱼ = ∂Φⁱ\∂ξⱼ
    # The metric tensor can be computed as
    #   g = Jᵗ ⋅ J

    J = jacobian(geometry, element_id, xi)

    g = transpose.(J) .* J
    sqrt_g = sqrt.(abs.(det.(g)))

    return g, sqrt_g
end

"""
    inv_metric(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Returns the inverse metric, the metric and its determinant.

# Arguments
- `::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.

# Returns
- `::Vector{SMatrix{manifold_dim, manifold_dim}}`: Inverse metric tensor per evaluation
    point.
- `g::Vector{SMatrix{manifold_dim, manifold_dim}}`: Metric tensor per evaluation point.
- `sqrt_g::Vector{Float64}`: Square-root of the determinant of the metric per evaluation
    point.
"""
function inv_metric(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    g, sqrt_g = metric(geometry, element_id, xi)

    return inv.(g), g, sqrt_g
end

"""
    metric_derivatives(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Returns the inverse metric, the metric and its determinant.

# Arguments
- `geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry.
- `element_id::Int`: Global element id.
- `xi::Points.AbstractPoints{manifold_dim}`: Evaluation points in the canonical domain.

# Returns
- `J::Vector{SMatrix{image_dim, manifold_dim}}`: Jacobian per point.
- `inv_g::Vector{SMatrix{manifold_dim, manifold_dim}}`: Inverse metric tensor per point.
- `g::Vector{SMatrix{manifold_dim, manifold_dim}}`: Metric tensor per point.
- `sqrt_g::Vector{Float64}`: Square-root of the determinant of the metric per point.
- `dgdxs::NTuple{manifold_dim, Vector{SMatrix{manifold_dim, manifold_dim}}}`: Derivative of
    the metric tensor per direction in manifold_dim.
- `dinv_g_dxs::NTuple{manifold_dim, Vector{SMatrix{manifold_dim, manifold_dim}}}`:
    Derivative of the inverse metric tensor per direction in manifold_dim.
- `dsqrt_g_dxs::NTuple{manifold_dim, Vector{Float64}`: Derivative of the square-root of the
    determinant of the metric tensor per direction in manifold_dim.
- `Hs::Vector{NTuple{image_dim, SMatrix{manifold_dim, manifold_dim}}}`: Hessians per point.
"""
function metric_derivatives(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    J = jacobian(geometry, element_id, xi)
    inv_g, g, sqrt_g = inv_metric(geometry, element_id, xi)
    Hs = hessian(geometry, element_id, xi)

    # Compute all the below per coordinate direction in the canonical domain.
    dJ = ntuple(manifold_dim) do i
        return [
            transpose(
                SMatrix{image_dim, manifold_dim}(
                    # The values here are generated per row, but the constructor assumes
                    # that the values are ordered per column. We therefore construct the
                    # transpose first. That's why the image and manifold dims are flipped
                    # in the constructor.
                    (
                        Hs[p][j][i, :][l] for j in eachindex(Hs[p]) for
                        l in eachindex(Hs[p][j][i, :])
                    )...,
                ),
            ) for p in eachindex(Hs)
        ]
    end
    # g = J^T J, so dg = J^T dJ + (dJ)^T J
    dgdxs = ntuple(manifold_dim) do i
        return @. transpose(dJ[i]) * J + transpose(J) * dJ[i]
    end
    # dg^-1 = -g^-1 dg g
    dinv_g_dxs = ntuple(manifold_dim) do i
        return @. -inv_g * dgdxs[i] * inv_g
    end
    # sqrt(det(g)) = 0.5 sqrt(det(g)) tr(dg g^-1)
    dsqrt_g_dxs = ntuple(manifold_dim) do i
        return @. 0.5 * sqrt_g * LinearAlgebra.tr(dgdxs[i] * inv_g)
    end

    return J, inv_g, g, sqrt_g, dgdxs, dinv_g_dxs, dsqrt_g_dxs, Hs
end
