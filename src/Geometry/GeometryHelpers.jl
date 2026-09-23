################################################################################
# Some standard geometries
################################################################################

"""
    create_cartesian_box(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
    ) where {manifold_dim}

Create a Cartesian box geometry with `manifold_dim` dimensions, starting at
`starting_points` and with `box_sizes` and `num_elements` defining the size of the box.

# Arguments
  - `starting_points::NTuple{manifold_dim, Float64}`: The starting points of the box.
  - `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box.
  - `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.

# Output
  - `::CartesianGeometry{manifold_dim}`: The Cartesian box geometry.
"""
function create_cartesian_box(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
) where {manifold_dim}
    breakpoints = map(
        LinRange, starting_points, starting_points .+ box_sizes, num_elements .+ 1
    )
    return CartesianGeometry(breakpoints)
end

"""
    create_curvilinear_square(
        starting_points::NTuple{2, Float64},
        box_sizes::NTuple{2, Float64},
        num_elements::NTuple{2, Int};
        c::Float64=0.1,
    )

Create a single-patch curvilinear square geometry with `num_elements` elements in each
direction and a `c` parameter to change the deformation of the mapping. Not that the mapping
becomes singular with `c` = 0.3.

# Arguments
  - `num_elements::NTuple{2,Int}`: The number of elements in each direction.
  - `c::Float64 = 0.2`: The `c` parameter.

# Output
  - `geometry::MappedGeometry{2, 2, 1}`: The curvilinear square geometry.
"""
function create_curvilinear_square(
    starting_points::NTuple{2, Float64},
    box_sizes::NTuple{2, Float64},
    num_elements::NTuple{2, Int};
    crazy_c::Float64=0.1,
)
    # build underlying Cartesian geometry
    unit_square = create_cartesian_box(starting_points, box_sizes, num_elements)

    # build curved mapping
    function mapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return [
            x[1] + ((box_sizes[1]) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
            x[2] + ((box_sizes[2]) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
        ]
    end
    function dmapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return [
            1.0+pi * crazy_c * cospi(x1_new) * sinpi(x2_new) (box_sizes[1]/box_sizes[2])*pi*crazy_c*sinpi(x1_new)*cospi(x2_new)
            (box_sizes[2]/box_sizes[1])*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0+pi * crazy_c * sinpi(x1_new) * cospi(x2_new)
        ]
    end
    function ddmapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return (
            [
                -(2.0 / box_sizes[1])*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new) (2/box_sizes[2])*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new)
                (2.0/box_sizes[2])*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new) -(2 * box_sizes[1] / (box_sizes[2]^2))*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new)
            ],
            [
                -(2 * box_sizes[2] / (box_sizes[1]^2))*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new) (2.0/box_sizes[1])*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new)
                (2.0/box_sizes[1])*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new) -(2.0 / box_sizes[2])*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new)
            ],
        )
    end
    dimension = (2, 2)
    curved_mapping = Mapping(dimension, mapping, dmapping, ddmapping)

    return MappedGeometry(unit_square, curved_mapping)
end


function create_my_square(
    starting_points::NTuple{2, Float64},
    box_sizes::NTuple{2, Float64},
    num_elements::NTuple{2, Int},
)
    # build underlying Cartesian geometry
    unit_square = create_cartesian_box(starting_points, box_sizes, num_elements)

    # build curved mapping
    function mapping(x::AbstractVector)
        return [
            x[1]^2,
            x[2]^2,
        ]
    end
    function dmapping(x::AbstractVector)
        return [
            2*x[1] 0.0
            0.0 2*x[2]
        ]
    end
    function ddmapping(x::AbstractVector)
        return (
            [
                2.0 0.0
                0.0 0.0
            ],
            [
                0.0 0.0
                0.0 2.0
            ],
        )
    end
    dimension = (2, 2)
    curved_mapping = Mapping(dimension, mapping, dmapping, ddmapping)

    return MappedGeometry(unit_square, curved_mapping)
end



@doc raw"""
    create_L_shaped_domain(
        starting_points::NTuple{2, Float64},
        box_sizes::NTuple{2, Float64},
        num_elements::NTuple{2, Int},
        L_domain_verices_x::NTuple{6, Float64},
        L_domain_verices_y::NTuple{6, Float64}
    )

Create an L-shaped domain geometry by combining two Cartesian boxes and applying bilinear mappings to create the L-shape.

          6---5
          |   |
          |   |
4---------3   |
|          \  |
|           \ |
1-------------2

"""
function create_L_shaped_domain(
    starting_points::NTuple{2, Float64},
    box_sizes::NTuple{2, Float64},
    num_elements::NTuple{2, Int},
    L_domain_verices_x::NTuple{6, Float64},
    L_domain_verices_y::NTuple{6, Float64},
)
    # Cartesian geometry
    cart_geom = create_cartesian_box(starting_points, box_sizes, num_elements)
    # Mappings
    mapping1, dmapping1 = bilinear_map(
        (starting_points[1], starting_points[2]),
        (box_sizes[1] / 2, box_sizes[2]),
        (
            L_domain_verices_x[1],
            L_domain_verices_x[2],
            L_domain_verices_x[3],
            L_domain_verices_x[4],
        ),
        (
            L_domain_verices_y[1],
            L_domain_verices_y[2],
            L_domain_verices_y[3],
            L_domain_verices_y[4],
        ),
    )
    mapping2, dmapping2 = bilinear_map(
        (starting_points[1] + box_sizes[1] / 2, starting_points[2]),
        (box_sizes[1] / 2, box_sizes[2]),
        (
            L_domain_verices_x[2],
            L_domain_verices_x[5],
            L_domain_verices_x[6],
            L_domain_verices_x[3],
        ),
        (
            L_domain_verices_y[2],
            L_domain_verices_y[5],
            L_domain_verices_y[6],
            L_domain_verices_y[3],
        ),
    )
    # mapped_geometry
    function mapping(Ξ::AbstractVector)
        if Ξ[1] <= starting_points[1] + box_sizes[1] / 2
            return mapping1(Ξ)
        else
            return mapping2(Ξ)
        end
    end
    function dmapping(Ξ::AbstractVector)
        if Ξ[1] <= starting_points[1] + box_sizes[1] / 2
            return dmapping1(Ξ)
        else
            return dmapping2(Ξ)
        end
    end
    dimension = (2, 2)
    L_map = Mapping(dimension, mapping, dmapping)

    return MappedGeometry(cart_geom, L_map)
end

############################################################################################
#                                         Mappings                                         #
############################################################################################

function affine_map(xi, A, b)
    return xi * A + b
end

function affine_map(
    xi::Points.AbstractPoints{manifold_dim}, A::Matrix, b::Vector
) where {manifold_dim}
    num_points = Points.get_num_points(xi)
    mapped_dim = size(A, 1)
    mapped_xi = ntuple(mapped_dim) do _
        return zeros(Float64, num_points)
    end

    for (point_id, point) in enumerate(xi)
        for j in axes(A, 2)
            for i in axes(A, 1)
                mapped_xi[i][point_id] += affine_map(point[j], A[i, j], 0.0)
            end
        end

        for i in axes(b, 1)
            mapped_xi[i][point_id] += b[i]
        end
    end

    return mapped_xi
end

function bilinear_map(
    starting_points::NTuple{2, Float64},
    box_sizes::NTuple{2, Float64},
    X::NTuple{4, Number},
    Y::NTuple{4, Number},
)
    function mapping(Ξ::AbstractVector)
        ξ = (Ξ[1] - starting_points[1]) / box_sizes[1]
        η = (Ξ[2] - starting_points[2]) / box_sizes[2]
        return [
            (1 - ξ) * (1 - η) * X[1] +
            ξ * (1 - η) * X[2] +
            ξ * η * X[3] +
            (1 - ξ) * η * X[4],
            (1 - ξ) * (1 - η) * Y[1] +
            ξ * (1 - η) * Y[2] +
            ξ * η * Y[3] +
            (1 - ξ) * η * Y[4],
        ]
    end
    function dmapping(Ξ::AbstractVector)
        ξ = (Ξ[1] - starting_points[1]) / box_sizes[1]
        η = (Ξ[2] - starting_points[2]) / box_sizes[2]
        dxdΞ₁ = (-(1 - η) * X[1] + (1 - η) * X[2] + η * X[3] - η * X[4]) / box_sizes[1]
        dxdΞ₂ = (-(1 - ξ) * X[1] - ξ * X[2] + ξ * X[3] + (1 - ξ) * X[4]) / box_sizes[2]
        dydΞ₁ = (-(1 - η) * Y[1] + (1 - η) * Y[2] + η * Y[3] - η * Y[4]) / box_sizes[1]
        dydΞ₂ = (-(1 - ξ) * Y[1] - ξ * Y[2] + ξ * Y[3] + (1 - ξ) * Y[4]) / box_sizes[2]
        return [
            dxdΞ₁ dxdΞ₂
            dydΞ₁ dydΞ₂
        ]
    end
    return mapping, dmapping
end

############################################################################################
#                                      Other methods                                       #
############################################################################################

function vec_tuple_to_matrix(vec_tup)
    return hcat(vec_tup...)
end
