# Polynomial interface with matching gluing data at the outer boundaries.
function mapping_patch_1_pol(x::AbstractVector{Float64}, polynomial_factor)
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    a = polynomial_factor
    return [a * x[2]^4 * x[1] - 2.0 * a * x[2]^3 * x[1] + a * x[2]^2 * x[1] + x[1], x[2]]
end
function dmapping_patch_1_pol(x::AbstractVector{Float64}, polynomial_factor)
    # df[1]/dx df[1]/dy; df[2]/dx df[2]/dy
    a = polynomial_factor
    return [
        [a * x[2]^4 - 2.0 * a * x[2]^3 + a * x[2]^2 + 1.0 4.0 * a * x[2]^3 * x[1] -
                                                          6.0 * a * x[2]^2 * x[1] +
                                                          2.0 * a * x[2] * x[1]]
        [0.0 1.0]
    ]
end
function ddmapping_patch_1_pol(x::AbstractVector{Float64}, polynomial_factor)
    a = polynomial_factor
    return (
        [
            [0.0 4.0 * a * x[2]^3 - 6.0 * a * x[2]^2 + 2.0 * a * x[2]]
            [4.0 * a * x[2]^3 - 6.0 * a * x[2]^2 + 2.0 * a * x[2] 12.0 * a * x[2]^2 * x[1] -
                                                                  12.0 * a * x[2] * x[1] +
                                                                  2.0 * a * x[1]]
        ],
        [
            [0.0 0.0]
            [0.0 0.0]
        ],
    )
end
mapping_patch_1_poly = Mantis.Geometry.Mapping(
    (2, 2), mapping_patch_1_pol, dmapping_patch_1_pol, ddmapping_patch_1_pol
)
function mapping_patch_2_pol(x::AbstractVector{Float64}, polynomial_factor)
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    a = polynomial_factor
    return [
        a * x[2]^4 * (1.0 - x[1]) - 2.0 * a * x[2]^3 * (1.0 - x[1]) +
        a * x[2]^2 * (1.0 - x[1]) +
        x[1] +
        1.0,
        x[2],
    ]
end
function dmapping_patch_2_pol(x::AbstractVector{Float64}, polynomial_factor)
    # df[1]/dx df[1]/dy; df[2]/dx df[2]/dy
    a = polynomial_factor
    return [
        [-a * x[2]^4 + 2.0 * a * x[2]^3 - a * x[2]^2 + 1.0 4.0 * a * x[2]^3 * (1.0 - x[1]) -
                                                           6.0 * a * x[2]^2 * (1.0 - x[1]) +
                                                           2.0 * a * x[2] * (1.0 - x[1])]
        [0.0 1.0]
    ]
end
function ddmapping_patch_2_pol(x::AbstractVector{Float64}, polynomial_factor)
    a = polynomial_factor
    return (
        [
            [0.0 -4.0 * a * x[2]^3 + 6.0 * a * x[2]^2 - 2.0 * a * x[2]]
            [-4.0 * a * x[2]^3 + 6.0 * a * x[2]^2 - 2.0 * a * x[2] 12.0 *
                                                                   a *
                                                                   x[2]^2 *
                                                                   (1.0 - x[1]) -
                                                                   12.0 *
                                                                   a *
                                                                   x[2] *
                                                                   (1.0 - x[1]) +
                                                                   2.0 * a * (1.0 - x[1])]
        ],
        [
            [0.0 0.0]
            [0.0 0.0]
        ],
    )
end
mapping_patch_2_poly = Mantis.Geometry.Mapping(
    (2, 2), mapping_patch_2_pol, dmapping_patch_2_pol, ddmapping_patch_2_pol
)

function create_polynomial_2patch_mappings(polynomial_factor::Float64=0.5)
    function mapping_patch_1_pol(x::AbstractVector{Float64})
        # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
        a = polynomial_factor
        return [
            a * x[2]^4 * x[1] - 2.0 * a * x[2]^3 * x[1] + a * x[2]^2 * x[1] + x[1], x[2]
        ]
    end
    function dmapping_patch_1_pol(x::AbstractVector{Float64})
        # df[1]/dx df[1]/dy; df[2]/dx df[2]/dy
        a = polynomial_factor
        return [
            [a * x[2]^4 - 2.0 * a * x[2]^3 + a * x[2]^2 + 1.0 4.0 * a * x[2]^3 * x[1] -
                                                              6.0 * a * x[2]^2 * x[1] +
                                                              2.0 * a * x[2] * x[1]]
            [0.0 1.0]
        ]
    end
    function ddmapping_patch_1_pol(x::AbstractVector{Float64})
        a = polynomial_factor
        return (
            [
                [0.0 4.0 * a * x[2]^3 - 6.0 * a * x[2]^2 + 2.0 * a * x[2]]
                [4.0 * a * x[2]^3 - 6.0 * a * x[2]^2 + 2.0 * a * x[2] 12.0 *
                                                                      a *
                                                                      x[2]^2 *
                                                                      x[1] -
                                                                      12.0 *
                                                                      a *
                                                                      x[2] *
                                                                      x[1] + 2.0 * a * x[1]]
            ],
            [
                [0.0 0.0]
                [0.0 0.0]
            ],
        )
    end
    mapping_patch_1_poly = Mantis.Geometry.Mapping(
        (2, 2), mapping_patch_1_pol, dmapping_patch_1_pol, ddmapping_patch_1_pol
    )
    function mapping_patch_2_pol(x::AbstractVector{Float64})
        # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
        a = polynomial_factor
        return [
            a * x[2]^4 * (1.0 - x[1]) - 2.0 * a * x[2]^3 * (1.0 - x[1]) +
            a * x[2]^2 * (1.0 - x[1]) +
            x[1] +
            1.0,
            x[2],
        ]
    end
    function dmapping_patch_2_pol(x::AbstractVector{Float64})
        # df[1]/dx df[1]/dy; df[2]/dx df[2]/dy
        a = polynomial_factor
        return [
            [-a * x[2]^4 + 2.0 * a * x[2]^3 - a * x[2]^2 + 1.0 4.0 *
                                                               a *
                                                               x[2]^3 *
                                                               (1.0 - x[1]) -
                                                               6.0 *
                                                               a *
                                                               x[2]^2 *
                                                               (1.0 - x[1]) +
                                                               2.0 *
                                                               a *
                                                               x[2] *
                                                               (1.0 - x[1])]
            [0.0 1.0]
        ]
    end
    function ddmapping_patch_2_pol(x::AbstractVector{Float64})
        a = polynomial_factor
        return (
            [
                [0.0 -4.0 * a * x[2]^3 + 6.0 * a * x[2]^2 - 2.0 * a * x[2]]
                [-4.0 * a * x[2]^3 + 6.0 * a * x[2]^2 - 2.0 * a * x[2] 12.0 *
                                                                       a *
                                                                       x[2]^2 *
                                                                       (1.0 - x[1]) -
                                                                       12.0 *
                                                                       a *
                                                                       x[2] *
                                                                       (1.0 - x[1]) +
                                                                       2.0 *
                                                                       a *
                                                                       (1.0 - x[1])]
            ],
            [
                [0.0 0.0]
                [0.0 0.0]
            ],
        )
    end
    mapping_patch_2_poly = Mantis.Geometry.Mapping(
        (2, 2), mapping_patch_2_pol, dmapping_patch_2_pol, ddmapping_patch_2_pol
    )

    return mapping_patch_1_poly, mapping_patch_2_poly
end

function create_polynomial_2patch_geometry(
    num_elements_per_dim_per_patch::NTuple{num_patches, NTuple{2, Int}},
    polynomial_factor::Float64=0.5,
) where {num_patches}
    if num_patches != 1 && num_patches != 2
        error("Can only create crazy geometry with 1 or 2 patches.")
    end

    # Patch-wise mappings
    poly_mapping_patch_1_2p = Mantis.Geometry.Mapping(
        (2, 2),
        x -> mapping_patch_1_pol(x, polynomial_factor),
        x -> dmapping_patch_1_pol(x, polynomial_factor),
        x -> ddmapping_patch_1_pol(x, polynomial_factor),
    )

    poly_mapping_patch_2_2p = Mantis.Geometry.Mapping(
        (2, 2),
        x -> mapping_patch_2_pol(x, polynomial_factor),
        x -> dmapping_patch_2_pol(x, polynomial_factor),
        x -> ddmapping_patch_2_pol(x, polynomial_factor),
    )

    # Choose which geometry to use.
    mapping_obj_patch_1 = poly_mapping_patch_1_2p
    mapping_obj_patch_2 = poly_mapping_patch_2_2p
    mapping_objects = (mapping_obj_patch_1, mapping_obj_patch_2)

    return Mantis.Geometry.MultiPatchGeometry(
        ntuple(num_patches) do np
            geom_cart_patch_i = Geometry.CartesianGeometry((
                0.0:(1.0 / num_elements_per_dim_per_patch[np][1]):1.0,
                0.0:(1.0 / num_elements_per_dim_per_patch[np][2]):1.0,
            ))

            return Mantis.Geometry.MappedGeometry(geom_cart_patch_i, mapping_objects[np])
        end,
    )
end
