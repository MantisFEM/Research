using Mantis

function create_slanted_2patch_mappings(slant_factor::Float64=0.25)

    # Mappings to create the deformed geometries. The mappings are defined with reference
    # to the unit square [0,1]x[0,1] as parametric domain.
    # Slanted interface between two patches.
    function mapping_patch_1_slant(x::AbstractVector{Float64})
        # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
        return [x[1] + slant_factor * x[1] * x[2], x[2]]
    end
    function dmapping_patch_1_slant(x::AbstractVector{Float64})
        return [1.0+slant_factor * x[2] slant_factor*x[1]; 0.0 1.0]
    end
    function ddmapping_patch_1_slant(x::AbstractVector{Float64})
        return (
            [
                [0.0 slant_factor]
                [slant_factor 0.0]
            ],
            [
                [0.0 0.0]
                [0.0 0.0]
            ],
        )
    end
    mapping_patch_1_slanted = Geometry.Mapping(
        (2, 2), mapping_patch_1_slant, dmapping_patch_1_slant, ddmapping_patch_1_slant
    )
    function mapping_patch_2_slant(x::AbstractVector{Float64})
        # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
        return [x[1] + 1.0 + slant_factor * (1.0 - x[1]) * x[2], x[2]]
    end
    function dmapping_patch_2_slant(x::AbstractVector{Float64})
        return [
            [1.0 - slant_factor * x[2] slant_factor * (1.0 - x[1])]
            [0.0 1.0]
        ]
    end
    function ddmapping_patch_2_slant(x::AbstractVector{Float64})
        return (
            [
                [0.0 -slant_factor]
                [-slant_factor 0.0]
            ],
            [
                [0.0 0.0]
                [0.0 0.0]
            ],
        )
    end
    mapping_patch_2_slanted = Geometry.Mapping(
        (2, 2), mapping_patch_2_slant, dmapping_patch_2_slant, ddmapping_patch_2_slant
    )

    return mapping_patch_1_slanted, mapping_patch_2_slanted
end

function create_slanted_2patch_geometry(
    num_elements_per_dim_per_patch::NTuple{2, NTuple{2, Int}}, slant_factor=0.25
)
    mapping_patch_1_slanted, mapping_patch_2_slanted = create_slanted_2patch_mappings(
        slant_factor
    )

    geom_cart_patch_1 = Geometry.CartesianGeometry((
        0.0:(1.0 / num_elements_per_dim_per_patch[1][1]):1.0,
        0.0:(1.0 / num_elements_per_dim_per_patch[1][2]):1.0,
    ))
    geom_cart_patch_2 = Geometry.CartesianGeometry((
        0.0:(1.0 / num_elements_per_dim_per_patch[2][1]):1.0,
        0.0:(1.0 / num_elements_per_dim_per_patch[2][2]):1.0,
    ))

    geom_mapped_patch_1 = Geometry.MappedGeometry(
        geom_cart_patch_1, mapping_patch_1_slanted
    )
    geom_mapped_patch_2 = Geometry.MappedGeometry(
        geom_cart_patch_2, mapping_patch_2_slanted
    )

    geom_slanted_2patch = Geometry.MultiPatchGeometry((
        geom_mapped_patch_1, geom_mapped_patch_2
    ))

    return geom_slanted_2patch
end

function create_slanted_2patch_mappings_shifted(slant_factor::Float64=0.25)

    # Mappings to create the deformed geometries. The mappings are defined with reference
    # to the unit square [0,1]x[0,1] as parametric domain.
    # Slanted interface between two patches.
    function mapping_patch_1_slant(x::AbstractVector{Float64})
        # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
        return [x[1] + slant_factor * x[1] * x[2], x[2]]
    end
    function dmapping_patch_1_slant(x::AbstractVector{Float64})
        return [1.0+slant_factor * x[2] slant_factor*x[1]; 0.0 1.0]
    end
    function ddmapping_patch_1_slant(x::AbstractVector{Float64})
        return (
            [
                [0.0 slant_factor]
                [slant_factor 0.0]
            ],
            [
                [0.0 0.0]
                [0.0 0.0]
            ],
        )
    end
    mapping_patch_1_slanted = Geometry.Mapping(
        (2, 2), mapping_patch_1_slant, dmapping_patch_1_slant, ddmapping_patch_1_slant
    )
    function mapping_patch_2_slant(x::AbstractVector{Float64})
        # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
        x1 = x[1] - 1.0
        return [x1 + 1.0 + slant_factor * (1.0 - x[1]) * x[2] + 1.0, x[2]]
    end
    function dmapping_patch_2_slant(x::AbstractVector{Float64})
        x1 = x[1] - 1.0
        return [
            [1.0 - slant_factor * x[2] slant_factor * (1.0 - x1)]
            [0.0 1.0]
        ]
    end
    function ddmapping_patch_2_slant(x::AbstractVector{Float64})
        return (
            [
                [0.0 -slant_factor]
                [-slant_factor 0.0]
            ],
            [
                [0.0 0.0]
                [0.0 0.0]
            ],
        )
    end
    mapping_patch_2_slanted = Geometry.Mapping(
        (2, 2), mapping_patch_2_slant, dmapping_patch_2_slant, ddmapping_patch_2_slant
    )

    return mapping_patch_1_slanted, mapping_patch_2_slanted
end
