using Mantis

const slant_factor_x = 0.25
const slant_factor_y = 0.25

# Mappings to create the deformed geometries. The mappings are defined with reference
# to the unit square [0,1]x[0,1] as parametric domain.
# Slanted interface between two patches.
function mapping_patch_1_slant_4p(x::AbstractVector{Float64})
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    return [x[1] + slant_factor_x*x[1]*x[2], x[2] + slant_factor_y*x[1]*x[2]]
end
function dmapping_patch_1_slant_4p(x::AbstractVector{Float64})
    return [[1.0+slant_factor_x*x[2] slant_factor_x*x[1]]
            [slant_factor_y*x[2]     1.0+slant_factor_y*x[1]]]
end
function ddmapping_patch_1_slant_4p(x::AbstractVector{Float64})
    return (
        [[0.0 slant_factor_x]
         [slant_factor_x 0.0]],
        [[0.0 slant_factor_y]
         [slant_factor_y 0.0]]
    )
end

function mapping_patch_2_slant_4p(x::AbstractVector{Float64})
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    return [x[1] + 1.0 + slant_factor_x*(1.0-x[1])*x[2], x[2] + slant_factor_y*(1.0-x[1])*x[2]]
end
function dmapping_patch_2_slant_4p(x::AbstractVector{Float64})
    return [[1.0-slant_factor_x*x[2] slant_factor_x*(1.0-x[1])]
            [-slant_factor_y*x[2]    1.0+slant_factor_y*(1.0-x[1])]]
end
function ddmapping_patch_2_slant_4p(x::AbstractVector{Float64})
    return (
        [[0.0 -slant_factor_x]
         [-slant_factor_x 0.0]],
        [[0.0 -slant_factor_y]
         [-slant_factor_y 0.0]]
    )
end

function mapping_patch_3_slant_4p(x::AbstractVector{Float64})
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    return [x[1] + 1.0 + slant_factor_x*(1.0-x[1])*(1.0-x[2]), x[2] + 1.0 + slant_factor_y*(1.0-x[1])*(1.0-x[2])]
end
function dmapping_patch_3_slant_4p(x::AbstractVector{Float64})
    return [[1.0-slant_factor_x*(1.0-x[2]) -slant_factor_x*(1.0-x[1])]
            [-slant_factor_y*(1.0-x[2])    1.0-slant_factor_y*(1.0-x[1])]]
end
function ddmapping_patch_3_slant_4p(x::AbstractVector{Float64})
    return (
        [[0.0 slant_factor_x]
         [slant_factor_x 0.0]],
        [[0.0 slant_factor_y]
         [slant_factor_y 0.0]]
    )
end

function mapping_patch_4_slant_4p(x::AbstractVector{Float64})
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    return [x[1] + slant_factor_x*x[1]*(1.0-x[2]), x[2] + 1.0 + slant_factor_y*x[1]*(1.0-x[2])]
end
function dmapping_patch_4_slant_4p(x::AbstractVector{Float64})
    return [[1.0+slant_factor_x*(1.0-x[2]) -slant_factor_x*x[1]]
            [slant_factor_y*(1.0-x[2])     1.0-slant_factor_y*x[1]]]
end
function ddmapping_patch_4_slant_4p(x::AbstractVector{Float64})
    return (
        [[0.0 -slant_factor_x]
         [-slant_factor_x 0.0]],
        [[0.0 -slant_factor_y]
         [-slant_factor_y 0.0]]
    )
end

function create_slanted_4patch_geometry(
    num_elements_per_dim_per_patch::NTuple{4,NTuple{2,Int}},
)

    mapping_patch_1_slanted = Geometry.Mapping(
        (2,2), mapping_patch_1_slant_4p, dmapping_patch_1_slant_4p, ddmapping_patch_1_slant_4p
    )
    mapping_patch_2_slanted = Geometry.Mapping(
        (2,2), mapping_patch_2_slant_4p, dmapping_patch_2_slant_4p, ddmapping_patch_2_slant_4p
    )
    mapping_patch_3_slanted = Geometry.Mapping(
        (2,2), mapping_patch_3_slant_4p, dmapping_patch_3_slant_4p, ddmapping_patch_3_slant_4p
    )
    mapping_patch_4_slanted = Geometry.Mapping(
        (2,2), mapping_patch_4_slant_4p, dmapping_patch_4_slant_4p, ddmapping_patch_4_slant_4p
    )


    geom_cart_patch_1 = Geometry.CartesianGeometry(
        (
            0.0:1.0/num_elements_per_dim_per_patch[1][1]:1.0,
            0.0:1.0/num_elements_per_dim_per_patch[1][2]:1.0,
        )
    )
    geom_cart_patch_2 = Geometry.CartesianGeometry(
        (
            0.0:1.0/num_elements_per_dim_per_patch[2][1]:1.0,
            0.0:1.0/num_elements_per_dim_per_patch[2][2]:1.0,
        )
    )
    geom_cart_patch_3 = Geometry.CartesianGeometry(
        (
            0.0:1.0/num_elements_per_dim_per_patch[3][1]:1.0,
            0.0:1.0/num_elements_per_dim_per_patch[3][2]:1.0,
        )
    )
    geom_cart_patch_4 = Geometry.CartesianGeometry(
        (
            0.0:1.0/num_elements_per_dim_per_patch[4][1]:1.0,
            0.0:1.0/num_elements_per_dim_per_patch[4][2]:1.0,
        )
    )

    geom_mapped_patch_1 = Geometry.MappedGeometry(
        geom_cart_patch_1, mapping_patch_1_slanted
    )
    geom_mapped_patch_2 = Geometry.MappedGeometry(
        geom_cart_patch_2, mapping_patch_2_slanted
    )
    geom_mapped_patch_3 = Geometry.MappedGeometry(
        geom_cart_patch_3, mapping_patch_3_slanted
    )
    geom_mapped_patch_4 = Geometry.MappedGeometry(
        geom_cart_patch_4, mapping_patch_4_slanted
    )

    geom_slanted_4patch = Geometry.MultiPatchGeometry(
        (geom_mapped_patch_1, geom_mapped_patch_2, geom_mapped_patch_3, geom_mapped_patch_4)
    )

    return geom_slanted_4patch
end
