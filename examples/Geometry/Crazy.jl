using Mantis

# Helper functions
function map_interval(x::Float64, a_new::Float64, b_new::Float64, a_old::Float64, b_old::Float64)
    return a_new + (b_new - a_new)*(x - a_old)/(b_old - a_old)
end
function dmap_interval(x::Float64, a_new::Float64, b_new::Float64, a_old::Float64, b_old::Float64)
    return (b_new - a_new)/(b_old - a_old)
end

const Lleft = 0.0
const Lright = 1.0
const Lbottom = 0.0
const Ltop = 1.0

# Original crazy mapping and its derivative
function crazy_mapping(x::AbstractVector{Float64})
    x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
    x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
    return [x[1] + ((Lright-Lleft)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new), x[2] + ((Ltop-Lbottom)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new)]
end
function crazy_dmapping(x::AbstractVector{Float64})
    x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
    x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
    return [1.0 + pi*crazy_c*cospi(x1_new)*sinpi(x2_new) ((Lright-Lleft)/(Ltop-Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new); ((Ltop-Lbottom)/(Lright-Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0 + pi*crazy_c*sinpi(x1_new)*cospi(x2_new)]
end
# Patch-wise mappings
function mapping_patch_i_crazy(x::AbstractVector{Float64}, crazy_c, x1_new_f, x2_new_f)
    x1_new = x1_new_f(x[1])
    x2_new = x2_new_f(x[2])
    return [x1_new + ((Lright-Lleft)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new), x2_new + ((Ltop-Lbottom)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new)]
end
function dmapping_patch_i_crazy(x::AbstractVector{Float64}, crazy_c, x1_new_f, x2_new_f, dx1_new_f, dx2_new_f)
    x1_new = x1_new_f(x[1])
    x2_new = x2_new_f(x[2])
    dx1_new = dx1_new_f(x[1])
    dx2_new = dx2_new_f(x[2])
    return [
        [dx1_new + ((Lright-Lleft)/2.0)*dx1_new*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) ((Lright-Lleft)/2.0)*dx2_new*pi*crazy_c*sinpi(x1_new)*cospi(x2_new)]
        [((Ltop-Lbottom)/2.0)*dx1_new*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) dx2_new + ((Ltop-Lbottom)/2.0)*dx2_new*pi*crazy_c*sinpi(x1_new)*cospi(x2_new)]
    ]
end
function ddmapping_patch_i_crazy(x::AbstractVector{Float64}, crazy_c, x1_new_f, x2_new_f, dx1_new_f, dx2_new_f)
    x1_new = x1_new_f(x[1])
    x2_new = x2_new_f(x[2])
    dx1_new = dx1_new_f(x[1])
    dx2_new = dx2_new_f(x[2])
    return (
        [[-((Lright-Lleft)/2.0)*dx1_new^2*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new) ((Lright-Lleft)/2.0)*dx1_new*dx2_new*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new)]
            [((Lright-Lleft)/2.0)*dx1_new*dx2_new*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new) -((Lright-Lleft)/2.0)*dx2_new^2*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new)]],
        [[-((Ltop-Lbottom)/2.0)*dx1_new^2*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new) ((Ltop-Lbottom)/2.0)*dx1_new*dx2_new*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new)]
            [((Ltop-Lbottom)/2.0)*dx1_new*dx2_new*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new) -((Ltop-Lbottom)/2.0)*dx2_new^2*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new)]]
    )
end

# Crazy mesh up to four patches.
function create_crazy_geometry(
    num_elements_per_dim_per_patch::NTuple{num_patches,NTuple{2,Int}},
    crazy_c = 0.2,
    split_loc_lr = 0.1, # Between -1.0 and 1.0
    split_loc_bt = 0.0, # Between -1.0 and 1.0
) where {num_patches}
    if num_patches != 1 && num_patches != 2 && num_patches != 4
        error("Can only create crazy geometry with 1, 2 or 4 patches.")
    end
    if split_loc_lr < -1.0 || split_loc_lr > 1.0
        error("split_loc_lr must be between -1.0 and 1.0")
    end
    if split_loc_bt < -1.0 || split_loc_bt > 1.0
        error("split_loc_bt must be between -1.0 and 1.0")
    end

    # Patch-wise mappings
    crazy_mapping_patch_1_4p = Mantis.Geometry.Mapping(
        (2,2),
        x -> mapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, -1.0, split_loc_lr, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
        x -> dmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, -1.0, split_loc_lr, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_lr, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
        x -> ddmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, -1.0, split_loc_lr, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_lr, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
    )

    crazy_mapping_patch_2_4p = Mantis.Geometry.Mapping(
        (2,2),
        x -> mapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr, 1.0, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
        x -> dmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr, 1.0, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_lr, 1.0, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
        x -> ddmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr, 1.0, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_lr, 1.0, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
    )

    crazy_mapping_patch_3_4p = Mantis.Geometry.Mapping(
        (2,2),
        x -> mapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr, 1.0, 0.0, 1.0),
            y -> map_interval(y, split_loc_bt, 1.0, 0.0, 1.0)
        ),
        x -> dmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr, 1.0, 0.0, 1.0),
            y -> map_interval(y, split_loc_bt, 1.0, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_lr, 1.0, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_bt, 1.0, 0.0, 1.0)
        ),
        x -> ddmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr, 1.0, 0.0, 1.0),
            y -> map_interval(y, split_loc_bt, 1.0, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_lr, 1.0, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_bt, 1.0, 0.0, 1.0)
        ),
    )

    crazy_mapping_patch_4_4p = Mantis.Geometry.Mapping(
        (2,2),
        x -> mapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, -1.0, split_loc_lr, 0.0, 1.0),
            y -> map_interval(y, split_loc_bt, 1.0, 0.0, 1.0)
        ),
        x -> dmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, -1.0, split_loc_lr, 0.0, 1.0),
            y -> map_interval(y, split_loc_bt, 1.0, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_lr, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_bt, 1.0, 0.0, 1.0)
        ),
        x -> ddmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, -1.0, split_loc_lr, 0.0, 1.0),
            y -> map_interval(y, split_loc_bt, 1.0, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_lr, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_bt, 1.0, 0.0, 1.0)
        ),
    )

    # Choose which geometry to use.
    mapping_objects = (
        crazy_mapping_patch_1_4p,
        crazy_mapping_patch_2_4p,
        crazy_mapping_patch_3_4p,
        crazy_mapping_patch_4_4p,
    )

    return Mantis.Geometry.MultiPatchGeometry(ntuple(num_patches) do np
        geom_cart_patch_i = Geometry.CartesianGeometry((
            0.0:1.0/num_elements_per_dim_per_patch[np][1]:1.0,
            0.0:1.0/num_elements_per_dim_per_patch[np][2]:1.0,
        ))

        return Mantis.Geometry.MappedGeometry(
            geom_cart_patch_i, mapping_objects[np]
        )
    end
    )
end


# Crazy mesh with three patches.
function create_crazy_geometry3(
    num_elements_per_dim_per_patch::NTuple{num_patches,NTuple{2,Int}},
    crazy_c::Float64 = 0.2,
    split_loc_lr1 = -1/3, # Between -1.0 and 1.0
    split_loc_lr2 = 1/3, # Between -1.0 and 1.0
    split_loc_bt = 0.0, # Between -1.0 and 1.0
) where {num_patches}
    if num_patches != 3
        error("Can only create this crazy geometry with 3 patches.")
    end
    if split_loc_lr1 < -1.0 || split_loc_lr1 > 1.0
        error("split_loc_lr1 must be between -1.0 and 1.0")
    end
    if split_loc_lr2 < -1.0 || split_loc_lr2 > 1.0
        error("split_loc_lr2 must be between -1.0 and 1.0")
    end
    if split_loc_bt < -1.0 || split_loc_bt > 1.0
        error("split_loc_bt must be between -1.0 and 1.0")
    end

    # Patch-wise mappings
    crazy_mapping_patch_1_3p = Mantis.Geometry.Mapping(
        (2,2),
        x -> mapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, -1.0, split_loc_lr1, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
        x -> dmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, -1.0, split_loc_lr1, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_lr1, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
        x -> ddmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, -1.0, split_loc_lr1, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_lr1, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
    )

    crazy_mapping_patch_2_3p = Mantis.Geometry.Mapping(
        (2,2),
        x -> mapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr1, split_loc_lr2, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
        x -> dmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr1, split_loc_lr2, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_lr1, split_loc_lr2, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
        x -> ddmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr1, split_loc_lr2, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_lr1, split_loc_lr2, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
    )

    crazy_mapping_patch_3_3p = Mantis.Geometry.Mapping(
        (2,2),
        x -> mapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr2, 1.0, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
        x -> dmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr2, 1.0, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_lr2, 1.0, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
        x -> ddmapping_patch_i_crazy(
            x,
            crazy_c,
            y -> map_interval(y, split_loc_lr2, 1.0, 0.0, 1.0),
            y -> map_interval(y, -1.0, split_loc_bt, 0.0, 1.0),
            y -> dmap_interval(y, split_loc_lr2, 1.0, 0.0, 1.0),
            y -> dmap_interval(y, -1.0, split_loc_bt, 0.0, 1.0)
        ),
    )

    # Choose which geometry to use.
    mapping_obj_patch_1 = crazy_mapping_patch_1_3p
    mapping_obj_patch_2 = crazy_mapping_patch_2_3p
    mapping_obj_patch_3 = crazy_mapping_patch_3_3p
    mapping_objects = (
        mapping_obj_patch_1, mapping_obj_patch_2, mapping_obj_patch_3
    )

    return Mantis.Geometry.MultiPatchGeometry(ntuple(num_patches) do np
        geom_cart_patch_i = Geometry.CartesianGeometry((
            0.0:1.0/num_elements_per_dim_per_patch[np][1]:1.0,
            0.0:1.0/num_elements_per_dim_per_patch[np][2]:1.0,
        ))

        return Mantis.Geometry.MappedGeometry(
            geom_cart_patch_i, mapping_objects[np]
        )
    end
    )
end



# Patch-wise mappings
function mapping_simple(x::AbstractVector{Float64})
    return [x[1]^2, x[2]]
end
function dmapping_simple(x::AbstractVector{Float64})
    return [
        [2.0*x[1] 0.0]
        [0.0 1.0]
    ]
end
function ddmapping_simple(x::AbstractVector{Float64})
    return (
        [[2.0 0.0]
         [0.0 0.0]],
        [[0.0 0.0]
         [0.0 0.0]]
    )
end

function mapping_simple_2(x::AbstractVector{Float64})
    return [2.0*x[1]^2+1.0, x[2]]
end
function dmapping_simple_2(x::AbstractVector{Float64})
    return [
        [4.0*x[1] 0.0]
        [0.0 1.0]
    ]
end
function ddmapping_simple_2(x::AbstractVector{Float64})
    return (
        [[4.0 0.0]
         [0.0 0.0]],
        [[0.0 0.0]
         [0.0 0.0]]
    )
end

function create_simple_geometry(
    num_elements_per_dim_per_patch::NTuple{num_patches,NTuple{2,Int}},
) where {num_patches}
    if num_patches != 1 && num_patches != 2
        error("Can only create simple geometry with 1 or 2 patches.")
    end

    # Patch-wise mappings
    simple_mapping_patch_1 = Mantis.Geometry.Mapping(
        (2,2),
        mapping_simple,
        dmapping_simple,
        ddmapping_simple,
    )

    simple_mapping_patch_2 = Mantis.Geometry.Mapping(
        (2,2),
        mapping_simple_2,
        dmapping_simple_2,
        ddmapping_simple_2,
    )

    # Choose which geometry to use.
    mapping_objects = (simple_mapping_patch_1, simple_mapping_patch_2)

    return Mantis.Geometry.MultiPatchGeometry(ntuple(num_patches) do np
        geom_cart_patch_i = Geometry.CartesianGeometry((
            0.0:1.0/num_elements_per_dim_per_patch[np][1]:1.0,
            0.0:1.0/num_elements_per_dim_per_patch[np][2]:1.0,
        ))

        return Mantis.Geometry.MappedGeometry(
            geom_cart_patch_i, mapping_objects[np]
        )
    end
    )
end


function mapping_simple2(x::AbstractVector{Float64})
    return [x[1] + 0.1*sinpi(2.0*x[1])*sinpi(2.0*x[2]), x[2]]
end
function dmapping_simple2(x::AbstractVector{Float64})
    return [
        [1.0 + 0.1*2.0*pi*cospi(2.0*x[1])*sinpi(2.0*x[2])  0.1*2.0*pi*sinpi(2.0*x[1])*cospi(2.0*x[2])]
        [0.0 1.0]
    ]
end
function ddmapping_simple2(x::AbstractVector{Float64})
    return (
        [[-0.1*4.0*pi^2*sinpi(2.0*x[1])*sinpi(2.0*x[2])  0.1*4.0*pi^2*cospi(2.0*x[1])*cospi(2.0*x[2])]
         [0.1*4.0*pi^2*cospi(2.0*x[1])*cospi(2.0*x[2])  -0.1*4.0*pi^2*sinpi(2.0*x[1])*sinpi(2.0*x[2])]],
        [[0.0 0.0]
         [0.0 0.0]]
    )
end

function create_simple2_geometry(
    num_elements_per_dim_per_patch::NTuple{num_patches,NTuple{2,Int}},
) where {num_patches}
    # Patch-wise mappings
    simple_mapping_patch_1_4p = Mantis.Geometry.Mapping(
        (2,2),
        mapping_simple2,
        dmapping_simple2,
        ddmapping_simple2,
    )

    # Choose which geometry to use.
    mapping_obj_patch_1 = simple_mapping_patch_1_4p
    mapping_objects = (mapping_obj_patch_1,)

    return Mantis.Geometry.MultiPatchGeometry(ntuple(num_patches) do np
        geom_cart_patch_i = Geometry.CartesianGeometry((
            0.0:1.0/num_elements_per_dim_per_patch[np][1]:1.0,
            0.0:1.0/num_elements_per_dim_per_patch[np][2]:1.0,
        ))

        return Mantis.Geometry.MappedGeometry(
            geom_cart_patch_i, mapping_objects[np]
        )
    end
    )
end
