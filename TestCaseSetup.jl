# 1-patch
const mesh_con_1patch = (((0, 0), (0, 0), (0, 0), (0, 0)),)
# 2-patch polynomial
const patch_1_con = ((0, 0), (2, 4), (0, 0), (0, 0))
const patch_2_con = ((0, 0), (0, 0), (0, 0), (1, 2))
const mesh_con_2patch = (patch_1_con, patch_2_con)
# 3-patches
const patch_1_con3 = ((0, 0), (2, 4), (0, 0), (0, 0))
const patch_2_con3 = ((0, 0), (3, 4), (0, 0), (1, 2))
const patch_3_con3 = ((0, 0), (0, 0), (0, 0), (2, 2))
const mesh_con_3patch = (patch_1_con3, patch_2_con3, patch_3_con3)
# 4-patch cartesian unit square
const patch_1_con4 = ((0, 0), (2, 4), (4, 1), (0, 0))
const patch_2_con4 = ((0, 0), (0, 0), (3, 1), (1, 2))
const patch_3_con4 = ((2, 3), (0, 0), (0, 0), (4, 2))
const patch_4_con4 = ((1, 3), (3, 4), (0, 0), (0, 0))
const mesh_con_4patch = (patch_1_con4, patch_2_con4, patch_3_con4, patch_4_con4)
# 5-patch car part
const patch_1_con5 = ((5, 4), (0, 0), (0, 0), (2, 1))
const patch_2_con5 = ((1, 4), (0, 0), (0, 0), (3, 1))
const patch_3_con5 = ((2, 4), (0, 0), (0, 0), (4, 1))
const patch_4_con5 = ((3, 4), (0, 0), (0, 0), (5, 1))
const patch_5_con5 = ((4, 4), (0, 0), (0, 0), (1, 1))
const mesh_con_5patch = (
    patch_1_con5, patch_2_con5, patch_3_con5, patch_4_con5, patch_5_con5
)
# All together
const mesh_con_vec = (
    mesh_con_1patch, mesh_con_2patch, mesh_con_3patch, mesh_con_4patch, mesh_con_5patch
)

function create_function_spaces(geo, starting_points, L, num_elements, p, r, mesh_con_vec)
    # Assumes that the number of elements is the same for each patch.
    num_patches = Geometry.get_num_patches(geo)

    Wif = FunctionSpaces.create_multi_patch_c0_space(
        ntuple(Val(num_patches)) do i
            return FunctionSpaces.create_bspline_space(
                starting_points[i],  # Starting points
                (L, L),  # Box sizes
                num_elements,
                (p, p),  # degrees
                (r, r),  # regularities
            )
        end,
        mesh_con_vec[num_patches],
    )
    cftif = FunctionSpaces.create_multi_patch_c0_space(
        ntuple(Val(num_patches)) do i
            return FunctionSpaces.create_bspline_space(
                starting_points[i],  # Starting points
                (L, L),  # Box sizes
                num_elements,
                (p + 1, p + 1),  # degrees
                (r + 1, r + 1),  # regularities
            )
        end,
        mesh_con_vec[num_patches],
    )
    Gif_1 = FunctionSpaces.create_multi_patch_c0_space(
        ntuple(Val(num_patches)) do i
            return FunctionSpaces.create_bspline_space(
                starting_points[i],  # Starting points
                (L, L),  # Box sizes
                num_elements,
                (p, p),  # degrees
                (r - 1, r - 1),  # regularities
            )
        end,
        mesh_con_vec[num_patches],
    )
    Gif_2 = FunctionSpaces.create_multi_patch_c0_space(
        ntuple(Val(num_patches)) do i
            return FunctionSpaces.create_bspline_space(
                starting_points[i],  # Starting points
                (L, L),  # Box sizes
                num_elements,
                (p, p),  # degrees
                (r - 1, r - 1),  # regularities
            )
        end,
        mesh_con_vec[num_patches],
    )
    Qif = FunctionSpaces.create_multi_patch_c0_space(
        ntuple(Val(num_patches)) do i
            return FunctionSpaces.create_bspline_space(
                starting_points[i],  # Starting points
                (L, L),  # Box sizes
                num_elements,
                (p - 1, p - 1),  # degrees
                (r - 1, r - 1),  # regularities
            )
        end,
        mesh_con_vec[num_patches],
    )
    return Wif, cftif, Gif_1, Gif_2, Qif
end

function save_to_csv(
    pfilename,
    hs,
    num_dofs,
    num_dofs_theta,
    p,
    errors_L2,
    errors_H1,
    errors_H2,
    errors_jump,
    errors_theta,
)
    println("Saving to CSV ...")
    df = DataFrame(; h=hs)
    df[!, Symbol("num_dofs_w_p$p")] = num_dofs
    df[!, Symbol("num_dofs_theta_p$p")] = num_dofs_theta
    df[!, Symbol("errors_w_L2_p$p")] = errors_L2
    df[!, Symbol("errors_w_H1_p$p")] = errors_L2 .+ errors_H1
    df[!, Symbol("errors_w_H2_p$p")] = errors_L2 .+ errors_H1 .+ errors_H2
    df[!, Symbol("errors_w_jump_p$p")] = errors_jump
    df[!, Symbol("errors_theta_p$p")] = errors_theta

    return CSV.write(pfilename * ".csv", df)
end
