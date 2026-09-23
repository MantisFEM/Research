using Mantis

# 5-patch car part. Geometry provided by Mario Kapl.
# The geometry is represented by degree 3 B-splines.

const breakpoints = [0.0, 1.0]
const degree = 3
const regularity = 2
const BS = FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D(breakpoints), degree, regularity)
const TP = FunctionSpaces.TensorProductSpace((BS, BS))
const DS = FunctionSpaces.DirectSumSpace((TP, TP))

# Patch 1
const control_points_patch_1_x = vec(
    [
        [0 5 / 6 5 / 3 5 / 2]
        [2 49 / 18 19 / 6 10 / 3]
        [11 / 3 37 / 9 77 / 18 25 / 6]
        [5 5 5 5]
    ]
)
const control_points_patch_1_y = vec(
    [
        [1 / 2 5 / 3 17 / 6 4]
        [-(1 / 6) 10 / 9 5 / 2 4]
        [-(3 / 2) 2 / 9 37 / 18 4]
        [-(7 / 2) -1 3 / 2 4]
    ],
)
const control_points_patch_1 = hcat(control_points_patch_1_x, control_points_patch_1_y)
const geo_patch_1 = Geometry.FEGeometry(TP, control_points_patch_1)

# Patch 2
const control_points_patch_2_x = vec(
    [
        [0 -1 -2 -3]
        [5 / 6 -(1 / 9) -(19 / 18) -2]
        [5 / 3 7 / 9 -(1 / 9) -1]
        [5 / 2 5 / 3 5 / 6 0]
    ],
)
const control_points_patch_2_y = vec(
    [
        [1 / 2 5 / 6 7 / 6 3 / 2]
        [5 / 3 17 / 9 19 / 9 7 / 3]
        [17 / 6 53 / 18 55 / 18 19 / 6]
        [4 4 4 4]
    ],
)
const control_points_patch_2 = hcat(control_points_patch_2_x, control_points_patch_2_y)
const geo_patch_2 = Geometry.FEGeometry(TP, control_points_patch_2)

# Patch 3
const control_points_patch_3_x = vec(
    [
        [0 -2 -(11 / 3) -5]
        [-1 -(25 / 9) -(37 / 9) -5]
        [-2 -(10 / 3) -(13 / 3) -5]
        [-3 -(11 / 3) -(13 / 3) -5]
    ],
)
const control_points_patch_3_y = vec(
    [
        [1 / 2 -(1 / 6) -(3 / 2) -(7 / 2)]
        [5 / 6 1 / 6 -(13 / 18) -(11 / 6)]
        [7 / 6 13 / 18 5 / 18 -(1 / 6)]
        [3 / 2 3 / 2 3 / 2 3 / 2]
    ],
)
const control_points_patch_3 = hcat(control_points_patch_3_x, control_points_patch_3_y)
const geo_patch_3 = Geometry.FEGeometry(TP, control_points_patch_3)

# Patch 4
const control_points_patch_4_x = vec(
    [
        [0 0 0 0]
        [-2 -(16 / 9) -(14 / 9) -(4 / 3)]
        [-(11 / 3) -(28 / 9) -(23 / 9) -2]
        [-5 -4 -3 -2]
    ],
)
const control_points_patch_4_y = vec(
    [
        [1 / 2 -(1 / 6) -(5 / 6) -(3 / 2)]
        [-(1 / 6) -(11 / 18) -(19 / 18) -(3 / 2)]
        [-(3 / 2) -(31 / 18) -(35 / 18) -(13 / 6)]
        [-(7 / 2) -(7 / 2) -(7 / 2) -(7 / 2)]
    ],
)
const control_points_patch_4 = hcat(control_points_patch_4_x, control_points_patch_4_y)
const geo_patch_4 = Geometry.FEGeometry(TP, control_points_patch_4)

# Patch 5
const control_points_patch_5_x = vec(
    [
        [0 2 11 / 3 5]
        [0 16 / 9 28 / 9 4]
        [0 14 / 9 23 / 9 3]
        [0 4 / 3 2 2]
    ]
)
const control_points_patch_5_y = vec(
    [
        [1 / 2 -(1 / 6) -(3 / 2) -(7 / 2)]
        [-(1 / 6) -(11 / 18) -(31 / 18) -(7 / 2)]
        [-(5 / 6) -(19 / 18) -(35 / 18) -(7 / 2)]
        [-(3 / 2) -(3 / 2) -(13 / 6) -(7 / 2)]
    ],
)
const control_points_patch_5 = hcat(control_points_patch_5_x, control_points_patch_5_y)
const geo_patch_5 = Geometry.FEGeometry(TP, control_points_patch_5)

# Multi-patch geometry
control_points_all = [
    control_points_patch_1,
    control_points_patch_2,
    control_points_patch_3,
    control_points_patch_4,
    control_points_patch_5,
]
function create_car_part_geometry()
    return Geometry.MultiPatchGeometry((
        geo_patch_1, geo_patch_2, geo_patch_3, geo_patch_4, geo_patch_5
    ))
end

function create_car_part_geometry(num_elements::Tuple{Int, Int})
    println("Refining Geometry to $(num_elements)-elements")
    return Geometry.MultiPatchGeometry(
        ntuple(5) do patch_id
            # The number of subdivisions equals the requested number of elements since the
            # original space only has 1 element.
            TS_TP, TP_fine = FunctionSpaces.build_two_scale_operator(TP, num_elements)

            # Compute the updated coefficients using the two-scale matrix.
            fine_cps_x = FunctionSpaces.get_child_basis_coefficients(
                control_points_all[patch_id][:, 1], TS_TP
            )
            fine_cps_y = FunctionSpaces.get_child_basis_coefficients(
                control_points_all[patch_id][:, 2], TS_TP
            )

            control_points_patch_i = hcat(fine_cps_x, fine_cps_y)

            return Geometry.FEGeometry(TP_fine, control_points_patch_i)
        end,
    )
end
# Plot.export_geometry_to_vtk(geo_patch_1, "patch-1")
# Plot.export_geometry_to_vtk(geo_patch_2, "patch-2")
# Plot.export_geometry_to_vtk(geo_patch_3, "patch-3")
# Plot.export_geometry_to_vtk(geo_patch_4, "patch-4")
# Plot.export_geometry_to_vtk(geo_patch_5, "patch-5")
# geo = create_car_part_geometry()
# Plot.export_geometry_to_vtk(geo, "car-part")

# geo_refined = create_car_part_geometry((11, 11))
# Plot.export_geometry_to_vtk(geo_refined, "car-part-refined")
