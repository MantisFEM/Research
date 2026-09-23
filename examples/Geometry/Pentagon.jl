using Mantis

# 5-patch pentagon
# The geometry is represented by degree 1 B-splines.

const breakpoints = [0.0, 1.0]
const degree = 1
const regularity = 0
const BS = FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D(breakpoints), degree, regularity)
const TP = FunctionSpaces.TensorProductSpace((BS, BS))

# Patch 1
const control_points_patch_1_x = [0.000000000000000, 0.951056516295154, 0.000000000000000, 0.726542528005361]
const control_points_patch_1_y = [0.000000000000000, 0.309016994374947, 1.000000000000000, 1.000000000000000]
const control_points_patch_1 = hcat(control_points_patch_1_x, control_points_patch_1_y)
const geo_patch_1 = Geometry.FEGeometry(TP, control_points_patch_1)

# Patch 2
const control_points_patch_2_x = [0.000000000000000, 0.000000000000000, -0.951056516295154, -0.726542528005361]
const control_points_patch_2_y = [0.000000000000000, 1.000000000000000, 0.309016994374948, 1.000000000000000]
const control_points_patch_2 = hcat(control_points_patch_2_x, control_points_patch_2_y)
const geo_patch_2 = Geometry.FEGeometry(TP, control_points_patch_2)

# Patch 3
const control_points_patch_3_x = [0.000000000000000, -0.951056516295154, -0.587785252292473, -1.175570504584946]
const control_points_patch_3_y = [0.000000000000000, 0.309016994374948, -0.809016994374947, -0.381966011250105]
const control_points_patch_3 = hcat(control_points_patch_3_x, control_points_patch_3_y)
const geo_patch_3 = Geometry.FEGeometry(TP, control_points_patch_3)

# Patch 4
const control_points_patch_4_x = [0.000000000000000, -0.587785252292473, 0.587785252292473, -0.000000000000000]
const control_points_patch_4_y = [0.000000000000000, -0.809016994374947, -0.809016994374948, -1.236067977499790]
const control_points_patch_4 = hcat(control_points_patch_4_x, control_points_patch_4_y)
const geo_patch_4 = Geometry.FEGeometry(TP, control_points_patch_4)

# Patch 5
const control_points_patch_5_x = [0.000000000000000, 0.587785252292473, 0.951056516295154, 1.175570504584946]
const control_points_patch_5_y = [0.000000000000000, -0.809016994374948, 0.309016994374947, -0.381966011250105]
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
function create_pentagon_geometry()
    return Geometry.MultiPatchGeometry((
        geo_patch_1, geo_patch_2, geo_patch_3, geo_patch_4, geo_patch_5
    ))
end

function create_pentagon_geometry(num_elements::Tuple{Int, Int})
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
# Plot.export_geometry_to_vtk(geo_patch_1, "pentagon-patch-1")
# Plot.export_geometry_to_vtk(geo_patch_2, "pentagon-patch-2")
# Plot.export_geometry_to_vtk(geo_patch_3, "pentagon-patch-3")
# Plot.export_geometry_to_vtk(geo_patch_4, "pentagon-patch-4")
# Plot.export_geometry_to_vtk(geo_patch_5, "pentagon-patch-5")
# geo = create_pentagon_geometry()
# Plot.export_geometry_to_vtk(geo, "pentagon")

# geo_refined = create_pentagon_geometry((16, 16))
# Plot.export_geometry_to_vtk(geo_refined, "pentagon-refined")
