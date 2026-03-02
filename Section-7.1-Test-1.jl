"""
	HodgeLaplacian

This example was used in the sub-section `7.1 Vector Laplace problem` of [Cabanas2025](@cite).
"""
module HodgeLaplacian

using Mantis

using Suppressor: @suppress_err

# Refer to the following file for method and variable definitions
include("HelperFunctions.jl")

############################################################################################
#                                      Problem setup                                       #
############################################################################################
# Mesh
starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
num_elements = (10, 10) # Ininital mesh size.

# B-spline parameters
p = (3, 3) # Polynomial degrees.
k = p .- 1 # Regularities. (Maximally smooth B-splines.)

# Hierarchical parameters.
truncate = true # true = THB, false = HB
simplified = false
num_sub = (2, 2) # Number of subdivisions per dimension per step.

# Create the Hierarchical space for each figure (7.1 a) and 7.1 b)). Still tensor-product
# because we have not refined them.
𝔅_a = @suppress_err Forms.create_hierarchical_de_rham_complex(
    starting_point, box_size, num_elements, p, k, num_sub, truncate, simplified
)
𝔅_b = @suppress_err Forms.create_hierarchical_de_rham_complex(
    starting_point, box_size, num_elements, p, k, num_sub, truncate, simplified
)

# Define the refinement domains
geometry = Geometry.get_base_geometry(Forms.get_geometry(𝔅_a[1]))
marked_elements_a = union(
    get_elements_in_box(geometry, (6, 2), (9, 5)),
    get_elements_in_box(geometry, (4, 4), (7, 7)),
    get_elements_in_box(geometry, (2, 6), (5, 9)),
)
marked_elements_b = union(
    get_elements_in_box(geometry, (4, 2), (9, 5)),
    get_elements_in_box(geometry, (2, 6), (7, 9)),
)

## Figure 7.1 a)
center_element = ceil(Int, prod(num_elements) / 2)
FunctionSpaces.update_space!(Forms.get_fe_space(𝔅_a[1]), [marked_elements_a, Int[]])
ℌ_a = @suppress_err Forms.update_hierarchical_de_rham_complex(𝔅_a, Forms.get_fe_space(𝔅_a[1]))

## Figure 7.1 b)
FunctionSpaces.update_space!(Forms.get_fe_space(𝔅_b[1]), [marked_elements_b, Int[]])
ℌ_b = @suppress_err Forms.update_hierarchical_de_rham_complex(𝔅_b, Forms.get_fe_space(𝔅_b[1]))

# Analytical solution (On each geometry)
f_expr(x) = [fill(2.0, size(x, 1)), zeros(size(x, 1))] # Forcing function
f_a = Forms.AnalyticalFormField(1, f_expr, Forms.get_geometry(ℌ_a[1]), "f")
f_b = Forms.AnalyticalFormField(1, f_expr, Forms.get_geometry(ℌ_b[1]), "f")
u_expr(x) = [x[:, 1] .* (1 .- x[:, 1]), zeros(size(x, 1))] # Vector field
u_a = Forms.AnalyticalFormField(1, u_expr, Forms.get_geometry(ℌ_a[1]), "u")
u_b = Forms.AnalyticalFormField(1, u_expr, Forms.get_geometry(ℌ_b[1]), "u")

VERBOSE = true # Set to true for problem information.
EXPORT_VTK = true # Set to true to export the solutions.

############################################################################################
#                                       Run problem                                        #
############################################################################################

# Quadrature rules
nq_assembly = 2 .* (p .+ 1)
nq_error = nq_assembly .* 2
∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(
    Quadrature.gauss_legendre, nq_assembly, nq_error
)
# Solve problem
dΩ_a, dΩ_b = (
    Quadrature.StandardQuadrature(∫ₐ, Forms.get_num_elements(ℌ_a[1])),
    Quadrature.StandardQuadrature(∫ₐ, Forms.get_num_elements(ℌ_b[1])),
)
δuₕ_a, uₕ_a = @suppress_err Assemblers.solve_one_form_hodge_laplacian(
    ℌ_a[1], ℌ_a[2], f_a, dΩ_a
)
δuₕ_b, uₕ_b = @suppress_err Assemblers.solve_one_form_hodge_laplacian(
    ℌ_b[1], ℌ_b[2], f_b, dΩ_b
)

if VERBOSE
	dΩ_a, dΩ_b = (
		Quadrature.StandardQuadrature(∫ₑ, Forms.get_num_elements(ℌ_a[1])),
		Quadrature.StandardQuadrature(∫ₑ, Forms.get_num_elements(ℌ_b[1])),
	)
    error_a = @suppress_err Analysis.L2_norm(uₕ_a - u_a, dΩ_a)
    error_b = @suppress_err Analysis.L2_norm(uₕ_b - u_b, dΩ_b)
    @show error_a
    @show error_b
end

if EXPORT_VTK
    file_name = "Section-7.1-Test-1"
	u_diff = @suppress_err u_a - uₕ_a
    export_form_fields_to_vtk(
        (uₕ_a, uₕ_b, u_diff), ("uₕ_a", "uₕ_b", "|uₕ_a - u_a|"), file_name
    )
end

end
