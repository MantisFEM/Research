using Mantis
using DataFrames
using CSV
using LinearAlgebra

include("TestCaseSetup.jl")
include("examples/Geometry/Polynomial2Patch.jl")

# Options:
# "AP_TH_p_r_n": WC0 = S^p_{r}, G = [S^p_{r-1} S^p_{r-1}, S^p_{r-1} S^p_{r-1}], Q = S^{p-1}_{r-1}
const case_pre = "AP_TH_2_1_2_polynomial"
const case = "AP_TH_3_2_2_polynomial"
const case_post = "AP_TH_4_3_2_polynomial"
const num_elements_study = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
const save_csv = true  # Save to file?

const L = 1.0
const hs = [L / n[1] for n in num_elements_study]

const starting_points = ((0.0, 0.0), (0.0, 0.0))

# Exact solution:
function exact_sol_func(x::Matrix{Float64})
    return [@. sin(pi * x[:, 1])^2 * sin(pi * x[:, 2])^2]
end
function exact_dsol_func(x::Matrix{Float64})
    return [
        pi .* sin.(2 .* pi .* x[:, 1]) .* sin.(pi .* x[:, 2]) .^ 2,
        pi .* sin.(pi .* x[:, 1]) .^ 2 .* sin.(2 .* pi .* x[:, 2]),
    ]
end
function exact_ddsol_func(x::Matrix{Float64})
    return [
        @. 2 * pi^2 * cos(2 * pi * x[:, 1]) * sin(pi * x[:, 2])^2 +
            2 * pi^2 * sin(pi * x[:, 1])^2 * cos(2 * pi * x[:, 2])
    ]
end
function forcing_function(x::Matrix{Float64})
    return [
        @. -8 * pi^4 * cos(2 * pi * x[:, 1]) * sin(pi * x[:, 2])^2 +
           8 * pi^4 * cos(2 * pi * x[:, 1]) * cos(2 * pi * x[:, 2]) -
            8 * pi^4 * sin(pi * x[:, 1])^2 * cos(2 * pi * x[:, 2])
    ]
end

mapping_obj_patch_1, mapping_obj_patch_2 = create_polynomial_2patch_mappings()
function alpha_patch_1_right_poly(eta::Float64)
    oneeta = [1.0, eta]
    return LinearAlgebra.det(
        [mapping_obj_patch_1.dmapping(oneeta)[:, 1] mapping_obj_patch_1.dmapping(
            oneeta
        )[
            :, 2
        ]],
    )
end
function beta_patch_1_right_poly(eta::Float64)
    oneeta = [1.0, eta]
    tau = sqrt(
        mapping_obj_patch_1.dmapping(oneeta)[1, 2]^2 +
        mapping_obj_patch_1.dmapping(oneeta)[2, 2]^2,
    ) # length of tangent vector
    t_0 = mapping_obj_patch_1.dmapping(oneeta)[:, 2] ./ tau
    return (
        mapping_obj_patch_1.dmapping(oneeta)[1, 1] * t_0[1] +
        mapping_obj_patch_1.dmapping(oneeta)[2, 1] * t_0[2]
    ) / tau
end
# Minus signs on the second patch are filled for outward pointing normals.
function alpha_patch_2_left_poly(eta::Float64)
    zeroeta = [0.0, eta]
    return LinearAlgebra.det(
        [-mapping_obj_patch_2.dmapping(zeroeta)[:, 1] mapping_obj_patch_2.dmapping(
            zeroeta
        )[
            :, 2
        ]],
    )
end
function beta_patch_2_left_poly(eta::Float64)
    zeroeta = [0.0, eta]
    tau = sqrt(
        mapping_obj_patch_2.dmapping(zeroeta)[1, 2]^2 +
        mapping_obj_patch_2.dmapping(zeroeta)[2, 2]^2,
    ) # length of tangent vector
    t_0 = mapping_obj_patch_2.dmapping(zeroeta)[:, 2] ./ tau
    return (
        -mapping_obj_patch_2.dmapping(zeroeta)[1, 1] * t_0[1] +
        -1.0 * mapping_obj_patch_2.dmapping(zeroeta)[2, 1] * t_0[2]
    ) / tau
end
# Create gluing data per patch as obtained from the geometry. Note that
# in case of a boundary edge, the gluing data is set to α = 1 and β = 0
# along the entire edge. Currently specified as two functions (α and β)
# per edge in the same order as the connectivity.
gluing_data_patch_1_poly = (
    (one, zero),             # Bottom edge, boundary
    (alpha_patch_1_right_poly, beta_patch_1_right_poly), # Right edge, shared with patch 2
    (one, zero),             # Top edge, boundary
    (one, zero),
)             # Left edge, boundary
gluing_data_patch_2_poly = (
    (one, zero),           # Bottom edge, boundary
    (one, zero),           # Right edge, boundary
    (one, zero),           # Top edge, boundary
    (alpha_patch_2_left_poly, beta_patch_2_left_poly),
) # Left edge, shared with patch 1
gluing_data_2patch_poly = (
    gluing_data_patch_1_poly, gluing_data_patch_2_poly
)

function run_case(case)
    split_case = split(case, "_")
    p = parse(Int, split_case[3])
    r = parse(Int, split_case[4])
    num_patches = parse(Int, split_case[5])
    which_geometry = split_case[6]
    println(
        "Setting up Ainsworth-Parker problem using $num_patches-patch TH splines on a $which_geometry geometry.",
    )
    println("p = $p and r = $r")

    canonical_qrule = Quadrature.tensor_product_rule(
        (p + 3, p + 3), Quadrature.gauss_legendre
    )
    canonical_qrule_A = Quadrature.tensor_product_rule(
        (3 * p + 1, 3 * p + 1), Quadrature.gauss_legendre
    )

    errors_L2 = Float64[]
    errors_H1 = Float64[]
    errors_jump = Float64[]
    errors_theta = Float64[]
    errors_H2 = Float64[]
    num_dofs = Int[]
    num_dofs_theta = Int[]
    errors_L2_C1 = Float64[]
    errors_H1_C1 = Float64[]
    errors_jump_C1 = Float64[]
    errors_H2_C1 = Float64[]
    num_dofs_C1 = Int[]
    for num_elements in num_elements_study
        println("Current number of elements: ", num_elements)
        # Create geometry
        if num_patches == 2
            tp_spaces = ntuple(Val(num_patches)) do _
                return FunctionSpaces.create_bspline_space(
                    (0.0, 0.0),  # Starting points
                    (L, L),  # Box sizes
                    num_elements,
                    (p, p),  # degrees
                    (r, r),  # regularities
                )
            end
            AC1 = FunctionSpaces.create_approximate_C1_space(
                tp_spaces, mesh_con_vec[num_patches], gluing_data_2patch_poly
            )
        else
            error(
                "The polynomial geometry can only deal with 2 patches, got $num_patches.",
            )
        end
        breakpoints_per_patch = map(
            LinRange, (0.0, 0.0), (0.0, 0.0) .+ (L, L), num_elements .+ 1
        )
        # Single patch cartesian with multiple mappings, so the same parametric
        # geomety.
        geometry_ic = Geometry.CartesianGeometry(breakpoints_per_patch)
        geometry_i = Geometry.MappedGeometry(
            geometry_ic, create_polynomial_2patch_mappings()
        )

        # FunctionSpaces
        Wif, cftif, Gif_1, Gif_2, Qif = create_function_spaces(
            geometry_i, starting_points, L, num_elements, p, r, mesh_con_vec
        )

        Wi = Forms.FormSpace(Val(0), geometry_i, Wif, "w_h")
        Gi = Forms.ModifiedOneFormSpace(
            geometry_i, FunctionSpaces.DirectSumSpace((Gif_1, Gif_2)), "theta_h"
        )
        Qi = Forms.ModifiedVolumeFormSpace(geometry_i, Qif, "q_h")

        f⁰_i = Forms.AnalyticalFormField(Val(0), forcing_function, geometry_i, "f⁰")

        dΩ_i = Quadrature.StandardQuadrature(
            canonical_qrule, Geometry.get_num_elements(geometry_i)
        )
        dΩ_Ai = Quadrature.StandardQuadrature(
            canonical_qrule_A, Geometry.get_num_elements(geometry_i)
        )

        # Exact solutions, also used for the boundary conditions.
        w⁰_exact = Forms.AnalyticalFormField(Val(0), exact_sol_func, geometry_i, "w⁰_exact")
        dw⁰_exact = Forms.AnalyticalFormField(
            Val(1), exact_dsol_func, geometry_i, "dw⁰_exact"
        )
        ddw⁰_exact = Forms.AnalyticalFormField(
            Val(0), exact_ddsol_func, geometry_i, "ddw⁰_exact"
        )

        println("Setting up and solving main problem ...")
        w⁰_h, θ, zp, q = Assemblers.solve_Ainsworth_Parker_Taylor_Hood(
            Wi, Gi, Qi, false, dΩ_i, f⁰_i; clamped=true
        )

        error_L2i = Analysis.L2_norm(w⁰_h - w⁰_exact, dΩ_Ai)
        append!(errors_L2, error_L2i)
        println("\tL^2 error: ", error_L2i)
        error_H1i = Analysis.L2_norm(d(w⁰_h) - dw⁰_exact, dΩ_Ai)
        append!(errors_H1, error_H1i)
        println("\tH^1 error: ", error_L2i + error_H1i)
        error_jumpi = Analysis.compute_max_jump_1form(d(w⁰_h), mesh_con_vec[num_patches])
        append!(errors_jump, error_jumpi)
        error_thetai = Analysis.L2_norm(θ - dw⁰_exact, dΩ_Ai)
        println("\ttheta error: ", error_thetai)
        append!(errors_theta, error_thetai)
        println("\tmax jump: ", error_jumpi)
        error_H2i = Analysis.L2_norm(δ(d(w⁰_h)) - ddw⁰_exact, dΩ_Ai)
        append!(errors_H2, error_H2i)
        println("\tΔ error: ", error_L2i + error_H1i + error_H2i)
        append!(num_dofs, Forms.get_num_basis(Wi))
        append!(num_dofs_theta, Forms.get_num_basis(Gi))

        if p > 2
            println("\tComputing approximate C1 solution...")
            WC1 = Forms.FormSpace(Val(0), geometry_i, AC1, "w_h_C1")
            w⁰_h_c1 = Assemblers.solve_zero_form_biharmonic(WC1, f⁰_i, dΩ_i)

            error_L2i = Analysis.L2_norm(w⁰_h_c1 - w⁰_exact, dΩ_Ai)
            append!(errors_L2_C1, error_L2i)
            println("\tL^2 error: ", error_L2i)
            error_H1i = Analysis.L2_norm(d(w⁰_h_c1) - dw⁰_exact, dΩ_Ai)
            append!(errors_H1_C1, error_H1i)
            println("\tH^1 error: ", error_H1i)
            error_jumpi = Analysis.compute_max_jump_1form(d(w⁰_h_c1))
            append!(errors_jump_C1, error_jumpi)
            println("\tmax jump: ", error_jumpi)
            error_H2i = Analysis.L2_norm(δ(d(w⁰_h_c1)) - ddw⁰_exact, dΩ_Ai)
            append!(errors_H2_C1, error_H2i)
            println("\tΔ error: ", error_L2i + error_H1i + error_H2i)
            append!(num_dofs_C1, Forms.get_num_basis(WC1))
        end
    end

    if save_csv
        pfilename = "TestCase2-nels$(num_elements_study)-p$(p)-r$(r)"
        save_to_csv(
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

        if p > 2
            pfilenameC1 = "TestCase2C1-nels$(num_elements_study)-p$(p)-r$(r)"
            save_to_csv(
                pfilenameC1,
                hs,
                num_dofs_C1,
                num_dofs_theta,
                p,
                errors_L2_C1,
                errors_H1_C1,
                errors_H2_C1,
                errors_jump_C1,
                errors_theta,
            )
        end
    end
end

run_case(case_pre)
run_case(case)
run_case(case_post)
