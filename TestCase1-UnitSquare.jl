using Mantis
using DataFrames
using CSV
using LinearAlgebra

include("TestCaseSetup.jl")

# Options:
# "AP_TH_p_r_n": WC0 = S^p_{r}, G = [S^p_{r-1} S^p_{r-1}, S^p_{r-1} S^p_{r-1}], Q = S^{p-1}_{r-1}
const case1 = "AP_TH_3_2_4_cartesianunitsquare"
const case2 = "AP_TH_5_4_4_cartesianunitsquare"
const num_elements_study = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
const save_csv = true  # Save to file?

const L = 0.5
const hs = [L / n[1] for n in num_elements_study]
const starting_points = ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))

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
    errors_cft = Float64[]
    errors_H2 = Float64[]
    num_dofs = Int[]
    num_dofs_theta = Int[]
    for num_elements in num_elements_study
        println("Current number of elements: ", num_elements)
        # Create geometry
        if L != 0.5
            error("The domain length is not 0.5, but $L. Please change this.")
        end
        breakpoints_per_patch = (
            map(
                LinRange,
                starting_points[1],
                starting_points[1] .+ (L, L),
                num_elements .+ 1,
            ),
            map(
                LinRange,
                starting_points[2],
                starting_points[2] .+ (L, L),
                num_elements .+ 1,
            ),
            map(
                LinRange,
                starting_points[3],
                starting_points[3] .+ (L, L),
                num_elements .+ 1,
            ),
            map(
                LinRange,
                starting_points[4],
                starting_points[4] .+ (L, L),
                num_elements .+ 1,
            ),
        )
        geometry_i = Geometry.CartesianGeometry(
            ntuple(Val(num_patches)) do i
                return breakpoints_per_patch[i]
            end,
        )

        # Function Spaces
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
    end

    pfilename = "TestCase1-nels$(num_elements_study)-p$(p)-r$(r)"
    if save_csv
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
    end
end

run_case(case1)
run_case(case2)
