using Mantis
using DataFrames
using CSV
using LinearAlgebra

include("TestCaseSetup.jl")

# Options:
# "AP_TH_p_r_n":
# WC0 = S^p_{r}, G = [S^p_{r-1} S^p_{r-1}, S^p_{r-1} S^p_{r-1}], Q = S^{p-1}_{r-1}
const case_pre = "AP_TH_2_1_1_cartesianl2"
const case = "AP_TH_3_2_1_cartesianl2"
const case_post = "AP_TH_4_3_1_cartesianl2"
const case_post2 = "AP_TH_5_4_1_cartesianl2"
const case_post3 = "AP_TH_6_5_1_cartesianl2"
const pz = 1 # 1 or 2, corresponding to table 2 or 3, respectively.
const num_elements_study = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
const save_csv = true  # Save to file?

const L = 1.0
const hs = [L / n[1] for n in num_elements_study]

# Problem data
function exact_sol_func(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [[sin(pi*x)^4*sin(pi*y)^4 for (x,y) in zip(xs, ys)]]
end
function exact_dsol_func(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [
        [4*pi*cos(pi*x)*sin(pi*x)^3*sin(pi*y)^4 for (x,y) in zip(xs, ys)],
        [4*pi*cos(pi*y)*sin(pi*x)^4*sin(pi*y)^3 for (x,y) in zip(xs, ys)],
    ]
end
function exact_ddsol_func(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [[4*pi^2*sin(pi*x)^2*sin(pi*y)^2*(3*sin(pi*x)^2 - 8*sin(pi*x)^2*sin(pi*y)^2 + 3*sin(pi*y)^2) for (x,y) in zip(xs, ys)]]
end
function exact_dddsol_func(xx::Matrix{Float64}) # grad laplacian
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [
        [8*pi^3*cos(pi*x)*sin(pi*x)*sin(pi*y)^2*(6*sin(pi*x)^2 - 16*sin(pi*x)^2*sin(pi*y)^2 + 3*sin(pi*y)^2) for (x,y) in zip(xs, ys)],
        [8*pi^3*cos(pi*y)*sin(pi*x)^2*sin(pi*y)*(3*sin(pi*x)^2 - 16*sin(pi*x)^2*sin(pi*y)^2 + 6*sin(pi*y)^2) for (x,y) in zip(xs, ys)],
    ]
end
function forcing_function(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [[8*pi^4*(36*sin(pi*x)^2*sin(pi*y)^2 - 78*sin(pi*x)^2*sin(pi*y)^4 - 78*sin(pi*x)^4*sin(pi*y)^2 + 128*sin(pi*x)^4*sin(pi*y)^4 + 3*sin(pi*x)^4 + 3*sin(pi*y)^4) for (x,y) in zip(xs, ys)]]
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
    canonical_qrule = Quadrature.tensor_product_rule((12, 12), Quadrature.gauss_legendre)
    canonical_qrule_A = Quadrature.tensor_product_rule((16, 16), Quadrature.gauss_legendre)

    errors_L2 = Float64[]
    errors_H1 = Float64[]
    errors_L2_z = Float64[]
    errors_H1_z = Float64[]
    errors_jump = Float64[]
    errors_theta = Float64[]
    errors_theta_H1 = Float64[]
    errors_theta_Hc = Float64[]
    errors_theta_Hd = Float64[]
    errors_H2 = Float64[]
    num_dofs = Int[]
    num_dofs_theta = Int[]
    num_dofs_z = Int[]

    for num_elements in num_elements_study
        println("Current number of elements: ", num_elements)

        # Create geometry
        geometry_i = Geometry.create_curvilinear_square(
            (0.0, 0.0), (L, L), num_elements; crazy_c=0.2
        )

        starting_points = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        Wif, Gif_1, Gif_2, Qif, Zif = create_function_spaces(
            geometry_i, starting_points, num_elements, p, r, pz
        )

        Wi = Forms.FormSpace(Val(0), geometry_i, Wif, "w_h")
        Zi = Forms.FormSpace(Val(0), geometry_i, Zif, "z_h")
        Gi = Forms.ModifiedOneFormSpace(
            geometry_i, FunctionSpaces.DirectSumSpace((Gif_1, Gif_2)), "theta_h"
        )
        Qi = Forms.ModifiedVolumeFormSpace(geometry_i, Qif, "q_h")

        dΩ_i = Quadrature.StandardQuadrature(
            canonical_qrule, Geometry.get_num_elements(geometry_i)
        )

        f⁰_i = Forms.AnalyticalFormField(Val(0), forcing_function, geometry_i, "f⁰")
        dΩ_Ai = Quadrature.StandardQuadrature(
            canonical_qrule_A, Geometry.get_num_elements(geometry_i)
        )

        # Exact solutions
        w⁰_exact = Forms.AnalyticalFormField(Val(0), exact_sol_func, geometry_i, "w⁰_exact")
        dw⁰_exact = Forms.AnalyticalFormField(
            Val(1), exact_dsol_func, geometry_i, "dw⁰_exact"
        )
        ddw⁰_exact = Forms.AnalyticalFormField(
            Val(0), exact_ddsol_func, geometry_i, "ddw⁰_exact"
        )
        dddw⁰_exact = Forms.AnalyticalFormField(
            Val(1), exact_dddsol_func, geometry_i, "dddw⁰_exact"
        )

        println("Setting up and solving main problem ...")
        w⁰_h, θ, zp, q = Assemblers.solve_Ainsworth_Parker_Taylor_Hood(
            Wi,
            Gi,
            Qi,
            false,
            dΩ_i,
            f⁰_i,
            Zi,
            false,
            clamped=true,
            bc_w_coeffs=zeros(Forms.get_num_basis(Wi)),
            bc_theta=zeros(Forms.get_num_basis(Gi)),
        )

        error_L2i = Analysis.L2_norm(w⁰_h - w⁰_exact, dΩ_Ai)
        append!(errors_L2, error_L2i)
        println("\tL^2 error: ", error_L2i)
        error_H1i = Analysis.L2_norm(d(w⁰_h) - dw⁰_exact, dΩ_Ai)
        if num_elements != num_elements_study[1]
            println("\tEstimated rate of convergence: $(round(log(errors_L2[end-1]/errors_L2[end])/log(hs[end-1]/hs[end]), digits=2))")
        end
        append!(errors_H1, error_H1i)
        println("\tH^1 error: ", error_L2i + error_H1i)
        error_jumpi = Analysis.compute_max_jump_1form(d(w⁰_h), mesh_con_vec[num_patches])
        if num_elements != num_elements_study[1]
            println("\tEstimated rate of convergence: $(round(log(errors_H1[end-1]/errors_H1[end])/log(hs[end-1]/hs[end]), digits=2))")
        end
        append!(errors_jump, error_jumpi)

        error_L2zi = Analysis.L2_norm(zp - (-ddw⁰_exact), dΩ_Ai)
        append!(errors_L2_z, error_L2zi)
        println("\tz L^2 error: ", error_L2zi)
        if num_elements != num_elements_study[1]
            println("\tEstimated rate of convergence: $(round(log(errors_L2_z[end-1]/errors_L2_z[end])/log(hs[end-1]/hs[end]), digits=2))")
        end
        error_H1zi = Analysis.L2_norm(d(zp) - (-dddw⁰_exact), dΩ_Ai)
        append!(errors_H1_z, error_H1zi)
        println("\tz H^1 error: ", error_H1zi)
        if num_elements != num_elements_study[1]
            println("\tEstimated rate of convergence: $(round(log(errors_H1_z[end-1]/errors_H1_z[end])/log(hs[end-1]/hs[end]), digits=2))")
        end

        error_thetai = Analysis.L2_norm(θ - dw⁰_exact, dΩ_Ai)
        println("\ttheta L^2 error: ", error_thetai)
        append!(errors_theta, error_thetai)
        error_thetaHdi = Analysis.L2_norm(δ(θ) - ddw⁰_exact, dΩ_Ai)
        println("\ttheta H(div) error: ", error_thetai+error_thetaHdi)
        println("\ttheta H(div) semi error: ", error_thetaHdi)
        append!(errors_theta_Hd, error_thetaHdi)
        error_thetaHci = Analysis.L2_norm(d(θ), dΩ_Ai)
        println("\ttheta H(curl) error: ", error_thetai+error_thetaHci)
        println("\ttheta H(curl) semi error: ", error_thetaHci)
        append!(errors_theta_Hc, error_thetaHci)
        println("\ttheta H1 error: ", error_thetai + error_thetaHci + error_thetaHdi)
        println("\ttheta H1 semi error: ", error_thetaHci + error_thetaHdi)
        append!(errors_theta_H1, error_thetaHci + error_thetaHdi)
        if num_elements != num_elements_study[1]
            println("\tEstimated rate of convergence L2: $(round(log(errors_theta[end-1]/errors_theta[end])/log(hs[end-1]/hs[end]), digits=2))")
            println("\tEstimated rate of convergence H(div): $(round(log(errors_theta_Hd[end-1]/errors_theta_Hd[end])/log(hs[end-1]/hs[end]), digits=2))")
            println("\tEstimated rate of convergence H(curl): $(round(log(errors_theta_Hc[end-1]/errors_theta_Hc[end])/log(hs[end-1]/hs[end]), digits=2))")
            println("\tEstimated rate of convergence H1: $(round(log(errors_theta_H1[end-1]/errors_theta_H1[end])/log(hs[end-1]/hs[end]), digits=2))")
        end
        println("\tmax jump: ", error_jumpi)
        error_H2i = Analysis.L2_norm(δ(d(w⁰_h)) - ddw⁰_exact, dΩ_Ai)
        append!(errors_H2, error_H2i)
        println("\tΔ error: ", error_L2i + error_H1i + error_H2i)
        if num_elements != num_elements_study[1]
            println("\tEstimated rate of convergence: $(round(log(errors_H2[end-1]/errors_H2[end])/log(hs[end-1]/hs[end]), digits=2))")
        end
        append!(num_dofs, Forms.get_num_basis(Wi))
        append!(num_dofs_theta, Forms.get_num_basis(Gi))
        append!(num_dofs_z, Forms.get_num_basis(Zi))
    end

    pfilename = "TestCase5-zhdegree$(pz)-nels$(num_elements_study)-p$(p)-r$(r)"
    if save_csv
        save_to_csv(
            pfilename,
            hs,
            num_dofs,
            num_dofs_theta,
            num_dofs_z,
            p,
            pz,
            errors_L2,
            errors_H1,
            errors_H2,
            errors_jump,
            errors_theta,
            errors_theta_H1,
            errors_theta_Hc,
            errors_theta_Hd,
            errors_L2_z,
            errors_H1_z,
        )
    end

    return nothing
end

function create_function_spaces(geo, starting_points, num_elements, p, r, pz)
    Zif = FunctionSpaces.create_bspline_space(
        starting_points[1],  # Starting points
        (L, L),  # Box sizes
        num_elements,
        (pz, pz),  # degrees
        (pz-1, pz-1),  # regularities
    )
    Wif = FunctionSpaces.create_bspline_space(
        starting_points[1],  # Starting points
        (L, L),  # Box sizes
        num_elements,
        (p, p),  # degrees
        (r, r),  # regularities
    )
    # Create single patch spline space G.
    Gif_1 = FunctionSpaces.create_bspline_space(
        starting_points[1],  # Starting points
        (L, L),  # Box sizes
        num_elements,
        (p, p),  # degrees
        (r - 1, r - 1),  # regularities
    )
    Gif_2 = FunctionSpaces.create_bspline_space(
        starting_points[1],  # Starting points
        (L, L),  # Box sizes
        num_elements,
        (p, p),  # degrees
        (r - 1, r - 1),  # regularities
    )
    # Create single patch spline space Q.
    Qif = FunctionSpaces.create_bspline_space(
        starting_points[1],  # Starting points
        (L, L),  # Box sizes
        num_elements,
        (p - 1, p - 1),  # degrees
        (r - 1, r - 1),  # regularities
    )
    return Wif, Gif_1, Gif_2, Qif, Zif
end

function save_to_csv(
    pfilename,
    hs,
    num_dofs,
    num_dofs_theta,
    num_dofs_z,
    p,
    pz,
    errors_L2,
    errors_H1,
    errors_H2,
    errors_jump,
    errors_theta,
    errors_theta_H1,
    errors_theta_Hc,
    errors_theta_Hd,
    errors_z_L2,
    errors_z_H1,
)
    println("Creating table ...")
    println("p = $p")
    dfp = DataFrame(; h=hs)
    dfp[!, Symbol("ew_L2")] = errors_L2
    dfp[!, Symbol("r_L2")] = vcat(0, [round(log(errors_L2[i]/errors_L2[i+1])/log(hs[i]/hs[i+1]), digits=2) for i in eachindex(errors_L2)[1:end-1]])
    dfp[!, Symbol("ew_H1")] = errors_L2 .+ errors_H1
    errH1full = errors_L2 .+ errors_H1
    dfp[!, Symbol("r_H1")] = vcat(0, [round(log(errH1full[i]/errH1full[i+1])/log(hs[i]/hs[i+1]), digits=2) for i in eachindex(errH1full)[1:end-1]])
    dfp[!, Symbol("ew_H2")] = errors_L2 .+ errors_H1 .+ errors_H2
    errH2full = errors_L2 .+ errors_H1 .+ errors_H2
    dfp[!, Symbol("r_H2")] = vcat(0, [round(log(errH2full[i]/errH2full[i+1])/log(hs[i]/hs[i+1]), digits=2) for i in eachindex(errH2full)[1:end-1]])


    dfp[!, Symbol("et_L2")] = errors_theta
    dfp[!, Symbol("rt_L2")] = vcat(0, [round(log(errors_theta[i]/errors_theta[i+1])/log(hs[i]/hs[i+1]), digits=2) for i in eachindex(errors_theta)[1:end-1]])
    dfp[!, Symbol("et_H1")] = errors_theta .+ errors_theta_H1
    er_t_h1 = errors_theta .+ errors_theta_H1
    dfp[!, Symbol("rt_H1")] = vcat(0, [round(log(er_t_h1[i]/er_t_h1[i+1])/log(hs[i]/hs[i+1]), digits=2) for i in eachindex(er_t_h1)[1:end-1]])

    dfp[!, Symbol("ez_L2")] = errors_z_L2
    dfp[!, Symbol("rz_L2")] = vcat(0, [round(log(errors_z_L2[i]/errors_z_L2[i+1])/log(hs[i]/hs[i+1]), digits=2) for i in eachindex(errors_z_L2)[1:end-1]])
    dfp[!, Symbol("ez_H1")] = errors_z_L2 .+ errors_z_H1
    errzH1full = errors_z_L2 .+ errors_z_H1
    dfp[!, Symbol("rz_H1")] = vcat(0, [round(log(errzH1full[i]/errzH1full[i+1])/log(hs[i]/hs[i+1]), digits=2) for i in eachindex(errzH1full)[1:end-1]])

    show(stdout, MIME("text/latex"), dfp)

    println("Saving to CSV ...")
    df = DataFrame(; h=hs)
    df[!, Symbol("num_dofs_w_p$p")] = num_dofs
    df[!, Symbol("num_dofs_theta_p$p")] = num_dofs_theta
    df[!, Symbol("num_dofs_z_p$pz")] = num_dofs_z
    df[!, Symbol("errors_w_L2_p$p")] = errors_L2
    df[!, Symbol("errors_w_H1_p$p")] = errors_L2 .+ errors_H1
    df[!, Symbol("errors_w_H2_p$p")] = errors_L2 .+ errors_H1 .+ errors_H2
    df[!, Symbol("errors_w_jump_p$p")] = errors_jump
    df[!, Symbol("errors_theta_p$p")] = errors_theta
    df[!, Symbol("errors_z_L2_p$pz")] = errors_z_L2
    df[!, Symbol("errors_z_H1_p$pz")] = errors_z_L2 .+ errors_z_H1

    return CSV.write(pfilename * ".csv", df)
end

run_case(case_pre)
run_case(case)
run_case(case_post)
run_case(case_post2)
run_case(case_post3)
