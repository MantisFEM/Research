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
const num_elements_study = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
const save_csv = true  # Save to file?

const L = 1.0
const hs = [L / n[1] for n in num_elements_study]

# Problem data
function exact_sol_func(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]

    return [[abs(x-0.5)^(7/2) * x^2*(1 - x)^2 * y^2*(1 - y)^2 for (x,y) in zip(xs, ys)]]
end
function exact_dsol_func(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [
        [x^2*y^2*abs(x - 1/2)^(7/2)*(2*x - 2)*(y - 1)^2 + 2*x*y^2*abs(x - 1/2)^(7/2)*(x - 1)^2*(y - 1)^2 + (7*x^2*y^2*abs(x - 1/2)^(5/2)*(x - 1)^2*(y - 1)^2*(x + conj(x) - 1))/(4*((conj(x) - 1/2)*(x - 1/2))^(1/2)) for (x,y) in zip(xs, ys)],
        [x^2*y^2*abs(x - 1/2)^(7/2)*(2*y - 2)*(x - 1)^2 + 2*x^2*y*abs(x - 1/2)^(7/2)*(x - 1)^2*(y - 1)^2 for (x,y) in zip(xs, ys)],
    ]
end
function exact_ddsol_func(xx::Matrix{Float64}) # laplacian
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [[2*x^2*abs(x - 1/2)^(7/2)*(x - 1)^2*(y - 1)^2 + 2*y^2*abs(x - 1/2)^(7/2)*(x - 1)^2*(y - 1)^2 + 2*x^2*y^2*abs(x - 1/2)^(7/2)*(x - 1)^2 + 2*x^2*y^2*abs(x - 1/2)^(7/2)*(y - 1)^2 + 4*x*y^2*abs(x - 1/2)^(7/2)*(2*x - 2)*(y - 1)^2 + 4*x^2*y*abs(x - 1/2)^(7/2)*(2*y - 2)*(x - 1)^2 + (7*x^2*y^2*abs(x - 1/2)^(5/2)*(x - 1)^2*(y - 1)^2)/(2*((conj(x) - 1/2)*(x - 1/2))^(1/2)) + (7*x*y^2*abs(x - 1/2)^(5/2)*(x - 1)^2*(y - 1)^2*(x + conj(x) - 1))/((conj(x) - 1/2)*(x - 1/2))^(1/2) + (7*x^2*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(y - 1)^2*(x + conj(x) - 1))/(2*((conj(x) - 1/2)*(x - 1/2))^(1/2)) - (7*x^2*y^2*abs(x - 1/2)^(5/2)*(x - 1)^2*(y - 1)^2*(x + conj(x) - 1)^2)/(8*((conj(x) - 1/2)*(x - 1/2))^(3/2)) + (35*x^2*y^2*abs(x - 1/2)^(3/2)*(x - 1)^2*(y - 1)^2*(x + conj(x) - 1)^2)/(16*(conj(x) - 1/2)*(x - 1/2)) for (x,y) in zip(xs, ys)]]
end

function exact_dddsol_func(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [
        [2*x^2*abs(x - 1/2)^(7/2)*(2*x - 2)*(y - 1)^2 + 6*y^2*abs(x - 1/2)^(7/2)*(2*x - 2)*(y - 1)^2 + 4*x*y^2*abs(x - 1/2)^(7/2)*(x - 1)^2 + 12*x*y^2*abs(x - 1/2)^(7/2)*(y - 1)^2 + 2*x^2*y^2*abs(x - 1/2)^(7/2)*(2*x - 2) + 4*x*abs(x - 1/2)^(7/2)*(x - 1)^2*(y - 1)^2 + 8*x*y*abs(x - 1/2)^(7/2)*(2*y - 2)*(x - 1)^2 + 4*x^2*y*abs(x - 1/2)^(7/2)*(2*x - 2)*(2*y - 2) + (7*x^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(x - 1)^2*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (21*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(x - 1)^2*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (21*x^2*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (7*x^2*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(x - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (21*x^2*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (42*x*y^2*abs(x - 1/2)^(5/2)*(x - 1)^2*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (42*x*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*real(x) - 1)*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (14*x^2*y*abs(x - 1/2)^(5/2)*(2*y - 2)*(2*real(x) - 1)*(x - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) - (35*x^3*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)*(x - 1)^2*(y - 1)^2)/(2*(2*x - 1)*(x - 2*abs(x)^2)) - (35*x^3*y^2*abs(x - 1/2)^(3/2)*(8*real(x) - 4)*(x - 1)^2*(y - 1)^2)/(4*(2*x - 1)*(x - 2*abs(x)^2)) - (105*x^3*y^2*abs(x - 1/2)^(3/2)*(2*x - 2)*(2*real(x) - 1)^2*(y - 1)^2)/(4*(2*x - 1)*(x - 2*abs(x)^2)) - (105*x^2*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/(2*(2*x - 1)*(x - 2*abs(x)^2)) + (35*x^3*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/(2*(2*x - 1)^2*(x - 2*abs(x)^2)) - (35*x^4*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/(2*(2*x - 1)*(x - 2*abs(x)^2)^2) - (35*x^4*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^3*(x - 1)^2*(y - 1)^2)/(2*(2*x - 1)^2*(x - 2*abs(x)^2)^2) + (14*x^3*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (7*x^3*y^2*abs(x - 1/2)^(5/2)*(8*real(x) - 4)*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (21*x^3*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*real(x) - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) - (105*x^3*y^2*abs(x - 1/2)^(1/2)*(2*real(x) - 1)^3*(x - 1)^2*(y - 1)^2)/(8*(2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (42*x^2*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (42*x^4*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)^3*(x - 1)^2*(y - 1)^2)/((2*x - 1)^2*(x - 2*abs(x)^2)^2*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) for (x,y) in zip(xs, ys)],
        [6*x^2*abs(x - 1/2)^(7/2)*(2*y - 2)*(x - 1)^2 + 2*y^2*abs(x - 1/2)^(7/2)*(2*y - 2)*(x - 1)^2 + 12*x^2*y*abs(x - 1/2)^(7/2)*(x - 1)^2 + 4*x^2*y*abs(x - 1/2)^(7/2)*(y - 1)^2 + 2*x^2*y^2*abs(x - 1/2)^(7/2)*(2*y - 2) + 4*y*abs(x - 1/2)^(7/2)*(x - 1)^2*(y - 1)^2 + 8*x*y*abs(x - 1/2)^(7/2)*(2*x - 2)*(y - 1)^2 + 4*x*y^2*abs(x - 1/2)^(7/2)*(2*x - 2)*(2*y - 2) + (7*x^2*y^2*abs(x - 1/2)^(5/2)*(2*y - 2)*(x - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (14*x^2*y*abs(x - 1/2)^(5/2)*(x - 1)^2*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (28*x*y*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(x - 1)^2*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (14*x*y^2*abs(x - 1/2)^(5/2)*(2*y - 2)*(2*real(x) - 1)*(x - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (14*x^2*y*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*real(x) - 1)*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (7*x^2*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*y - 2)*(2*real(x) - 1))/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) - (35*x^3*y*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/(2*(2*x - 1)*(x - 2*abs(x)^2)) - (35*x^3*y^2*abs(x - 1/2)^(3/2)*(2*y - 2)*(2*real(x) - 1)^2*(x - 1)^2)/(4*(2*x - 1)*(x - 2*abs(x)^2)) + (14*x^3*y*abs(x - 1/2)^(5/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (7*x^3*y^2*abs(x - 1/2)^(5/2)*(2*y - 2)*(2*real(x) - 1)^2*(x - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) for (x,y) in zip(xs, ys)],
    ]
end

function scaling(x, num_elements_x, element_id, scale_on_elements_left, scale_on_elements_right, geo, xi, idx)
    if element_id in scale_on_elements_left
        return 1.0 / (1 - xi[idx][1])^(-0.5)
    elseif element_id in scale_on_elements_right
        return 1.0 / (xi[idx][1])^(-0.5)
    else
        return 1.0
    end
end

function forcing_function_with_scaling(xx::Matrix{Float64}, num_elements_x, element_id, scale_on_elements_left, scale_on_elements_right, geo, xi)
    xs = xx[:, 1]
    ys = xx[:, 2]

    return [[scaling(x, num_elements_x, element_id, scale_on_elements_left, scale_on_elements_right, geo, xi, idx)*(24*x^2*abs(x - 1/2)^(7/2)*(x - 1)^2 + 8*x^2*abs(x - 1/2)^(7/2)*(y - 1)^2 + 8*y^2*abs(x - 1/2)^(7/2)*(x - 1)^2 + 24*y^2*abs(x - 1/2)^(7/2)*(y - 1)^2 + 8*abs(x - 1/2)^(7/2)*(x - 1)^2*(y - 1)^2 + 8*x^2*y^2*abs(x - 1/2)^(7/2) + 16*x*y^2*abs(x - 1/2)^(7/2)*(2*x - 2) + 16*x^2*y*abs(x - 1/2)^(7/2)*(2*y - 2) + 16*x*abs(x - 1/2)^(7/2)*(2*x - 2)*(y - 1)^2 + 16*y*abs(x - 1/2)^(7/2)*(2*y - 2)*(x - 1)^2 + (28*x^2*y^2*abs(x - 1/2)^(5/2)*(x - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (84*x^2*y^2*abs(x - 1/2)^(5/2)*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + 32*x*y*abs(x - 1/2)^(7/2)*(2*x - 2)*(2*y - 2) + (28*x^2*abs(x - 1/2)^(5/2)*(x - 1)^2*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (84*y^2*abs(x - 1/2)^(5/2)*(x - 1)^2*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (28*x^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*real(x) - 1)*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (84*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*real(x) - 1)*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (56*x*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(x - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (168*x*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (28*x^2*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*real(x) - 1))/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (56*x*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(x - 1)^2*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (168*x*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(y - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (56*x^2*y*abs(x - 1/2)^(5/2)*(2*y - 2)*(x - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) + (112*x*y*abs(x - 1/2)^(5/2)*(2*y - 2)*(2*real(x) - 1)*(x - 1)^2)/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) - (35*x^3*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(x - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) - (105*x^3*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) - (105*x^3*y^2*abs(x - 1/2)^(3/2)*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) + (56*x^2*y*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*y - 2)*(2*real(x) - 1))/(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2) - (35*x^3*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) - (70*x^3*y*abs(x - 1/2)^(3/2)*(2*y - 2)*(2*real(x) - 1)^2*(x - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) - (70*x^3*y^2*abs(x - 1/2)^(3/2)*(2*x - 2)*(2*real(x) - 1)*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) - (35*x^3*y^2*abs(x - 1/2)^(3/2)*(2*x - 2)*(8*real(x) - 4)*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) - (105*x*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) - (140*x^2*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) - (70*x^2*y^2*abs(x - 1/2)^(3/2)*(8*real(x) - 4)*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) - (210*x^2*y^2*abs(x - 1/2)^(3/2)*(2*x - 2)*(2*real(x) - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)) + (105*x^3*y^2*abs(x - 1/2)^(3/2)*(2*x - 2)*(2*real(x) - 1)^2*(y - 1)^2)/(2*(2*x - 1)^2*(x - 2*abs(x)^2)) - (105*x^4*y^2*abs(x - 1/2)^(3/2)*(2*x - 2)*(2*real(x) - 1)^2*(y - 1)^2)/(2*(2*x - 1)*(x - 2*abs(x)^2)^2) - (105*x^4*y^2*abs(x - 1/2)^(3/2)*(2*x - 2)*(2*real(x) - 1)^3*(y - 1)^2)/((2*x - 1)^2*(x - 2*abs(x)^2)^2) + (105*x^2*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)^2*(x - 2*abs(x)^2)) - (105*x^3*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)^2) - (210*x^3*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^3*(x - 1)^2*(y - 1)^2)/((2*x - 1)^2*(x - 2*abs(x)^2)^2) + (105*x^4*y^2*(2*real(x) - 1)^4*(x - 1)^2*(y - 1)^2)/(16*abs(x - 1/2)^(1/2)*(2*x - 1)^2*(x - 2*abs(x)^2)^2) - (420*x^4*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)^2*(x - 2*abs(x)^2)^2) + (140*x^4*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^3*(x - 1)^2*(y - 1)^2)/((2*x - 1)^3*(x - 2*abs(x)^2)^2) - (140*x^5*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^3*(x - 1)^2*(y - 1)^2)/((2*x - 1)^2*(x - 2*abs(x)^2)^3) - (245*x^5*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)^4*(x - 1)^2*(y - 1)^2)/((2*x - 1)^3*(x - 2*abs(x)^2)^3) + (28*x^3*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)^2*(x - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (84*x^3*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (84*x^3*y^2*abs(x - 1/2)^(5/2)*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (28*x^3*abs(x - 1/2)^(5/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (56*x^3*y*abs(x - 1/2)^(5/2)*(2*y - 2)*(2*real(x) - 1)^2*(x - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (56*x^3*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*real(x) - 1)*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (28*x^3*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(8*real(x) - 4)*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (84*x*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (112*x^2*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (56*x^2*y^2*abs(x - 1/2)^(5/2)*(8*real(x) - 4)*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) - (105*x^3*y^2*abs(x - 1/2)^(1/2)*(2*x - 2)*(2*real(x) - 1)^3*(y - 1)^2)/(2*(2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (168*x^2*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*real(x) - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (168*x^4*y^2*abs(x - 1/2)^(5/2)*(2*x - 2)*(2*real(x) - 1)^3*(y - 1)^2)/((2*x - 1)^2*(x - 2*abs(x)^2)^2*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) - (105*x^2*y^2*abs(x - 1/2)^(1/2)*(2*real(x) - 1)^3*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) - (105*x^3*y^2*abs(x - 1/2)^(1/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) - (315*x^4*y^2*abs(x - 1/2)^(1/2)*(2*real(x) - 1)^4*(x - 1)^2*(y - 1)^2)/(2*(2*x - 1)^2*(x - 2*abs(x)^2)^2*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (336*x^3*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)^3*(x - 1)^2*(y - 1)^2)/((2*x - 1)^2*(x - 2*abs(x)^2)^2*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (336*x^4*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)^2*(x - 1)^2*(y - 1)^2)/((2*x - 1)^2*(x - 2*abs(x)^2)^2*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (420*x^5*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)^4*(x - 1)^2*(y - 1)^2)/((2*x - 1)^3*(x - 2*abs(x)^2)^3*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) - (105*x^4*y^2*abs(x - 1/2)^(3/2)*(2*real(x) - 1)*(8*real(x) - 4)*(x - 1)^2*(y - 1)^2)/(2*(2*x - 1)^2*(x - 2*abs(x)^2)^2) - (105*x^3*y^2*abs(x - 1/2)^(1/2)*(2*real(x) - 1)*(8*real(x) - 4)*(x - 1)^2*(y - 1)^2)/(8*(2*x - 1)*(x - 2*abs(x)^2)*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2)) + (42*x^4*y^2*abs(x - 1/2)^(5/2)*(2*real(x) - 1)*(8*real(x) - 4)*(x - 1)^2*(y - 1)^2)/((2*x - 1)^2*(x - 2*abs(x)^2)^2*(-((2*x - 1)*(x - 2*abs(x)^2))/x)^(1/2))) for (idx, (x,y)) in enumerate(zip(xs, ys))]]
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
    canonical_qrule_gj_l = Quadrature.tensor_product_rule((Quadrature.gauss_jacobi(14, -1/2, 0), Quadrature.gauss_legendre(12)))
    canonical_qrule_gj_r = Quadrature.tensor_product_rule((Quadrature.gauss_jacobi(14, 0, -1/2), Quadrature.gauss_legendre(12)))
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
        geometry_i = Geometry.create_curvilinear_square((0.0, 0.0), (L, L), num_elements; crazy_c=0.2)
        geometry_cart = Geometry.get_base_geometry(geometry_i)

        starting_points = ((0.0, 0.0),)
        Wif, Gif_1, Gif_2, Qif = create_function_spaces(
            geometry_i, starting_points, num_elements, p, r
        )

        Wi = Forms.FormSpace(Val(0), geometry_i, Wif, "w_h")
        Gi = Forms.ModifiedOneFormSpace(
            geometry_i, FunctionSpaces.DirectSumSpace((Gif_1, Gif_2)), "theta_h"
        )
        Qi = Forms.ModifiedVolumeFormSpace(geometry_i, Qif, "q_h")


        dΩ_i = Quadrature.StandardQuadrature(
            canonical_qrule, Geometry.get_num_elements(geometry_i)
        )
        elements_at_singularity_l = Int[]
        elements_at_singularity_r = Int[]
        for element_id in 1:Geometry.get_num_elements(geometry_cart)
            if Geometry.get_constituent_element_id(geometry_cart, element_id)[1][1] == Int(num_elements[1]/2)
                push!(elements_at_singularity_l, element_id)
            end
            if Geometry.get_constituent_element_id(geometry_cart, element_id)[1][1] == Int(num_elements[1]/2)+1
                push!(elements_at_singularity_r, element_id)
            end
        end

        f⁰_i = Forms.AnalyticalFormField(
            Val(0),
            (x, id, xi) -> forcing_function_with_scaling(x, num_elements[1], id, elements_at_singularity_l, elements_at_singularity_r, geometry_i, xi),
            geometry_i,
            "f⁰",
            true,
        )
        dΩ_gj_i = Quadrature.SemiStandardQuadrature(
            canonical_qrule, canonical_qrule_gj_l, Set(elements_at_singularity_l), canonical_qrule_gj_r, Set(elements_at_singularity_r), Geometry.get_num_elements(geometry_i)
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
            Wi,
            false,
            dΩ_gj_i;
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
    end

    pfilename = "TestCase5-L2forcing-nels$(num_elements_study)-p$(p)-r$(r)"
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

    return nothing
end

function create_function_spaces(geo, starting_points, num_elements, p, r)
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

    return Wif, Gif_1, Gif_2, Qif
end

run_case(case_pre)
run_case(case)
run_case(case_post)
