using LinearAlgebra
using Printf

############################################################################################
#                                        Variables                                         #
############################################################################################
fpi = Float64(π)

############################################################################################
#                                     Hodge Laplacian                                      #
############################################################################################

function circular_data(rank::Int, geo::Geometry.AbstractGeometry; a=100, b=9 / 100)
    if rank != 1
        throw(ArgumentError("This problem data is only defined for 1-forms."))
    end

    ϕ(x, y) = (x * (1 - x) * y * (1 - y))^2

    ∇ϕ₁(x, y) = 2 * (y - 1)^2 * y^2 * (x - 1) * x * (2 * x - 1)
    d∇ϕ₁dx(x, y) = 2 * (y - 1)^2 * y^2 * (6 * x^2 - 6 * x + 1)
    d∇ϕ₁²dx₂(x, y) = 2 * (y - 1)^2 * y^2 * (12 * x - 6)
    d∇ϕ₁dy(x, y) = 4 * (x - 1) * x * (2 * x - 1) * (y - 1) * y * (2 * y - 1)
    d∇ϕ₁²dy₂(x, y) = 4 * (x - 1) * x * (2 * x - 1) * (6 * y^2 - 6 * y + 1)

    ∇ϕ₂(x, y) = 2 * (x - 1)^2 * x^2 * (y - 1) * y * (2 * y - 1)
    d∇ϕ₂dx(x, y) = 4 * (y - 1) * y * (2 * y - 1) * (x - 1) * x * (2 * x - 1)
    d∇ϕ₂²dx₂(x, y) = 4 * (y - 1) * y * (2 * y - 1) * (6 * x^2 - 6 * x + 1)
    d∇ϕ₂dy(x, y) = 2 * (x - 1)^2 * x^2 * (6 * y^2 - 6 * y + 1)
    d∇ϕ₂²dy₂(x, y) = 2 * (x - 1)^2 * x^2 * (12 * y - 6)

    φ(x, y) = tanh(a * ((x - 0.5)^2 + (y - 0.5)^2 - b))
    dφdx(x, y) = a * sech(a * ((x - 1 / 2)^2 + (y - 1 / 2)^2 - b))^2 * (2 * x - 1)
    dφ²dx₂(x, y) =
        -2 *
        a *
        sech((a * (2 * x^2 - 2 * x + 2 * (y - 1) * y - 2 * b + 1)) / 2)^2 *
        (
            a *
            (2 * x - 1)^2 *
            tanh((a * (2 * x^2 - 2 * x + 2 * (y - 1) * y - 2 * b + 1)) / 2) - 1
        )
    dφdy(x, y) = a * sech(a * ((y - 1 / 2)^2 + (x - 1 / 2)^2 - b))^2 * (2 * y - 1)
    dφ²dy₂(x, y) =
        -2 *
        a *
        sech((a * (2 * y^2 - 2 * y + 2 * (x - 1) * x - 2 * b + 1)) / 2)^2 *
        (
            a *
            (2 * y - 1)^2 *
            tanh((a * (2 * y^2 - 2 * y + 2 * (x - 1) * x - 2 * b + 1)) / 2) - 1
        )

    # u = [∇ϕ₁φ, ∇ϕ₁φ]
    u₁(x, y) = ∇ϕ₁(x, y) * φ(x, y)
    u₂(x, y) = ∇ϕ₂(x, y) * φ(x, y)

    du₁dx(x, y) = d∇ϕ₁dx(x, y) * φ(x, y) + ∇ϕ₁(x, y) * dφdx(x, y)
    du₁²dx₂(x, y) =
        d∇ϕ₁²dx₂(x, y) * φ(x, y) + 2 * d∇ϕ₁dx(x, y) * dφdx(x, y) + ∇ϕ₁(x, y) * dφ²dx₂(x, y)
    du₁dy(x, y) = d∇ϕ₁dy(x, y) * φ(x, y) + ∇ϕ₁(x, y) * dφdy(x, y)
    du₁²dy₂(x, y) =
        d∇ϕ₁²dy₂(x, y) * φ(x, y) + 2 * d∇ϕ₁dy(x, y) * dφdy(x, y) + ∇ϕ₁(x, y) * dφ²dy₂(x, y)

    du₂dx(x, y) = d∇ϕ₂dx(x, y) * φ(x, y) + ∇ϕ₂(x, y) * dφdx(x, y)
    du₂²dx₂(x, y) =
        d∇ϕ₂²dx₂(x, y) * φ(x, y) + 2 * d∇ϕ₂dx(x, y) * dφdx(x, y) + ∇ϕ₂(x, y) * dφ²dx₂(x, y)

    du₂dy(x, y) = d∇ϕ₂dy(x, y) * φ(x, y) + ∇ϕ₂(x, y) * dφdy(x, y)
    du₂²dy₂(x, y) =
        d∇ϕ₂²dy₂(x, y) * φ(x, y) + 2 * d∇ϕ₂dy(x, y) * dφdy(x, y) + ∇ϕ₂(x, y) * dφ²dy₂(x, y)

    function u_function(x)
        return [u₁.(x[:, 1], x[:, 2]), u₂.(x[:, 1], x[:, 2])]
    end

    # δu = -div u
    function δu_function(x)
        return [@. -(du₁dx(x[:, 1], x[:, 2]) + du₂dy(x[:, 1], x[:, 2]))]
    end

    # f = [-Δu₁, -Δu₂]
    function f_function(x)
        return [
            -(du₁²dx₂.(x[:, 1], x[:, 2]) + du₁²dy₂.(x[:, 1], x[:, 2])),
            -(du₂²dx₂.(x[:, 1], x[:, 2]) + du₂²dy₂.(x[:, 1], x[:, 2])),
        ]
    end

    δu = Forms.AnalyticalFormField(0, δu_function, geo, "δu")
    u = Forms.AnalyticalFormField(1, u_function, geo, "u")
    f = Forms.AnalyticalFormField(1, f_function, geo, "f")

    return δu, u, f
end

############################################################################################
#                                       Convenience                                        #
############################################################################################

function get_elements_in_box(
    geometry::Geometry.AbstractGeometry{2},
    first_element::NTuple{2, Int},
    last_element::NTuple{2, Int},
)
    lin_num_elements = Geometry.get_lin_num_elements(geometry)
    elements_in_box = Vector{Int}(
        undef,
        (last_element[1] - first_element[1] + 1) * (last_element[2] - first_element[2] + 1),
    )
    count = 1
    for y_element in first_element[2]:last_element[2]
        for x_element in first_element[1]:last_element[1]
			element = (x_element, y_element)
			elements_in_box[count] = lin_num_elements[element...]
			count += 1
        end
    end

    return elements_in_box
end
