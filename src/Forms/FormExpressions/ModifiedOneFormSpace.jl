############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    ModifiedOneFormSpace{manifold_dim, G, F} <:
    AbstractFormSpace{manifold_dim, 1, G}

Concrete implementation of a function space for differential forms.

# Fields
- `geometry::G`: The geometry of the manifold
- `fem_space::F`: The finite element space(s) used for the form components
- `label::String`: Label for the form space

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `G`: Type of the geometry
- `F`: Type of the finite element space

# Inner Constructors
- `FormSpace(form_rank::Int, geometry::G, fem_space::F, label::String)`: General
    constructor for differential form spaces.
"""
struct ModifiedOneFormSpace{manifold_dim, G, F} <: AbstractFormSpace{manifold_dim, 1, G}
    geometry::G
    fem_space::F
    label::String

    """
        ModifiedOneFormSpace(
            geometry::G, fem_space::F, label::String
        ) where {
            manifold_dim,
            num_patches,
            G <: Geometry.AbstractGeometry{manifold_dim},
            F <: FunctionSpaces.AbstractFESpace{manifold_dim, manifold_dim, num_patches},
        }

    General constructor for differential form spaces.

    # Arguments
    - `form_rank::Int`: Differential form rank.
    - `geometry::G`: The geometry where the form is defined.
    - `fem_space::F`: The function space used to represent the form.
    - `label::String`: The label of the form space.

    # Returns
    - `ModifiedOneFormSpace{manifold_dim, form_rank, G, F}`: The ModifiedOneFormSpace structure.
    """
    function ModifiedOneFormSpace(
        geometry::G, fem_space::F, label::String
    ) where {
        manifold_dim,
        num_patches,
        G <: Geometry.AbstractGeometry{manifold_dim},
        F <: FunctionSpaces.AbstractFESpace{manifold_dim, manifold_dim, num_patches},
    }
        if Geometry.get_image_dim(geometry) != manifold_dim
            throw(ArgumentError("Geometry image dimension must match manifold dimension."))
        end

        return new{manifold_dim, G, F}(geometry, fem_space, label)
    end
end
############################################################################################
#                                   Getters and setters                                    #
############################################################################################
get_estimated_nnz_per_elem(form_space::ModifiedOneFormSpace) = get_max_local_dim(form_space)
get_form(form_space::ModifiedOneFormSpace) = form_space

function get_form_space_tree(form_space::ModifiedOneFormSpace)
    return (get_form(form_space),)
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    form_space::ModifiedOneFormSpace{manifold_dim, G, F},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, F}
    # The one-form space is made up of components
    # e.g,
    #   1-forms:(dξ₁, dξ₂) (2D)
    #   1-forms:(dξ₁, dξ₂, dξ₃) (3D)
    # We use the numbering of the function space.

    # Evaluate the form spaces
    local_form_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_idx, xi, 0
    )  # (only evaluate the basis (0-th order derivative))

    return local_form_basis[1][1], form_basis_indices
end

function _evaluate_form_in_canonical_coordinates(
    form_space::ModifiedOneFormSpace{manifold_dim, G, F},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, F}
    # Evaluate the form spaces on parametric domain ...
    local_form_basis, form_basis_indices = FunctionSpaces.evaluate(
        get_fe_space(form_space), element_idx, xi, nderivatives
    )  # (only evaluate the basis (0-th order derivative))
    # ... scale them with the jacobian entries
    local_form_basis = _scale_modified_one_form_by_jacobian(
        get_geometry(form_space), local_form_basis, element_idx, xi
    )

    # We need to return form_basis_indices as a vector of vectors to allow for multiple
    # index expressions, like the wedge
    return local_form_basis, [form_basis_indices]
end

function _scale_modified_one_form_by_hessian(
    geometry::Geometry.AbstractGeometry{manifold_dim},
    form_evaluations::Vector{Vector{Vector{Matrix{Float64}}}},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    # Get the hessian at the evaluation points
    H = Geometry.hessian(geometry, element_idx, xi) # (num_eval_points, image_dim, manifold_dim)
    # Scale the form evaluations with the hessian entries
    scaled_form_evaluations = similar(form_evaluations)
    for i in eachindex(scaled_form_evaluations)
        scaled_form_evaluations[i] = similar(form_evaluations[i])
        for j in eachindex(scaled_form_evaluations[i])
            scaled_form_evaluations[i][j] = similar(form_evaluations[i][j])
            for k in 1:manifold_dim
                scaled_form_evaluations[i][j][k] = form_evaluations[i][j][1] .* H[:][1, k]
                for l in 2:manifold_dim
                    scaled_form_evaluations[i][j][k] +=
                        form_evaluations[i][j][l] .* H[:][l, k]
                end
            end
        end
    end

    return scaled_form_evaluations
end

function _scale_modified_one_form_by_jacobian(
    geometry::Geometry.AbstractGeometry{manifold_dim},
    form_evaluations::Vector{Vector{Vector{Matrix{Float64}}}},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    # Get the jacobian at the evaluation points
    J = Geometry.jacobian(geometry, element_idx, xi)
    # Scale the form evaluations with the jacobian entries
    scaled_form_evaluations = similar(form_evaluations)
    for i in eachindex(scaled_form_evaluations)
        scaled_form_evaluations[i] = similar(form_evaluations[i])
        for j in eachindex(scaled_form_evaluations[i])
            scaled_form_evaluations[i][j] = similar(form_evaluations[i][j])
            for k in 1:manifold_dim
                scaled_form_evaluations[i][j][k] = zeros(size(form_evaluations[i][j][k]))
                for p in eachindex(J)
                    scaled_form_evaluations[i][j][k][p, :] +=
                        form_evaluations[i][j][1][p, :] .* J[p][1, k]
                end
                # scaled_form_evaluations[i][j][k] = form_evaluations[i][j][1] .* J[:][1, k]
                for l in 2:manifold_dim # image_dim = manifold_dim here
                    for p in eachindex(J)
                        scaled_form_evaluations[i][j][k][p, :] +=
                            form_evaluations[i][j][l][p, :] .* J[p][l, k]
                    end
                end
            end
        end
    end

    return scaled_form_evaluations
end
