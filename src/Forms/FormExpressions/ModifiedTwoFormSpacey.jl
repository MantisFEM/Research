############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    ModifiedVolumeFormSpace{manifold_dim, G, F} <:
    AbstractFormSpace{manifold_dim, manifold_dim, G}

Concrete implementation of a function space for differential forms.

# Fields
- `geometry::G`: The geometry of the manifold
- `fem_space::F`: The finite element space(s) used for the form components
- `label::String`: Label for the form space

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `G`: Type of the geometry
- `F`: Type of the finite element space

# Inner Constructors
- `FormSpace(geometry::G, fem_space::F, label::String)`: General
    constructor for differential form spaces.
"""
struct ModifiedVolumeFormSpace{manifold_dim, G, F} <:
       AbstractFormSpace{manifold_dim, manifold_dim, G}
    geometry::G
    fem_space::F
    label::String

    function ModifiedVolumeFormSpace(
        geometry::G, fem_space::F, label::String
    ) where {
        manifold_dim,
        num_patches,
        G <: Geometry.AbstractGeometry{manifold_dim},
        F <: FunctionSpaces.AbstractFESpace{manifold_dim, 1, num_patches},
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
get_estimated_nnz_per_elem(form_space::ModifiedVolumeFormSpace) =
    get_max_local_dim(form_space)

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    form_space::ModifiedVolumeFormSpace{manifold_dim, G, F},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, G, F}
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
    form_space::ModifiedVolumeFormSpace{manifold_dim, G, F},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, F}
    # Evaluate the form spaces on parametric domain ...
    local_form_basis, form_basis_indices = FunctionSpaces.evaluate(
        get_fe_space(form_space), element_idx, xi, nderivatives
    )  # (only evaluate the basis (0-th order derivative))
    # ... scale them with the jacobian entries ...
    _scale_modified_volume_form!(
        local_form_basis, get_geometry(form_space), element_idx, xi
    )
    # We need to return form_basis_indices as a vector of vectors to allow for multiple
    # index expressions, like the wedge
    return local_form_basis, [form_basis_indices]
end

function _scale_modified_volume_form!(
    form_evaluations::Vector{Vector{Vector{Matrix{Float64}}}},
    geometry::Geometry.AbstractGeometry{manifold_dim},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    # Get the metric (determinant) at the evaluation points
    g, sqrt_g = Geometry.metric(geometry, element_idx, xi)
    # Scale the form evaluations with the jacobian entries
    for i in eachindex(form_evaluations)
        for j in eachindex(form_evaluations[i])
            # There is only 1 component for a volume form.
            form_evaluations[i][j][1] .*= sqrt_g
        end
    end

    return form_evaluations
end
