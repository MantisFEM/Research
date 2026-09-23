abstract type AbstractMultiPatchGeometry{manifold_dim, image_dim, num_patches} <:
              AbstractGeometry{manifold_dim, image_dim, num_patches} end

"""
    MultiPatchGeometry{manifold_dim, num_patches, GP} <: AbstractGeometry{manifold_dim}

A geometry consisting of multiple patches, each with its own geometry.

# Fields
- `geometry_per_patch::NTuple{GP, num_patches}`: The geometries for each patch.
"""
struct MultiPatchGeometry{manifold_dim, image_dim, num_patches, GT} <:
       AbstractMultiPatchGeometry{manifold_dim, image_dim, num_patches}
    geometry_per_patch::GT
    n_elements::Int
    image_dim::Int

    function MultiPatchGeometry(
        geometry_per_patch::GT
    ) where {
        manifold_dim,
        image_dim,
        num_patches,
        GT <: NTuple{num_patches, AbstractGeometry{manifold_dim, image_dim, 1}},
    }
        num_elements = 0
        for patch_id in 1:1:num_patches
            num_elements += get_num_elements(geometry_per_patch[patch_id])
        end
        return new{manifold_dim, image_dim, num_patches, GT}(
            geometry_per_patch, num_elements, image_dim
        )
    end
end

# Getters and setters.
"""
    get_geometry_on_patch(MPGeo::AbstractMultiPatchGeometry, patch_id::Int)

Get the geometry on a specific patch.

# Arguments
- `MPGeo::MultiPatchGeometry`: The multi-patch geometry.
- `patch_id::Int`: The patch ID.

# Returns
- `::GP <: AbstractGeometry{manifold_dim}`: The geometry on the specified patch.
"""
function get_geometry_on_patch(geometry::AbstractMultiPatchGeometry, patch_id::Int)
    return geometry.geometry_per_patch[patch_id]
end

function get_num_elements(geometry::AbstractMultiPatchGeometry)
    return geometry.n_elements
end

function get_num_elements_per_patch(geometry::AbstractMultiPatchGeometry)
    return ntuple(get_num_patches(geometry)) do i
        return get_num_elements(get_geometry_on_patch(geometry, i))
    end
end

get_num_patches(
    geometry::MultiPatchGeometry{manifold_dim, image_dim, num_patches, GT}
) where {manifold_dim, image_dim, num_patches, GT} = num_patches

function get_domain_dim(
    geometry::AbstractMultiPatchGeometry{manifold_dim}
) where {manifold_dim}
    return manifold_dim
end

function get_image_dim(geometry::AbstractMultiPatchGeometry)
    return geometry.image_dim
end

# Evaluation (and related) methods.
"""
    _find_element_on_patch(geometry::AbstractMultiPatchGeometry, element_id::Int)

Find the patch and local element ID for the given global element ID.

# Arguments
- `geometry::MultiPatchGeometry{manifold_dim, num_patches, GP}`: The multi-patch geometry.
- `element_id::Int`: The element of interest.

# Returns
- `patch_id::Int`: The patch on which the given element is located.
- `local_element_id::Int`: The local element ID on the patch.

# Throws
- `ArgumentError`: If the given element number is larger than the total number of elements.
"""
function _find_element_on_patch(geometry::AbstractMultiPatchGeometry, element_id::Int)
    elements_total = 0
    for patch_id in 1:1:get_num_patches(geometry)
        elements_on_patch = get_num_elements(get_geometry_on_patch(geometry, patch_id))
        elements_total += elements_on_patch
        if element_id <= elements_total
            return patch_id, element_id - elements_total + elements_on_patch
        end
    end
    throw(
        ArgumentError(
            "The element_id $element_id is too large for the multi-patch geometry. It has only $elements_total elements.",
        ),
    )
end

function evaluate(
    geometry::AbstractMultiPatchGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    # Find the patch on which the given element resides.
    patch_id, local_element_id = _find_element_on_patch(geometry, element_id)

    # Evaluate the geometry on the patch.
    return evaluate(geometry.geometry_per_patch[patch_id], local_element_id, xi)
end

function jacobian(
    geometry::AbstractMultiPatchGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    # Find the patch on which the given element resides.
    patch_id, local_element_id = _find_element_on_patch(geometry, element_id)

    # Evaluate the jacobian on the patch.
    return jacobian(geometry.geometry_per_patch[patch_id], local_element_id, xi)
end

function hessian(
    geometry::AbstractMultiPatchGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    # Find the patch on which the given element resides.
    patch_id, local_element_id = _find_element_on_patch(geometry, element_id)

    # Evaluate the jacobian on the patch.
    return hessian(geometry.geometry_per_patch[patch_id], local_element_id, xi)
end

function get_element_lengths(geometry::MultiPatchGeometry, element_id::Int)
    patch_id, local_element_id = _find_element_on_patch(geometry, element_id)
    return get_element_lengths(geometry.geometry_per_patch[patch_id], local_element_id)
end
