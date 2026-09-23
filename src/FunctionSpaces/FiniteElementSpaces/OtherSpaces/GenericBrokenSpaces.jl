struct MultiPatchBrokenSpace{manifold_dim, num_patches, T, TE, TI, TJ} <:
       AbstractFESpace{manifold_dim, 1, num_patches}
    function_spaces::T  # NTuple{num_patches, TensorProductSpace{manifold_dim, BSplineSpace{Bernstein}, BSplineSpace{Bernstein}}} ?
    extraction_op::ExtractionOperator{1, TE, TI, TJ}
    dof_partition::Vector{Vector{Vector{Int}}}
    global_to_local_dof_dict::Dict{Int, Dict{Int, Int}}
end

function create_multi_patch_broken_space(
    function_spaces::T
) where {
    manifold_dim, num_patches, T <: NTuple{num_patches, AbstractFESpace{manifold_dim, 1, 1}}
}
    num_elements = 0
    max_global_dof = 0
    elems_per_patch = zeros(Int, num_patches)
    elems_per_patch_offset = zeros(Int, num_patches + 1)
    num_basis_per_patch = zeros(Int, num_patches)
    num_basis_per_patch_offset = zeros(Int, num_patches + 1)
    for patch_idx in 1:1:num_patches
        elems_on_patch = get_num_elements(function_spaces[patch_idx])
        elems_per_patch[patch_idx] = elems_on_patch
        elems_per_patch_offset[patch_idx + 1] =
            elems_on_patch + elems_per_patch_offset[patch_idx]
        num_elements += elems_on_patch

        num_basis_on_patch = get_num_basis(function_spaces[patch_idx])
        num_basis_per_patch[patch_idx] = num_basis_on_patch
        num_basis_per_patch_offset[patch_idx + 1] =
            num_basis_on_patch + num_basis_per_patch_offset[patch_idx]
        max_global_dof += num_basis_on_patch
    end
    global_to_local_dof_dict = Dict{Int, Dict{Int, Int}}()
    for patch_idx in 1:1:num_patches
        for dof_i in 1:1:num_basis_per_patch[patch_idx]
            global_to_local_dof_dict[num_basis_per_patch_offset[patch_idx] + dof_i] = Dict{
                Int, Int
            }(
                patch_idx => dof_i
            )
        end
    end

    extraction_coefficients = Vector{NTuple{1, LinearAlgebra.UniformScaling{Bool}}}(
        undef, num_elements
    )
    basis_indices = Vector{Indices{1, Vector{Int}, UnitRange{Int}}}(undef, num_elements)
    for patch_idx in 1:1:num_patches
        for elem_idx in 1:1:get_num_elements(function_spaces[patch_idx])
            global_elem_id = elems_per_patch_offset[patch_idx] + elem_idx

            # Get the local extraction coefficients and basis indices.
            extr_coeffs, indices = get_extraction(function_spaces[patch_idx], elem_idx)
            indices = get_basis_indices(function_spaces[patch_idx], elem_idx)

            extraction_coefficients[global_elem_id] = (LinearAlgebra.I,)

            basis_indices[global_elem_id] = Indices(
                [
                    num_basis_per_patch_offset[patch_idx] + local_dof for
                    local_dof in indices
                ],
                (1:length(indices),),
            )
        end
    end

    E = ExtractionOperator(
        extraction_coefficients, basis_indices, num_elements, max_global_dof
    )

    dof_partition = [
        get_dof_partition(function_space)[1] for function_space in function_spaces
    ]

    return MultiPatchBrokenSpace{manifold_dim, num_patches, T, get_EIJ_types(E)...}(
        function_spaces, E, dof_partition, global_to_local_dof_dict
    )
end

function get_num_elements_per_patch(
    space::MultiPatchBrokenSpace{manifold_dim, num_patches}
) where {manifold_dim, num_patches}
    return ntuple(num_patches) do patch_id
        return get_num_elements(space.function_spaces[patch_id])
    end
end

function get_max_local_dim(
    space::MultiPatchBrokenSpace{manifold_dim, num_patches}
) where {manifold_dim, num_patches}
    max_local_dim = 0
    for patch_idx in 1:num_patches
        max_local_dim_patch = 0
        for local_space in space.function_spaces
            max_local_dim_patch += get_max_local_dim(local_space)
        end
        max_local_dim = max(max_local_dim, max_local_dim_patch)
    end
    return max_local_dim
end

function get_local_basis(
    space::MultiPatchBrokenSpace{manifold_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
    component_id::Int=1,
) where {manifold_dim, num_patches}
    patch_id, local_element_id = get_patch_and_local_element_id(space, element_id)

    # Get the constituent space on this patch and its local basis.
    return evaluate(space.function_spaces[patch_id], local_element_id, xi, nderivatives)[1]
end
