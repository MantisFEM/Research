struct MultiPatchC0Space{num_patches, T, TE, TI, TJ} <: AbstractFESpace{2, 1, num_patches}
    function_spaces::T
    extraction_op::ExtractionOperator{1, TE, TI, TJ}
    dof_partition::Vector{Vector{Vector{Int}}}
    global_to_local_dof_dict::Dict{Int, Dict{Int, Int}}
    local_to_global_dof_dict::Dict{Tuple{Int, Int}, Int}
end

function _edge_number_to_dofpart_division(edge_number::Int)
    if edge_number == 1
        # Bottom edge, so vertices BL and BR, and edge B.
        return 1, 2, 3
    elseif edge_number == 2
        # Right edge, so vertices BR and TR, and edge R.
        return 3, 6, 9
    elseif edge_number == 3
        # Top edge, so vertices TL and TR, and edge T.
        return 7, 8, 9
    elseif edge_number == 4
        # Left edge, so vertices BL and TL, and edge L
        return 1, 4, 7
    end
end

function create_multi_patch_c0_space(
    function_spaces::T, patch_connectivity::NTuple{num_patches, NTuple{4, NTuple{2, Int}}}
) where {num_patches, T <: NTuple{num_patches, AbstractFESpace{2, 1, 1}}}
    # Compute the total number of elements and the offsets for each patch.
    num_elements = 0
    elems_per_patch = zeros(Int, num_patches)
    num_basis_per_patch = zeros(Int, num_patches)
    for patch_idx in 1:1:num_patches
        elems_on_patch = get_num_elements(function_spaces[patch_idx])
        elems_per_patch[patch_idx] = elems_on_patch
        num_elements += elems_on_patch

        num_basis_per_patch[patch_idx] = get_num_basis(function_spaces[patch_idx])
    end
    elems_per_patch_offset = vcat(0, cumsum(elems_per_patch[1:(end - 1)]))

    for patch_idx in 1:1:num_patches
        for patch_i in 1:1:num_patches
            if patch_idx != patch_i &&
                function_spaces[patch_idx] === function_spaces[patch_i]
                throw(
                    ArgumentError(
                        "The function spaces on different patches need to be different instances ($patch_idx, $patch_i).",
                    ),
                )
            end
        end
    end

    # Create the dof partition, accounting for shared dofs.
    global_dof = 1
    dof_partition = Vector{Vector{Vector{Int}}}(undef, num_patches)
    global_to_local_dof_dict = Dict{Int, Dict{Int, Int}}()
    local_to_global_dof_dict = Dict{Tuple{Int, Int}, Int}()
    local_dof_partition = [
        get_dof_partition(function_spaces[patch_i])[1] for patch_i in 1:1:num_patches
    ]
    for patch_i in 1:1:num_patches
        # Allocate the dof partition for this patch.
        dof_partition[patch_i] = Vector{Vector{Int}}(undef, 9)

        # Keep track of which dof groups on a patch have already been
        # assigned. Relevant for patch_i > 1.
        dof_groups_assigned = Vector{Int}(undef, 0)

        if patch_i == 1
            # First patch.
            for dof_group_i in eachindex(local_dof_partition[patch_i])
                dof_partition[patch_i][dof_group_i] = Vector{Int}(
                    undef, size(local_dof_partition[patch_i][dof_group_i])
                )

                for (idx, local_dof_i) in pairs(local_dof_partition[patch_i][dof_group_i])
                    # Directly assign the new global dof numbers to all
                    # dofs, as nothing is shared yet.
                    dof_partition[patch_i][dof_group_i][idx] = global_dof

                    global_to_local_dof_dict[global_dof] = Dict{Int, Int}(
                        patch_i => local_dof_i
                    )
                    local_to_global_dof_dict[(
                        patch_i, local_dof_partition[patch_i][dof_group_i][idx]
                    )] = global_dof
                    global_dof += 1
                end
            end

        else
            # Next patch. All numbers that are not shared with the
            # previously processed patches are inherited. The others are
            # given the already assigned number.
            for edge_i in eachindex(patch_connectivity[patch_i])

                # Get the neighbouring patch and the edge number.
                (neighbour_patch, neighbour_edge) = patch_connectivity[patch_i][edge_i]
                # Get the dof groups on the current edge.
                vertex_i1, edge_nr1, vertex_i2 = _edge_number_to_dofpart_division(edge_i)

                # Only look at actual neighbours (skipping 0) and the
                # patches that have already been processed.
                if neighbour_patch != 0 && neighbour_patch < patch_i
                    # Get the dofs on the edge that are shared.

                    # Get the dof groups on the neighbouring edge.
                    vertex_n1, edge_nr2, vertex_n2 = _edge_number_to_dofpart_division(
                        neighbour_edge
                    )

                    # The global partition for the current patch needs
                    # to be updated with the already assigned numbers.
                    # The order of the newly added dofs matters, but in
                    # the C0 case the ordering is the same, so no
                    # shuffle is needed.
                    dof_partition[patch_i][vertex_i1] = dof_partition[neighbour_patch][vertex_n1]

                    dof_partition[patch_i][edge_nr1] = dof_partition[neighbour_patch][edge_nr2]

                    dof_partition[patch_i][vertex_i2] = dof_partition[neighbour_patch][vertex_n2]

                    # Update the global to local dof dict. The global
                    # dofs are now already defined.
                    for dof_group_i in [vertex_i1, edge_nr1, vertex_i2]
                        for (global_dof_i, local_dof_i) in zip(
                            dof_partition[patch_i][dof_group_i],
                            local_dof_partition[patch_i][dof_group_i],
                        )
                            global_to_local_dof_dict[global_dof_i][patch_i] = local_dof_i
                            local_to_global_dof_dict[(patch_i, local_dof_i)] = global_dof_i
                        end
                    end

                    push!(dof_groups_assigned, vertex_i1, edge_nr1, vertex_i2)
                end
            end

            # Add the non-shared dofs.
            for dof_group_i in eachindex(local_dof_partition[patch_i])
                # Only add the dofs that have not been assigned yet.
                if !(dof_group_i in dof_groups_assigned)
                    dof_partition[patch_i][dof_group_i] = Vector{Int}(
                        undef, size(local_dof_partition[patch_i][dof_group_i])
                    )

                    for (idx, local_dof_i) in
                        enumerate(local_dof_partition[patch_i][dof_group_i])
                        dof_partition[patch_i][dof_group_i][idx] = global_dof

                        # All the global dofs are unique again
                        global_to_local_dof_dict[global_dof] = Dict{Int, Int}(
                            patch_i => local_dof_i
                        )
                        local_to_global_dof_dict[(
                            patch_i, local_dof_partition[patch_i][dof_group_i][idx]
                        )] = global_dof
                        global_dof += 1
                    end
                end
            end
        end
    end
    global_dof -= 1  # Last global dof that was processed.

    # Create the global extraction operator.
    extraction_coefficients = Vector{NTuple{1, LinearAlgebra.UniformScaling{Bool}}}(
        undef, num_elements
    )
    basis_indices = Vector{Indices{1, Vector{Int}, UnitRange{Int}}}(undef, num_elements)
    for patch_idx in 1:1:num_patches
        for elem_idx in 1:1:get_num_elements(function_spaces[patch_idx])
            global_elem_id = elems_per_patch_offset[patch_idx] + elem_idx

            # Get the local extraction coefficients and basis indices.
            indices = get_basis_indices(function_spaces[patch_idx], elem_idx)

            extraction_coefficients[global_elem_id] = (LinearAlgebra.I,)

            basis_indices[global_elem_id] = Indices(
                [local_to_global_dof_dict[(patch_idx, local_dof)] for local_dof in indices],
                (1:length(indices),),
            )
        end
    end

    E = ExtractionOperator(extraction_coefficients, basis_indices, num_elements, global_dof)

    return MultiPatchC0Space{num_patches, T, get_EIJ_types(E)...}(
        function_spaces,
        E,
        dof_partition,
        global_to_local_dof_dict,
        local_to_global_dof_dict,
    )
end

function get_local_basis(
    space::MultiPatchC0Space,
    element_id::Int,
    xi::Points.AbstractPoints{2},
    nderivatives::Int,
    component_id::Int=1,
)
    patch_id, local_element_id = get_patch_and_local_element_id(space, element_id)

    # Only keep the evaluations, not the indices.
    return evaluate(space.function_spaces[patch_id], local_element_id, xi, nderivatives)[1]
end

function get_num_elements_per_patch(space::MultiPatchC0Space)
    return get_num_elements.(space.function_spaces)
end

function get_max_local_dim(space::MultiPatchC0Space{num_patches}) where {num_patches}
    max_local_dim = 0
    for patch_idx in 1:1:num_patches
        max_local_dim_patch = 0
        for local_space in space.function_spaces
            max_local_dim_patch += get_max_local_dim(local_space)
        end
        max_local_dim = max(max_local_dim, max_local_dim_patch)
    end
    return max_local_dim
end

function get_element_lengths(space::MultiPatchC0Space, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(space, element_id)
    return get_element_lengths(space.function_spaces[patch_id], local_element_id)
end

function get_element_vertices(space::MultiPatchC0Space, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(space, element_id)
    return get_element_vertices(space.function_spaces[patch_id], local_element_id)
end

function get_support(space::MultiPatchC0Space, basis_id::Int)
    local_basis_ids = space.global_to_local_dof_dict[basis_id]
    support = Int[]
    for (patch_id, local_basis_id) in local_basis_ids
        local_support = get_support(space.function_spaces[patch_id], local_basis_id)
        global_support = local_support
        for i in 1:(patch_id - 1)
            global_support .+= get_num_elements(space.function_spaces[i])
        end
        append!(support, global_support)
    end
    return support
end
