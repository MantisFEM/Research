"""
This file contains the construction for the approximate C1 splines space
and its associated functions.
"""

"""
    ApproximateC1Space{manifold_dim, num_patches} <: AbstractUnstructeredSpace{manifold_dim, num_patches}

An `manifold_dim`-variate approximate C1 multi-patch space with `num_patches` patches.

For the construction of this space, see [`create_approximate_C1_space(patch_tp_spaces::NTuple{num_patches, F}, patch_connectivity::NTuple{num_patches, NTuple{4, NTuple{2, Int}}}) where {manifold_dim, num_patches, F <: AbstractFESpace{manifold_dim}}`](@ref).

# Fields
- `function_spaces::NTuple{num_patches, NTuple{5, AbstractFESpace{manifold_dim}}}`: Collection of 5 tensor product spaces out of which the approximate C1 basis functions are build.
- `extraction_op::ExtractionOperator`: Extraction operator that specifies how to combine functions from the `function_spaces` spaces into functions for the approximate C1 space.
- `dof_partition::Vector{Vector{Int}}`: Partition of the global degrees of freedom.
- `local_dof_partition::Vector{Vector{Int}}`: Partition of the local degrees of freedom.

# Notes
- No connectivity data is stored in the space. While this is needed for
its construction, it is inherited from the underlying mesh and not stored here.
"""
struct ApproximateC1Space{manifold_dim, num_patches, F, E} <:
       AbstractFESpace{manifold_dim, 1, num_patches}
    function_spaces::F#NTuple{num_patches, NTuple{5, TP}}
    extraction_op::E
    dof_partition::Vector{Vector{Vector{Int}}}
    local_dof_partition::Vector{Vector{Vector{Int}}}

    # Temporary, only for easier acces to the boundary dofs.
    N_outer_ring_per_patch::NTuple{num_patches, Int}
    global_to_local_dof_dict::Dict{Int, Dict{Int, Int}}
    local_to_global_dof_dict::Dict{Tuple{Int, Int}, Int}

    function ApproximateC1Space(
        function_spaces::F,
        extraction_op::E,
        dof_partition::Vector{Vector{Vector{Int}}},
        local_dof_partition::Vector{Vector{Vector{Int}}},
        N_outer_ring_per_patch::NTuple{num_patches, Int},
        global_to_local_dof_dict::Dict{Int, Dict{Int, Int}},
        local_to_global_dof_dict::Dict{Tuple{Int, Int}, Int},
    ) where {F, E, num_patches}
        return new{2, num_patches, F, E}(
            function_spaces,
            extraction_op,
            dof_partition,
            local_dof_partition,
            N_outer_ring_per_patch,
            global_to_local_dof_dict,
            local_to_global_dof_dict,
        )
    end
end

# """
#     create_approximate_C1_space()

# Constructs an approximate ``C^1`` multi-patch splines spaces from given
# tensor product spaces, patch connectivity, and gluing data.

# # Arguments
# - `patch_tp_spaces::NTuple{num_patches, F<:AbstractFESpace{manifold_dim}}`: Tensor-product b-spline space per patch.

# # Returns
# - `::ApproximateC1Space`: The constructed Approximate ``C^1`` space.

# # Throws
# - `ArgumentError`: If the tensor product regularity on any patch does not satisfy r >= 1 (excluding endpoint of an open knot vector).
# - `ArgumentError`: If the tensor product polynomial degree on any patch does not satisfy p >= 2.

# # Notes and References
# The `patch_tp_spaces` are used to define the approximate C1 space. The
# interior basis function are the same as the TP space, while the edge and
# vertex functions are a linear combination of the tp space functions and
# univariate splines spaces on the boundaries of the same or one degree
# lower (always with maximal regularity). The degree and elements of these
# additional spaces are inherited from the tp spaces.

# The approximate ``C^1`` construction was developed in [Weinmuller2021](@cite)
# and [Weinmuller2022](@cite). Also consult these papers for more details
# on the construction.
# """
# function create_approximate_C1_space(
#     patch_tp_spaces::NTuple{num_patches, F},
#     patch_connectivity::NTuple{num_patches, NTuple{4, NTuple{2, Int}}},
#     gluing_data::NTuple{num_patches, NTuple{4, NTuple{2, Function}}},
#     dummy_arg::Bool,
#     p_tilde_lr::Int=-1,
#     r_tilde_lr::Int=-2,
#     p_tilde_bt::Int=-1,
#     r_tilde_bt::Int=-2,
#     interpolate_vertices::Bool=false,
# ) where {manifold_dim, num_patches, F <: AbstractFESpace{manifold_dim}}

#     # Start by creating patch local information (local numbering, the
#     # intermediate spaces, and the 5 local basis TP spaces).
#     local_dof_partition_per_patch = Vector{Vector{Vector{Int}}}(undef, num_patches)
#     constituent_spaces_per_patch_vec = Vector{
#         Tuple{
#             TensorProductSpace{manifold_dim}, # Interior space
#             BSplineSpace, # Interface space bottom-top +
#             BSplineSpace, # Interface space bottom-top -
#             BSplineSpace, # Interface space left-right +
#             BSplineSpace, # Interface space left-right -
#             BSplineSpace, # Gluing data space bottom-top
#             BSplineSpace, # Gluing data space left-right
#             TensorProductSpace{manifold_dim}, # Edge TP 1LR space
#             TensorProductSpace{manifold_dim}, # Edge TP 1LR-gd space
#             TensorProductSpace{manifold_dim}, # Edge TP 2BT space
#             TensorProductSpace{manifold_dim},
#         },
#     }(
#         undef, num_patches
#     ) # Edge TP 2BT-gd space
#     constituent_spaces_per_patch_vec_tp = Vector{NTuple{5, AbstractFESpace{manifold_dim}}}(
#         undef, num_patches
#     ) # Only the TP spaces.
#     N_outer_ring_per_patch = Vector{Int}(undef, num_patches)

#     # Loop over each patch to create the local numbering and spaces.
#     for patch_idx in 1:1:num_patches
#         # Tensor product space on the patch. The other spaces are
#         # directly derived from this one.
#         tp_space = patch_tp_spaces[patch_idx]

#         # Create all intermediate spaces used in the approx. C1 constr.
#         (Sbt_plus, Sbt_minus, Slr_plus, Slr_minus, Sgd_bt, Sgd_lr, S1_Slr_plus, S1_Slr_minus_Sgd, S2_Sbt_plus, S2_sbt_minus_Sgd) = _create_approx_C1_patch_constituent_spaces(
#             patch_idx, tp_space, p_tilde_lr, r_tilde_lr, p_tilde_bt, r_tilde_bt
#         )

#         # Get constituent function spaces of the tp space.
#         bspl_1 = tp_space.constituent_spaces[1]
#         bspl_2 = tp_space.constituent_spaces[2]

#         # Number of basis functions per dimension.
#         N1 = get_num_basis(bspl_1)
#         N2 = get_num_basis(bspl_2)
#         Nlr_plus = get_num_basis(Slr_plus)
#         Nlr_minus = get_num_basis(Slr_minus)
#         Nbt_plus = get_num_basis(Sbt_plus)
#         Nbt_minus = get_num_basis(Sbt_minus)

#         constituent_spaces_per_patch_vec[patch_idx] = (
#             tp_space,
#             Sbt_plus,
#             Sbt_minus,
#             Slr_plus,
#             Slr_minus,
#             Sgd_bt,
#             Sgd_lr,
#             S1_Slr_plus,
#             S1_Slr_minus_Sgd,
#             S2_Sbt_plus,
#             S2_sbt_minus_Sgd,
#         )
#         constituent_spaces_per_patch_vec_tp[patch_idx] = (
#             tp_space, S1_Slr_plus, S1_Slr_minus_Sgd, S2_Sbt_plus, S2_sbt_minus_Sgd
#         )
#         N_outer_ring_per_patch[patch_idx] =
#             Nbt_plus + (Nlr_plus - 1) + (Nbt_plus - 1) + (Nlr_plus - 2)

#         # Compute the total number of basis functions in the approx C1 space.
#         # Interior: (N1 - 2 - 2) * (N2 - 2 - 2) # -2 on both sides and per dim.
#         # Edge (left and right): (Nlr_plus - 3 - 3) + (Nlr_minus - 2 - 2)
#         # Edge (bottom and top): (Nbt_plus - 3 - 3) + (Nbt_minus - 2 - 2)
#         # Vertex: 6 functions per vertex (4 vertices), always
#         N_C1_int = (N1 - 2 - 2) * (N2 - 2 - 2)
#         N_C1_edge_lr = ((Nlr_plus - 3 - 3) + (Nlr_minus - 2 - 2))
#         N_C1_edge_bt = ((Nbt_plus - 3 - 3) + (Nbt_minus - 2 - 2))
#         N_C1_vertex = 6
#         N_total_C1 = N_C1_int + 2 * N_C1_edge_lr + 2 * N_C1_edge_bt + 4 * N_C1_vertex

#         # Create the local dof partition for the approx. C1 space. Note
#         # that the number of dofs is not the same as that of the tp
#         # space and that the dofs are distributed and numbered differently.
#         local_dof_partition_per_patch[patch_idx] = _create_approx_C1_local_dof_partition_patch(
#             N_total_C1,
#             N_C1_int,
#             N_C1_edge_bt,
#             N_C1_edge_lr,
#             N_C1_vertex,
#             Nbt_plus,
#             Nbt_minus,
#             Nlr_plus,
#             Nlr_minus,
#         )
#     end

#     # The constituent spaces including the 1D spaces are used in
#     # defining the global dofs. Only the tensor product spaces are used
#     # in the extraction and evaluation.
#     constituent_spaces_per_patch_all = Tuple(constituent_spaces_per_patch_vec)
#     constituent_spaces_per_patch_tps = Tuple(constituent_spaces_per_patch_vec_tp)

#     # Create the global dof numbering from the local one, the global to
#     # local dof dict, and the dof partition based on the global dofs.
#     (global_to_local_dof_dict, global_dof_partition_per_patch, max_global_dof) = _create_global_dof_dicts(
#         local_dof_partition_per_patch,
#         patch_connectivity,
#         Tuple(N_outer_ring_per_patch),
#         interpolate_vertices,
#     )

#     # Extraction matrices.
#     extraction = _extract_approximateC1_to_constituent(
#         constituent_spaces_per_patch_all,
#         global_to_local_dof_dict,
#         global_dof_partition_per_patch,
#         max_global_dof,
#         gluing_data,
#         interpolate_vertices,
#         "interpolate",
#     )

#     # Hacky way to create the local to global dof dict, because I didn't
#     # create it in the global dof assignment function.
#     # TODO: Fix this.
#     local_to_global_dof_dict = Dict{Tuple{Int, Int}, Int}()
#     for (global_dof, local_dof_dict) in global_to_local_dof_dict
#         for (patch_idx, local_dof) in local_dof_dict
#             local_to_global_dof_dict[(patch_idx, local_dof)] = global_dof
#         end
#     end

#     return ApproximateC1Space(
#         constituent_spaces_per_patch_tps,
#         extraction,
#         global_dof_partition_per_patch,
#         local_dof_partition_per_patch,
#         Tuple(N_outer_ring_per_patch),
#         global_to_local_dof_dict,
#         local_to_global_dof_dict,
#     )
# end

# Helper functions for the constructor.
"""
    _construct_approx_gluing_data_space(p_tilde::Int, r_tilde::Int, p_tp::Int, n_elems::Int, patch::Mesh.Patch1D)

Constructs the spline space used for the approximation of the gluing data.

# Arguments
- `p_tilde::Int`: Polynomial degree of the gluing data space.
- `r_tilde::Int`: Regularity of the gluing data space.
- `p_tp::Int`: Polynomial degree of the tensor product space.
- `n_elems::Int`: Number of elements in the tensor product space.
- `patch::Mesh.Patch1D`: The patch on which the space relevant part of
the tensor product space is defined.

# Returns
- `::BSplineSpace{1}`: The constructed spline space.
"""
function _construct_approx_gluing_data_space(
    p_tilde::Int, r_tilde::Int, p_tp::Int, n_elems::Int, patch::Mesh.Patch1D
)
    if p_tilde == -1
        p_tilde = p_tp - 1
    end

    if r_tilde == -2
        r_tilde = p_tp - 2
    end

    kvec_gd = fill(r_tilde, (n_elems + 1,))
    kvec_gd[1] = -1 # Open knot vector
    kvec_gd[end] = -1

    return BSplineSpace(patch, p_tilde, kvec_gd)
end

"""
    _create_approx_C1_patch_constituent_spaces(patch_idx::Int, patch_tp_space::TensorProductSpace{2, BSplineSpace, BSplineSpace}, p_tilde_lr::Int, r_tilde_lr::Int, p_tilde_bt::Int, r_tilde_bt::Int)

Creates all spaces needed on the given patchduring the construction of the approximate C1 space.

# Arguments
- `patch_idx::Int`: The index of the current patch.
- `patch_tp_space::TensorProductSpace{2, BSplineSpace, BSplineSpace}`: The tensor product space on the patch.
- `p_tilde_lr::Int`: Polynomial degree of the gluing data space on the left and right edges.
- `r_tilde_lr::Int`: Regularity of the gluing data space on the left and right edges.
- `p_tilde_bt::Int`: Polynomial degree of the gluing data space on the bottom and top edges.
- `r_tilde_bt::Int`: Regularity of the gluing data space on the bottom and top edges.

# Returns
- `Sbt_plus::BSplineSpace{1}`: Spline space for the bottom and top edges, outer ring.
- `Sbt_minus::BSplineSpace{1}`: Spline space for the bottom and top edges, inner ring.
- `Slr_plus::BSplineSpace{1}`: Spline space for the left and right edges, outer ring.
- `Slr_minus::BSplineSpace{1}`: Spline space for the left and right edges, inner ring.
- `Sgd_bt::BSplineSpace{1}`: Spline space for the bottom-top gluing data.
- `Sgd_lr::BSplineSpace{1}`: Spline space for the left-right gluing data.
- `S1_Slr_plus::TensorProductSpace`: Tensor product space for the edge functions in the left-right direction.
- `S1_Slr_minus_Sgd::TensorProductSpace`: Tensor product space for the edge functions in the left-right direction with gluing data.
- `S2_Sbt_plus::TensorProductSpace`: Tensor product space for the edge functions in the bottom-top direction.
- `S2_sbt_minus_Sgd::TensorProductSpace`: Tensor product space for the edge functions in the bottom-top direction with gluing data.
"""
function _create_approx_C1_patch_constituent_spaces(
    patch_idx::Int,
    patch_tp_space::TensorProductSpace,
    p_tilde_lr::Int,
    r_tilde_lr::Int,
    p_tilde_bt::Int,
    r_tilde_bt::Int,
)
    # Get constituent function spaces of the tp space.
    bspl_1 = patch_tp_space.constituent_spaces[1]
    bspl_2 = patch_tp_space.constituent_spaces[2]

    # Get polynomial degree in each direction.
    p1 = get_polynomial_degree(bspl_1)
    p2 = get_polynomial_degree(bspl_2)

    # Check if the polynomial degree for the patch interior spaces
    # is large enough.
    if p1 <= 1 || p2 <= 1
        msg1 = "Minimal polynomial degrees for the interior basis functions must be greater than or equal to 2."
        msg2 = " Instead, the degrees on patch $patch_idx are ($p1, $p2)."
        throw(ArgumentError(msg1 * msg2))
    end

    # Check if the regularities are large enough as well (both for
    # open and closed knot vectors).
    rvec1 = get_regularity_vector(bspl_1)
    if rvec1[1] == -1 && rvec1[end] == -1
        if any(rvec1[2:(end - 1)] .< 1) || any(rvec1[2:(end - 1)] .> p1 - 1)
            # Open knot vector.
            msg1 = "The regularity r must satisfy 1 <= r <= p-1."
            msg2 = " Instead, the regularity vector (in direction 1) on patch $patch_idx is $rvec1."
            throw(ArgumentError(msg1 * msg2))
        end
    else
        if any(rvec1 .< 1) || any(rvec1 .> p1 - 1)
            msg1 = "The regularity r must satisfy 1 <= r <= p-1."
            msg2 = " Instead, the regularity vector (in direction 1) on patch $patch_idx is $rvec1."
            throw(ArgumentError(msg1 * msg2))
        end
    end
    rvec2 = get_regularity_vector(bspl_2)
    if rvec2[1] == -1 && rvec2[end] == -1
        if any(rvec2[2:(end - 1)] .< 1) || any(rvec2[2:(end - 1)] .> p2 - 1)
            # Open knot vector.
            msg1 = "The regularity r must satisfy 1 <= r <= p-1."
            msg2 = " Instead, the regularity vector (in direction 2) on patch $patch_idx is $rvec."
            throw(ArgumentError(msg1 * msg2))
        end
    else
        if any(rvec2 .< 1) || any(rvec2 .> p2 - 1)
            msg1 = "The regularity r must satisfy 1 <= r <= p-1."
            msg2 = " Instead, the regularity vector (in direction 2) on patch $patch_idx is $rvec."
            throw(ArgumentError(msg1 * msg2))
        end
    end

    # Create spaces S+ and S-, which are spline spaces used for the
    # (approximation) of the trace at the edge and directional
    # derivative in the direction of the approximate normal, resp.
    # Left and right
    kveclr_plus = fill(p2 - 1, (get_num_elements(bspl_2) + 1,))
    kveclr_plus[1] = -1 # Open knot vector
    kveclr_plus[end] = -1
    Slr_plus = BSplineSpace(get_patch(bspl_2), p2, kveclr_plus)

    kveclr_minus = fill(p2 - 2, (get_num_elements(bspl_2) + 1,))
    kveclr_minus[1] = -1 # Open knot vector
    kveclr_minus[end] = -1
    Slr_minus = BSplineSpace(get_patch(bspl_2), p2 - 1, kveclr_minus)

    # Bottom and top
    kvecbt_plus = fill(p1 - 1, (get_num_elements(bspl_1) + 1,))
    kvecbt_plus[1] = -1 # Open knot vector
    kvecbt_plus[end] = -1
    Sbt_plus = BSplineSpace(get_patch(bspl_1), p1, kvecbt_plus)

    kvecbt_minus = fill(p1 - 2, (get_num_elements(bspl_1) + 1,))
    kvecbt_minus[1] = -1 # Open knot vector
    kvecbt_minus[end] = -1
    Sbt_minus = BSplineSpace(get_patch(bspl_1), p1 - 1, kvecbt_minus)

    # Create spaces for the gluing data. If the default values are used,
    # these spaces are the same as the S+ spaces above.
    Sgd_lr = _construct_approx_gluing_data_space(
        p_tilde_lr, r_tilde_lr, p2, get_num_elements(bspl_2), get_patch(bspl_2)
    )
    Sgd_bt = _construct_approx_gluing_data_space(
        p_tilde_bt, r_tilde_bt, p1, get_num_elements(bspl_1), get_patch(bspl_1)
    )

    # For the edge functions, we need the derivative and product space,
    # after which we can create the tensor product spaces.
    #Slr_plus_der = create_derivative_space(Slr_plus) # == Slr_minus
    Slr_minus_Sgd = create_product_space(Slr_minus, Sgd_lr)

    #Sbt_plus_der = create_derivative_space(Sbt_plus) # == Sbt_minus
    Sbt_minus_Sgd = create_product_space(Sbt_minus, Sgd_bt)

    # Create the tensor product spaces needed for the evaluation of the
    # edge functions.
    S1_Slr_plus = TensorProductSpace((bspl_1, Slr_plus))
    S1_Slr_minus_Sgd = TensorProductSpace((bspl_1, Slr_minus_Sgd))

    S2_Sbt_plus = TensorProductSpace((Sbt_plus, bspl_2))
    S2_sbt_minus_Sgd = TensorProductSpace((Sbt_minus_Sgd, bspl_2))

    return Sbt_plus,
    Sbt_minus,
    Slr_plus,
    Slr_minus,
    Sgd_bt,
    Sgd_lr,
    S1_Slr_plus,
    S1_Slr_minus_Sgd,
    S2_Sbt_plus,
    S2_sbt_minus_Sgd
end

"""
    _shuffle_global_dofs(global_dofs::Vector{Int}, local_dofs::Vector{Int}, N_outer_ring::Int, which::String)

Reorders the global dofs for inherited dofs.

Note that the shuffled vertices will always contain 6 values. If no
connection is made, which can happen if the vertices are not
interpolated, the value will be 0.

# Arguments
- `global_dofs::Vector{Int}`: The global dofs of the neighbouring patch.
- `local_dofs::Vector{Int}`: The local dofs of the neighbouring patch.
- `N_outer_ring::Int`: The number of dofs in the outer ring on the
                       neighbouring patch.
- `which::String`: The type of dofs. Either 'edge' or 'vertexij', where
                   i is the vertex dof group and j the edge group that
                   is being processed (which thus indicates from which
                   side the dofs are inherited.). Note that these dof
                   groups are with respect to the current patch.
- `interpolate_vertices::Bool`: If true, the vertices are interpolated
                                from the neighbouring patch. This
                                changes the connectivity of the vertices.

# Returns
- `::Vector{Int}`: The reordered global dofs.
"""
function _shuffle_global_dofs(
    global_dofs::Vector{Int},
    local_dofs::Vector{Int},
    N_outer_ring::Int,
    which::String,
    interpolate_vertices::Bool,
)
    if which == "edge" || which == "edgeB" || which == "edgeC"
        # Count the number of dofs in the inner and outer ring.
        outer_ring_dofs = Int[]
        inner_ring_dofs = Int[]
        for dof_i in eachindex(global_dofs, local_dofs)
            if local_dofs[dof_i] <= N_outer_ring
                push!(outer_ring_dofs, global_dofs[dof_i])
            else
                push!(inner_ring_dofs, global_dofs[dof_i])
            end
        end

        if which[end] == 'B'
            outer_ring_dofs2 = outer_ring_dofs[1:(end - 1)]
            inner_ring_dofs2 = vcat(inner_ring_dofs, outer_ring_dofs[end])
            new_order = Vector{Int}(undef, length(global_dofs))
            for dof_i in length(outer_ring_dofs2):-1:1
                new_order[length(outer_ring_dofs2) - dof_i + 1] = outer_ring_dofs2[dof_i]
            end
            for dof_i in length(inner_ring_dofs2):-1:1
                new_order[length(outer_ring_dofs2) + length(inner_ring_dofs2) - dof_i + 1] = inner_ring_dofs2[dof_i]
            end

            return new_order
        elseif which[end] == 'C'
            new_order = Vector{Int}(undef, length(global_dofs))
            new_order[1] = outer_ring_dofs[2]
            new_order[2] = outer_ring_dofs[1]
            for dof_i in length(outer_ring_dofs):-1:3
                new_order[length(outer_ring_dofs) - dof_i + 3] = outer_ring_dofs[dof_i]
            end
            if length(inner_ring_dofs) > 0
                new_order[length(outer_ring_dofs) + 1] = inner_ring_dofs[1]
                for dof_i in length(inner_ring_dofs):-1:2
                    new_order[length(outer_ring_dofs) + length(inner_ring_dofs) - dof_i + 2] = inner_ring_dofs[dof_i]
                end
            end
            # new_order[length(outer_ring_dofs)+1] = inner_ring_dofs[1]
            # for dof_i in length(inner_ring_dofs):-1:2
            #     new_order[length(outer_ring_dofs)+length(inner_ring_dofs)-dof_i+2] = inner_ring_dofs[dof_i]
            # end

            return new_order
        else
            new_order = Vector{Int}(undef, length(global_dofs))
            # Flip the order of the dofs, because the neighbouring patches
            # are always ordered in the opposite direction.
            for dof_i in length(outer_ring_dofs):-1:1
                new_order[length(outer_ring_dofs) - dof_i + 1] = outer_ring_dofs[dof_i]
            end
            for dof_i in length(inner_ring_dofs):-1:1
                new_order[length(outer_ring_dofs) + length(inner_ring_dofs) - dof_i + 1] = inner_ring_dofs[dof_i]
            end

            return new_order
        end

    elseif which[1:(end - 2)] == "vertex"
        # There are always 6 dofs per vertex, so we can just shuffle the
        # dofs manually, though how to shuffle them depends on the
        # vertex. The number at the end indicates which vertex is
        # receiving the new numbers.
        if which[end - 1] == '1'
            if which[end] == '4'
                # Bottom left vertex, inherit from the left.
                if interpolate_vertices
                    return [
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[6],
                    ]
                else
                    return [
                        global_dofs[3],
                        global_dofs[2],
                        0,
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[6],
                    ]
                end
            else # which[end] == '2'
                # Bottom left vertex, inherit from the bottom.
                if interpolate_vertices
                    return [
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[6],
                    ]
                else
                    return [
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        0,
                        global_dofs[4],
                        global_dofs[6],
                    ]
                end
            end

        elseif which[end - 1] == '3'
            if which[end] == '6'
                # Bottom right vertex, inherit from the right.
                if interpolate_vertices
                    return [
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[6],
                    ]
                else
                    return [
                        0,
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[6],
                    ]
                end
            else # which[end] == '2'
                # Bottom right vertex, inherit from the bottom.
                if interpolate_vertices
                    return [
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[6],
                    ]
                else
                    return [
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[3],
                        global_dofs[2],
                        0,
                        global_dofs[6],
                    ]
                end
            end

        elseif which[end - 1] == '9'
            if which[end] == '6'
                # Top right vertex, inherit from the right.
                if interpolate_vertices
                    return [
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[6],
                    ]
                else
                    return [
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[3],
                        global_dofs[2],
                        0,
                        global_dofs[6],
                    ]
                end
            else # which[end] == '8'
                # Top right vertex, inherit from the top.
                if interpolate_vertices
                    return [
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[6],
                    ]
                else
                    return [
                        0,
                        global_dofs[4],
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[6],
                    ]
                end
            end

        elseif which[end - 1] == '7'
            if which[end] == '4'
                # Top left vertex, inherit from the left.
                if interpolate_vertices
                    return [
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[6],
                    ]
                else
                    return [
                        0,
                        global_dofs[4],
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[6],
                    ]
                end
            else # which[end] == '8'
                # Top left vertex, inherit from the top.
                if interpolate_vertices
                    return [
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[5],
                        global_dofs[4],
                        global_dofs[6],
                    ]
                else
                    return [
                        global_dofs[3],
                        global_dofs[2],
                        global_dofs[1],
                        global_dofs[5],
                        0,
                        global_dofs[6],
                    ]
                end
            end

        else
            throw(
                ArgumentError(
                    "The vertex number must be 1, 3, 7, or 9, not $(which[end-1])."
                ),
            )
        end

    else
        throw(
            ArgumentError(
                "The argument 'which' must be either 'edge' or 'vertex..', not $which."
            ),
        )
    end
end

"""
    _project_gluing_data(spline_space::S, gluing_data::F, which::String="Greville") where {S <: AbstractFESpace{1}, F <: Function}

Project the gluing data onto the given (approximate gluing data) spline space.

# Arguments
- `spline_space::S`: The spline space on which to project the gluing data.
- `gluing_data::F`: The gluing data to be projected.
- `which::String`: The type of projection. Currently only 'interpolate'
                   is supported.

# Returns
- `::Vector{Float64}`: The coefficients of the projected gluing data.
"""
function _project_gluing_data(
    spline_space::BSplineSpace, gluing_data::F, which::String="interpolate"
) where {F <: Function}
    if which == "interpolate"
        # Project the gluing data onto the spline space by interpolating
        # at the Greville points.
        greville_points = get_greville_points(spline_space)[1]

        # Evaluate the b-splines at the Greville points.
        bspline_eval = _create_interpolation_matrix(spline_space, greville_points)

        # Solve the interpolation problem.
        return bspline_eval \ gluing_data.(greville_points)

    else
        throw(ArgumentError("The argument 'which' must be 'interpolate', not $which."))
    end
end

"""
    create_approximate_C1_space()

Constructs an approximate ``C^1`` multi-patch splines spaces from given
tensor product spaces, patch connectivity, and gluing data.

# Arguments
- `patch_tp_spaces::NTuple{2, F<:AbstractFESpace{manifold_dim}}`: Tensor-product b-spline space per patch.

# Returns
- `::ApproximateC1Space`: The constructed Approximate ``C^1`` space.

# Throws
- `ArgumentError`: If the tensor product regularity on any patch does not satisfy r >= 1 (excluding endpoint of an open knot vector).
- `ArgumentError`: If the tensor product polynomial degree on any patch does not satisfy p >= 2.

# Notes and References
The `patch_tp_spaces` are used to define the approximate C1 space. The
interior basis function are the same as the TP space, while the edge and
vertex functions are a linear combination of the tp space functions and
univariate splines spaces on the boundaries of the same or one degree
lower (always with maximal regularity). The degree and elements of these
additional spaces are inherited from the tp spaces.

The approximate ``C^1`` construction was developed in [Weinmuller2021](@cite)
and [Weinmuller2022](@cite). Also consult these papers for more details
on the construction.
"""
function create_approximate_C1_space(
    patch_tp_spaces::NTuple{2, F},
    patch_connectivity::NTuple{2, NTuple{4, NTuple{2, Int}}},
    gluing_data::NTuple{2, NTuple{4, NTuple{2, Function}}},
    p_tilde_lr::Int=-1,
    r_tilde_lr::Int=-2,
    p_tilde_bt::Int=-1,
    r_tilde_bt::Int=-2,
) where {manifold_dim, F <: AbstractFESpace{manifold_dim}}

    # Start by creating patch local information (local numbering, the
    # intermediate spaces, and the 5 local basis TP spaces).
    local_dof_partition_per_patch = Vector{Vector{Vector{Int}}}(undef, 2)
    constituent_spaces_per_patch_vec = Vector{
        Tuple{
            TensorProductSpace{manifold_dim}, # Interior space
            BSplineSpace, # Interface space bottom-top +
            BSplineSpace, # Interface space bottom-top -
            BSplineSpace, # Interface space left-right +
            BSplineSpace, # Interface space left-right -
            BSplineSpace, # Gluing data space bottom-top
            BSplineSpace, # Gluing data space left-right
            TensorProductSpace{manifold_dim}, # Edge TP 1LR space
            TensorProductSpace{manifold_dim}, # Edge TP 1LR-gd space
            TensorProductSpace{manifold_dim}, # Edge TP 2BT space
            TensorProductSpace{manifold_dim},
        },
    }(
        undef, 2
    ) # Edge TP 2BT-gd space
    constituent_spaces_per_patch_vec_tp = Vector{NTuple{5, AbstractFESpace{manifold_dim}}}(
        undef, 2
    ) # Only the TP spaces.
    N_outer_ring_per_patch = Vector{Int}(undef, 2)

    # Loop over each patch to create the local numbering and spaces.
    for patch_idx in 1:1:2
        # Tensor product space on the patch. The other spaces are
        # directly derived from this one.
        tp_space = patch_tp_spaces[patch_idx]

        # Create all intermediate spaces used in the approx. C1 constr.
        (Sbt_plus, Sbt_minus, Slr_plus, Slr_minus, Sgd_bt, Sgd_lr, S1_Slr_plus, S1_Slr_minus_Sgd, S2_Sbt_plus, S2_sbt_minus_Sgd) = _create_approx_C1_patch_constituent_spaces(
            patch_idx, tp_space, p_tilde_lr, r_tilde_lr, p_tilde_bt, r_tilde_bt
        )

        # Get constituent function spaces of the tp space.
        bspl_1 = tp_space.constituent_spaces[1]
        bspl_2 = tp_space.constituent_spaces[2]

        # Number of basis functions per dimension.
        N1 = get_num_basis(bspl_1)
        N2 = get_num_basis(bspl_2)
        Nlr_plus = get_num_basis(Slr_plus)
        Nlr_minus = get_num_basis(Slr_minus)
        Nbt_plus = get_num_basis(Sbt_plus)
        Nbt_minus = get_num_basis(Sbt_minus)

        constituent_spaces_per_patch_vec[patch_idx] = (
            tp_space,
            Sbt_plus,
            Sbt_minus,
            Slr_plus,
            Slr_minus,
            Sgd_bt,
            Sgd_lr,
            S1_Slr_plus,
            S1_Slr_minus_Sgd,
            S2_Sbt_plus,
            S2_sbt_minus_Sgd,
        )
        constituent_spaces_per_patch_vec_tp[patch_idx] = (
            tp_space, S1_Slr_plus, S1_Slr_minus_Sgd, S2_Sbt_plus, S2_sbt_minus_Sgd
        )
        #N_outer_ring_per_patch[patch_idx] = Nbt_plus + (Nlr_plus - 1) + (Nbt_plus - 1) + (Nlr_plus - 2)
        N_outer_ring_per_patch[patch_idx] = N1 + Nlr_plus - 1 + N1 - 1 + N2 - 2

        # Compute the total number of basis functions in the approx C1 space.
        # Non-shared edges, vertices, and interior: As TP
        # Edge (shared): Nlr_plus + Nlr_minus

        # Create the local dof partition for the approx. C1 space. Note
        # that the number of dofs is not the same as that of the tp
        # space and that the dofs are distributed and numbered differently.
        local_dof_partition_per_patch[patch_idx] = _create_approx_C1_local_dof_partition_patch(
            patch_idx, N1, N2, Nlr_plus, Nlr_minus
        )
    end

    # The constituent spaces including the 1D spaces are used in
    # defining the global dofs. Only the tensor product spaces are used
    # in the extraction and evaluation.
    constituent_spaces_per_patch_all = Tuple(constituent_spaces_per_patch_vec)
    constituent_spaces_per_patch_tps = Tuple(constituent_spaces_per_patch_vec_tp)

    # Create the global dof numbering from the local one, the global to
    # local dof dict, and the dof partition based on the global dofs.
    (global_to_local_dof_dict, global_dof_partition_per_patch, max_global_dof) = _create_global_dof_dicts(
        local_dof_partition_per_patch, patch_connectivity, Tuple(N_outer_ring_per_patch)
    )

    # Extraction matrices.
    extraction = _extract_approximateC1_to_constituent(
        constituent_spaces_per_patch_all,
        global_to_local_dof_dict,
        global_dof_partition_per_patch,
        max_global_dof,
        gluing_data,
        "interpolate",
    )

    # Hacky way to create the local to global dof dict, because I didn't
    # create it in the global dof assignment function.
    # TODO: Fix this.
    local_to_global_dof_dict = Dict{Tuple{Int, Int}, Int}()
    for (global_dof, local_dof_dict) in global_to_local_dof_dict
        for (patch_idx, local_dof) in local_dof_dict
            local_to_global_dof_dict[(patch_idx, local_dof)] = global_dof
        end
    end

    return ApproximateC1Space(
        constituent_spaces_per_patch_tps,
        extraction,
        global_dof_partition_per_patch,
        local_dof_partition_per_patch,
        Tuple(N_outer_ring_per_patch),
        global_to_local_dof_dict,
        local_to_global_dof_dict,
    )
end

# Helper functions for the constructor.
"""
    _create_approx_C1_local_dof_partition_patch(N_total_C1::Int, N_C1_int::Int, N_C1_edge_bt::Int, N_C1_edge_lr::Int, N_C1_vertex::Int, Nbt_plus::Int, Nbt_minus::Int, Nlr_plus::Int, Nlr_minus::Int)

Creates the local dof partition for the approximate C1 space on a patch.

# Arguments
- `N_total_C1::Int`: Total number of basis functions in the approx C1 space on this patch.
- `N_C1_int::Int`: Number of basis functions in the interior of the patch.
- `N_C1_edge_bt::Int`: Number of basis functions on the bottom and top edges.
- `N_C1_edge_lr::Int`: Number of basis functions on the left and right edges.
- `N_C1_vertex::Int`: Number of basis functions at the vertices.
- `Nbt_plus::Int`: Number of basis functions in the bottom and top spaces, outer ring.
- `Nbt_minus::Int`: Number of basis functions in the bottom and top spaces, inner ring.
- `Nlr_plus::Int`: Number of basis functions in the left and right spaces, outer ring.
- `Nlr_minus::Int`: Number of basis functions in the left and right spaces, inner ring.

# Returns
- `::Vector{Vector{Int}}`: The local dof partition.
"""
function _create_approx_C1_local_dof_partition_patch(
    patch_idx::Int, N1::Int, N2::Int, Nlr_plus::Int, Nlr_minus::Int
)
    # Create vectors for each dof group.
    interior_dofs = Int[]

    edge_B_dofs = Int[]
    edge_L_dofs = Int[]
    edge_R_dofs = Int[]
    edge_T_dofs = Int[]

    vertex_BL_dofs = Int[]
    vertex_BR_dofs = Int[]
    vertex_TL_dofs = Int[]
    vertex_TR_dofs = Int[]

    Ntotal = N1 * N2 - 2 * N2 + Nlr_plus + Nlr_minus
    if patch_idx == 1
        # Patch 1, only need to modify the right side.

        # Assign the dofs.
        for i in 1:1:Ntotal
            # Outer loop
            if i <= N1 - 2
                # Bottom edge.
                if i == 1
                    push!(vertex_BL_dofs, i)
                else
                    push!(edge_B_dofs, i)
                end
            elseif i <= N1 + Nlr_plus
                # Right edge. (Two at the bottom, all right, two top)
                push!(edge_R_dofs, i)
            elseif i <= N1 + Nlr_plus + N1 - 2
                # Top edge.
                if i == N1 + Nlr_plus + N1 - 2
                    push!(vertex_TL_dofs, i)
                else
                    push!(edge_T_dofs, i)
                end
            elseif i <= N1 + Nlr_plus + N1 + N2 - 4
                # Left edge.
                push!(edge_L_dofs, i)

                # Inner loop
            elseif i <= N1 + Nlr_plus + N1 + N2 - 4 + N1 - 3
                # Bottom edge.
                push!(edge_B_dofs, i)
            elseif i <= N1 + Nlr_plus + N1 + N2 - 4 + N1 - 3 + Nlr_minus - 2
                # Right edge.
                push!(edge_R_dofs, i)
            elseif i <= N1 + Nlr_plus + N1 + N2 - 4 + N1 - 3 + Nlr_minus - 2 + N1 - 3
                # Top edge.
                push!(edge_T_dofs, i)
            elseif i <=
                N1 + Nlr_plus + N1 + N2 - 4 + N1 - 3 + Nlr_minus - 2 + N1 - 3 + N2 - 4
                # Left edge.
                push!(edge_L_dofs, i)

                # Interior
            else
                push!(interior_dofs, i)
            end
        end

    else
        # Patch 2, only need to modify the left side.

        # Assign the dofs.
        for i in 1:1:Ntotal
            # Outer loop
            if i <= 2
                # Left edge at the bottom
                push!(edge_L_dofs, i)
            elseif i <= N1
                # Bottom edge.
                if i == N1
                    push!(vertex_BR_dofs, i)
                else
                    push!(edge_B_dofs, i)
                end
            elseif i <= N1 + N2 - 1
                # Right edge.
                if i == N1 + N2 - 1
                    push!(vertex_TR_dofs, i)
                else
                    push!(edge_R_dofs, i)
                end
            elseif i <= N1 + N2 + N1 - 4
                # Top edge.
                push!(edge_T_dofs, i)
            elseif i <= N1 + N2 + N1 + Nlr_plus - 4
                # Left edge, two top as well.
                push!(edge_L_dofs, i)

                # Inner loop
            elseif i == N1 + N2 + N1 + Nlr_plus - 4 + 1
                # Left edge.
                push!(edge_L_dofs, i)
            elseif i <= N1 + N2 + N1 + Nlr_plus - 4 + N1 - 2
                # Bottom edge.
                push!(edge_B_dofs, i)
            elseif i <= N1 + N2 + N1 + Nlr_plus - 4 + N1 - 2 + N2 - 2 - 1
                # Right edge.
                push!(edge_R_dofs, i)
            elseif i <= N1 + N2 + N1 + Nlr_plus - 4 + N1 - 2 + N2 - 2 + N1 - 2 - 3
                # Top edge.
                push!(edge_T_dofs, i)
            elseif i == N1 + N2 + N1 + Nlr_plus - 4 + N1 - 2 + N2 - 2 + N1 - 2 - 2
                # Left edge.
                push!(edge_L_dofs, i)
            elseif i <=
                N1 + N2 + N1 + Nlr_plus - 4 + N1 - 2 + N2 - 2 + N1 - 2 + Nlr_minus - 3 -
                   3
                # Left edge.
                push!(edge_L_dofs, i)

                # Interior
            else
                push!(interior_dofs, i)
            end
        end
    end

    return [
        vertex_BL_dofs,
        edge_B_dofs,
        vertex_BR_dofs,
        edge_L_dofs,
        interior_dofs,
        edge_R_dofs,
        vertex_TL_dofs,
        edge_T_dofs,
        vertex_TR_dofs,
    ]
end

"""
    _create_global_dof_dicts(local_dof_partition::Vector{Vector{Vector{Int}}}, patch_connectivity::NTuple{2, NTuple{4, NTuple{2, Int}}}, N_outer_ring_per_patch::NTuple{2, Int})

Create the global numbering based on the patch connectivity and patch-
wise dof_partition.

# Arguments
- `local_dof_partition::Vector{Vector{Vector{Int}}}`: The local dof
  partition per patch (the dof partition using the local dof numbers).
- `patch_connectivity::NTuple{2, NTuple{4, NTuple{2, Int}}}`:
  The connectivity between the patches.
- `N_outer_ring_per_patch::NTuple{2, Int}`: The number of dofs
  in the outer ring per patch.
- `interpolate_vertices::Bool`: Whether to interpolate the vertices.

# Returns
- `global_to_local_dof_dict::Dict{Int, Dict{Int, Int}}`: The global dof
  to local dof dict {global_dof -> {patch -> local_dof}}.
- `global_dof_partition::Vector{Vector{Vector{Int}}}`: The global dof
  partition per patch (the dof partition using the global dof numbers).
- `max_global_dof::Int`: The maximum global dof number.
"""
function _create_global_dof_dicts(
    local_dof_partition::Vector{Vector{Vector{Int}}},
    patch_connectivity::NTuple{2, NTuple{4, NTuple{2, Int}}},
    N_outer_ring_per_patch::NTuple{2, Int},
)
    global_to_local_dof_dict = Dict{Int, Dict{Int, Int}}()
    global_dof_partition = Vector{Vector{Vector{Int}}}(undef, 2)

    global_dof = 1
    for patch_i in 1:1:2
        # Allocate the dof partition for this patch.
        global_dof_partition[patch_i] = Vector{Vector{Int}}(undef, 9)

        # Keep track of which dof groups on a patch have already been
        # assigned. Relevant for patch_i > 1.
        dof_groups_assigned = Vector{Int}(undef, 0)

        if patch_i == 1
            # First patch.
            for dof_group_i in eachindex(local_dof_partition[patch_i])
                global_dof_partition[patch_i][dof_group_i] = Vector{Int}(
                    undef, size(local_dof_partition[patch_i][dof_group_i])
                )

                for (idx, local_dof_i) in
                    enumerate(local_dof_partition[patch_i][dof_group_i])
                    # Directly assign the new global dof numbers to all
                    # dofs, as nothing is shared yet.
                    global_dof_partition[patch_i][dof_group_i][idx] = global_dof

                    global_to_local_dof_dict[global_dof] = Dict{Int, Int}(
                        patch_i => local_dof_i
                    )
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
                    # The order of the newly added dofs matters, hence
                    # the shuffle.
                    global_dof_partition[patch_i][vertex_i1] = Int[]

                    global_dof_partition[patch_i][edge_nr1] = _shuffle_global_dofs(
                        global_dof_partition[neighbour_patch][edge_nr2],
                        local_dof_partition[neighbour_patch][edge_nr2],
                        N_outer_ring_per_patch[neighbour_patch],
                        "edge",
                        false,
                    )

                    global_dof_partition[patch_i][vertex_i2] = Int[]

                    # Update the global to local dof dict. The global
                    # dofs are now already defined.
                    for dof_group_i in [vertex_i1, edge_nr1, vertex_i2]
                        for (global_dof_i, local_dof_i) in zip(
                            global_dof_partition[patch_i][dof_group_i],
                            local_dof_partition[patch_i][dof_group_i],
                        )
                            if global_dof_i == 0
                                # The dof was not shared, so assign a
                                # new value.
                                idx = findfirst(
                                    global_dof_partition[patch_i][dof_group_i] .== 0
                                )
                                global_dof_partition[patch_i][dof_group_i][idx] = global_dof

                                global_to_local_dof_dict[global_dof] = Dict{Int, Int}(
                                    patch_i => local_dof_i
                                )
                                global_dof += 1

                            else
                                global_to_local_dof_dict[global_dof_i][patch_i] =
                                    local_dof_i
                            end
                        end
                    end

                    push!(dof_groups_assigned, vertex_i1, edge_nr1, vertex_i2)
                end
            end

            # Add the non-shared dofs.
            for dof_group_i in eachindex(local_dof_partition[patch_i])
                # Only add the dofs that have not been assigned yet.
                if !(dof_group_i in dof_groups_assigned)
                    global_dof_partition[patch_i][dof_group_i] = Vector{Int}(
                        undef, size(local_dof_partition[patch_i][dof_group_i])
                    )

                    for (idx, local_dof_i) in
                        enumerate(local_dof_partition[patch_i][dof_group_i])
                        global_dof_partition[patch_i][dof_group_i][idx] = global_dof

                        # All the global dofs are unique again
                        global_to_local_dof_dict[global_dof] = Dict{Int, Int}(
                            patch_i => local_dof_i
                        )
                        global_dof += 1
                    end
                end
            end
        end
    end

    return global_to_local_dof_dict, global_dof_partition, global_dof - 1
end

"""
    _extract_approximateC1_to_constituent(patch_spaces::NTuple{2, Tuple{AbstractFESpace{manifold_dim}, AbstractFESpace{1}, AbstractFESpace{1}, AbstractFESpace{1}, AbstractFESpace{1}, AbstractFESpace{1}, AbstractFESpace{1}, AbstractFESpace{manifold_dim}, AbstractFESpace{manifold_dim}, AbstractFESpace{manifold_dim}, AbstractFESpace{manifold_dim}}},
                                          global_to_local_dof_dict::Dict{Int, Dict{Int, Int}},
                                          global_dof_partition::Vector{Vector{Vector{Int}}},
                                          max_global_dof::Int,
                                          gluing_data::NTuple{2, NTuple{4, NTuple{2, Function}}},
                                          create_vertex_spaces::Bool) where {manifold_dim}

Compute the extraction coefficients of approximate C1 basis functions in
terms of the 5 constituent spaces per patch.

# Arguments
- `patch_spaces::NTuple{2,
                        Tuple{AbstractFESpace{manifold_dim}, # Interior space
                        AbstractFESpace{1},                  # Interface space bottom-top +
                        AbstractFESpace{1},                  # Interface space bottom-top -
                        AbstractFESpace{1},                  # Interface space left-right +
                        AbstractFESpace{1},                  # Interface space left-right -
                        AbstractFESpace{1},                  # Gluing data space bottom-top
                        AbstractFESpace{1},                  # Gluing data space left-right
                        AbstractFESpace{manifold_dim},       # Edge TP 1LR space
                        AbstractFESpace{manifold_dim},       # Edge TP 1LR-gd space
                        AbstractFESpace{manifold_dim},       # Edge TP 2BT space
                        AbstractFESpace{manifold_dim}}},     # Edge TP 2BT-gd space
- `global_to_local_dof_dict::Dict{Int, Dict{Int, Int}}`: The global to
local dof dict {global_dof -> {patch -> local_dof}}.
- `global_dof_partition::Vector{Vector{Vector{Int}}}`: The global dof
partition per patch (the dof partition using the global dof numbers).
- `max_global_dof::Int`: The maximum global dof number.
- `gluing_data::NTuple{2, NTuple{4, NTuple{2, Function}}}`: The
gluing data for each patch.
- `create_vertex_spaces::Bool`: Whether to create the vertex spaces by
C2 interpolation at the vertices.

# Returns
- `ExtractionOperator`: The extraction operator containing the coefficients.
"""
function _extract_approximateC1_to_constituent(
    patch_spaces::NTuple{
        2,
        Tuple{
            TensorProductSpace{manifold_dim},   # Interior space
            BSplineSpace,              # Interface space bottom-top +
            BSplineSpace,              # Interface space bottom-top -
            BSplineSpace,              # Interface space left-right +
            BSplineSpace,              # Interface space left-right -
            BSplineSpace,              # Gluing data space bottom-top
            BSplineSpace,              # Gluing data space left-right
            TensorProductSpace{manifold_dim},   # Edge TP 1LR space
            TensorProductSpace{manifold_dim},   # Edge TP 1LR-gd space
            TensorProductSpace{manifold_dim},   # Edge TP 2BT space
            TensorProductSpace{manifold_dim},
        },
    }, # Edge TP 2BT-gd space
    global_to_local_dof_dict::Dict{Int, Dict{Int, Int}},
    global_dof_partition::Vector{Vector{Vector{Int}}},
    max_global_dof::Int,
    gluing_data::NTuple{2, NTuple{4, NTuple{2, Function}}},
    product_algorithm::String="interpolate",
) where {manifold_dim}

    # Count the number of elements and number of local basis functions.
    num_elements = 0
    local_basis_functions_total = 0
    local_basis_functions_per_patch = zeros(Int, 2)
    elements_per_patch = Vector{Int}(undef, 2)
    for patch_idx in 1:1:2
        # The number of tp elements equals that of the approx C1 space
        # on a given patch.
        elements_per_patch[patch_idx] = get_num_elements(patch_spaces[patch_idx][1])
        num_elements += elements_per_patch[patch_idx]

        for local_space in patch_spaces[patch_idx][[1, 8, 9, 10, 11]]
            num_basis = get_num_basis(local_space)
            local_basis_functions_total += num_basis
            local_basis_functions_per_patch[patch_idx] += num_basis
        end
    end

    # Build the global extraction matrix. The global dofs are on the
    # rows, the local dofs on the columns.
    global_extr_rows = Int[]
    global_extr_cols = Int[]
    global_extr_vals = Float64[]

    # The constituent spaces will be ordered per patch, so to use the
    # right space we need to offset the local indices with the number of
    # constituent basis functions per patch (cumulatively).
    offset = vcat(0, cumsum(local_basis_functions_per_patch[1:(end - 1)]))
    for patch_idx in 1:1:2
        # Loop over the patches and get all constituent spaces.
        tp_space = patch_spaces[patch_idx][1]
        Sbt_plus = patch_spaces[patch_idx][2]
        Sbt_minus = patch_spaces[patch_idx][3]
        Slr_plus = patch_spaces[patch_idx][4]
        Slr_minus = patch_spaces[patch_idx][5]
        Sgd_bt = patch_spaces[patch_idx][6]
        Sgd_lr = patch_spaces[patch_idx][7]
        tp_lr_1plus = patch_spaces[patch_idx][8]
        tp_lr_prod = patch_spaces[patch_idx][9]
        tp_bt_2plus = patch_spaces[patch_idx][10]
        tp_bt_prod = patch_spaces[patch_idx][11]

        N1, N2 = get_constituent_num_basis(tp_space)
        N_outer_ring = N1 + get_num_basis(Slr_plus) + N1 + N2 - 4

        # Loop over the global basis function partition on the patch.
        for dof_part_division in eachindex(global_dof_partition[patch_idx])

            # Manually identify the vertex dofs. They are never shared
            # and correspond directly to the tensor product basis functions.
            if patch_idx == 1 && (dof_part_division == 1 || dof_part_division == 7)
                if dof_part_division == 1
                    global_dof = global_dof_partition[patch_idx][dof_part_division][1]
                    push!(global_extr_rows, global_dof)
                    push!(global_extr_cols, 1 + offset[patch_idx])
                    push!(global_extr_vals, 1.0)
                else
                    global_dof = global_dof_partition[patch_idx][dof_part_division][1]
                    push!(global_extr_rows, global_dof)
                    push!(
                        global_extr_cols,
                        get_num_basis(tp_space) - N1 + 1 + offset[patch_idx],
                    )
                    push!(global_extr_vals, 1.0)
                end
            elseif patch_idx == 2 && (dof_part_division == 3 || dof_part_division == 9)
                if dof_part_division == 3
                    global_dof = global_dof_partition[patch_idx][dof_part_division][1]
                    push!(global_extr_rows, global_dof)
                    push!(global_extr_cols, N1 + offset[patch_idx])
                    push!(global_extr_vals, 1.0)
                else
                    global_dof = global_dof_partition[patch_idx][dof_part_division][1]
                    push!(global_extr_rows, global_dof)
                    push!(global_extr_cols, get_num_basis(tp_space) + offset[patch_idx])
                    push!(global_extr_vals, 1.0)
                end

            elseif dof_part_division == 5
                # Interior dofs. Interior dofs are simply the tensor
                # product basis functions. They are ordered in the same
                # way in the dof_partition, but the numbers do not match.
                tp_numbered_dofs = reshape(
                    1:1:get_num_basis(tp_space), get_constituent_num_basis(tp_space)
                )
                interior_tp_dofs = tp_numbered_dofs[3:(end - 2), 3:(end - 2)]

                for idx in eachindex(global_dof_partition[patch_idx][dof_part_division])
                    global_dof = global_dof_partition[patch_idx][dof_part_division][idx]
                    local_dof = Int(interior_tp_dofs[idx]) # Convert iterator element to actual Int value.

                    push!(global_extr_rows, global_dof)
                    push!(global_extr_cols, local_dof + offset[patch_idx])
                    push!(global_extr_vals, 1.0)
                end

            else
                # Edge dofs
                idxs_plus_left = 0
                idxs_plus_right = 0
                idxs_plus_bottom = 0
                idxs_plus_top = 0
                for idx in eachindex(global_dof_partition[patch_idx][dof_part_division])
                    # Note that idx also indicates if this global dofs
                    # is the first, second, etc. in the current division
                    # of the dof partition.
                    global_dof = global_dof_partition[patch_idx][dof_part_division][idx]

                    if dof_part_division == 4
                        # Left edge. Note that we start from the top
                        # with counting the dofs on this edge.

                        if patch_idx == 1
                            # The edge is not shared, so the local dofs
                            # are directly the tensor product basis functions.
                            if global_to_local_dof_dict[global_dof][patch_idx] <=
                                N_outer_ring
                                # Outer edge dofs.
                                local_dof = (N2 - idx - 1) * N1 + 1
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                                idxs_plus_left += 1
                            else
                                # Inner edge dofs.
                                local_dof = (N2 - (idx - idxs_plus_left) - 2) * N1 + 2
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                            end

                        else
                            # f4 = (b+ b1) + (b+ b2) + (alpha b- h/p b2) + (beta b+' h/p b2)

                            h = 1.0 / get_constituent_num_elements(tp_space)[1]  # Only true for uniform breakpoints.
                            p = get_constituent_polynomial_degree(tp_space)[1]

                            # Approximate the gluing data.
                            alpha_tilde_coeffs = _project_gluing_data(
                                Sgd_lr, gluing_data[patch_idx][4][1]
                            )
                            beta_tilde_coeffs = _project_gluing_data(
                                Sgd_lr, gluing_data[patch_idx][4][2]
                            )

                            # If the local dof on the current patch
                            # corresponding to the global dof is in the
                            # outer ring, it is a + dof. Only the dofs on
                            # the outer ring on the second row are - dofs.
                            if global_to_local_dof_dict[global_dof][patch_idx] <=
                               N_outer_ring &&
                                (!(idx == 1 || idx == 1 + get_num_basis(Slr_plus) + 1))
                                # + dofs, three terms.
                                if dof_part_division == 4
                                    idxs_plus_left += 1
                                end

                                # Counting from the top.
                                b_plus_idx = get_num_basis(Slr_plus) - (idx - 1) + 1

                                bplus_coeffs = zeros(get_num_basis(Slr_plus))
                                bplus_coeffs[b_plus_idx] = 1.0

                                bplusprime_space = create_derivative_space(Slr_plus) # TODO: Should be the same as Slr_minus. What happens if we use Slr_minus?
                                bplusprime_coeffs = compute_derivative_coefficients(
                                    bplus_coeffs, Slr_plus, bplusprime_space
                                )

                                beta_bplusprime_coeffs = compute_product_coefficients(
                                    beta_tilde_coeffs,
                                    Sgd_lr,
                                    bplusprime_coeffs,
                                    bplusprime_space,
                                    create_product_space(Sgd_lr, bplusprime_space),
                                    product_algorithm,
                                )

                                # b+(v) b1(u)
                                local_offset = get_num_basis(tp_space)
                                bplusb1_idx =
                                    (b_plus_idx - 1) *
                                    get_constituent_num_basis(tp_space)[1] + 1
                                local_dof = bplusb1_idx + local_offset
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)

                                # b+(v) b2(u)
                                bplusb2_idx =
                                    (b_plus_idx - 1) *
                                    get_constituent_num_basis(tp_space)[1] + 2
                                local_dof = bplusb2_idx + local_offset
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)

                                # beta(v) b+'(v) h/p b2(u)
                                local_offset =
                                    get_num_basis(tp_space) + get_num_basis(tp_lr_1plus)
                                val = (h / p) .* beta_bplusprime_coeffs
                                # The number of non-zero values can be
                                # larger than one here, so we add all of them.
                                for local_dof in eachindex(beta_bplusprime_coeffs)
                                    if abs(beta_bplusprime_coeffs[local_dof]) >= 1e-10#beta_bplusprime_coeffs[local_dof] != 0.0
                                        push!(global_extr_rows, global_dof)
                                        push!(
                                            global_extr_cols,
                                            (local_dof - 1) *
                                            get_constituent_num_basis(tp_lr_prod)[1] +
                                            2 +
                                            local_offset +
                                            offset[patch_idx],
                                        )
                                        push!(global_extr_vals, val[local_dof])
                                    end
                                end
                            else
                                # - dofs, only one term (alpha(v) b-(v) h/p b2(u)).
                                if idx == 1
                                    # - dof on the top row.
                                    b_minus_idx = get_num_basis(Slr_minus)
                                elseif idx == 1 + get_num_basis(Slr_plus) + 1
                                    # - dof on the bottom row.
                                    b_minus_idx = 1
                                else
                                    b_minus_idx =
                                        get_num_basis(Slr_minus) -
                                        (idx - idxs_plus_left - 2)
                                end

                                b_min_coeffs = zeros(get_num_basis(Slr_minus))
                                b_min_coeffs[b_minus_idx] = 1.0
                                alpha_bmin_coeffs = compute_product_coefficients(
                                    alpha_tilde_coeffs,
                                    Sgd_lr,
                                    b_min_coeffs,
                                    Slr_minus,
                                    create_product_space(Sgd_lr, Slr_minus),
                                    product_algorithm,
                                )

                                local_offset =
                                    get_num_basis(tp_space) + get_num_basis(tp_lr_1plus)
                                val = (h / p) .* alpha_bmin_coeffs
                                for local_dof in eachindex(alpha_bmin_coeffs)
                                    if abs(alpha_bmin_coeffs[local_dof]) >= 1e-10#alpha_bmin_coeffs[local_dof] != 0.0
                                        push!(global_extr_rows, global_dof)
                                        push!(
                                            global_extr_cols,
                                            (local_dof - 1) *
                                            get_constituent_num_basis(tp_lr_prod)[1] +
                                            2 +
                                            local_offset +
                                            offset[patch_idx],
                                        )
                                        push!(global_extr_vals, val[local_dof])
                                    end
                                end
                            end
                        end
                    end

                    if dof_part_division == 6
                        # Right edge. Note that we start from the bottom
                        # with counting the dofs on this edge.

                        if patch_idx == 2
                            # The edge is not shared, so the local dofs
                            # are directly the tensor product basis functions.
                            if global_to_local_dof_dict[global_dof][patch_idx] <=
                                N_outer_ring
                                # Outer edge dofs.
                                local_dof = (idx + 1) * N1
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                                idxs_plus_right += 1
                            else
                                # Inner edge dofs.
                                local_dof = ((idx - idxs_plus_right) + 2) * N1 - 1
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                            end

                        else
                            # f2 = (b+ bN1) + (b+ bN1-1) + (alpha b- h/p bN1-1) + (beta b+' h/p bN1-1)

                            h = 1.0 / get_constituent_num_elements(tp_space)[1]  # Only true for uniform breakpoints.
                            p = get_constituent_polynomial_degree(tp_space)[1]

                            # Approximate the gluing data.
                            alpha_tilde_coeffs = _project_gluing_data(
                                Sgd_lr, gluing_data[patch_idx][2][1]
                            )
                            beta_tilde_coeffs = _project_gluing_data(
                                Sgd_lr, gluing_data[patch_idx][2][2]
                            )

                            # If the local dof on the current patch
                            # corresponding to the global dof is in the
                            # outer ring, it is a + dof.
                            if global_to_local_dof_dict[global_dof][patch_idx] <=
                               N_outer_ring &&
                                (!(idx == 1 || idx == 1 + get_num_basis(Slr_plus) + 1))
                                # + dofs, three terms.
                                if dof_part_division == 6
                                    idxs_plus_right += 1
                                end

                                # Counting from the bottom.
                                b_plus_idx = idx - 1

                                bplus_coeffs = zeros(get_num_basis(Slr_plus))
                                bplus_coeffs[b_plus_idx] = 1.0

                                bplusprime_space = create_derivative_space(Slr_plus)
                                bplusprime_coeffs = compute_derivative_coefficients(
                                    bplus_coeffs, Slr_plus, bplusprime_space
                                )

                                beta_bplusprime_coeffs = compute_product_coefficients(
                                    beta_tilde_coeffs,
                                    Sgd_lr,
                                    bplusprime_coeffs,
                                    bplusprime_space,
                                    create_product_space(Sgd_lr, bplusprime_space),
                                    product_algorithm,
                                )

                                # b+(v) bN1(u)
                                local_offset = get_num_basis(tp_space)
                                bplusb1_idx =
                                    (b_plus_idx - 1) *
                                    get_constituent_num_basis(tp_lr_1plus)[1] +
                                    get_constituent_num_basis(tp_lr_1plus)[1]
                                local_dof = bplusb1_idx + local_offset
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)

                                # b+(v) bN1-1(u)
                                bplusb2_idx =
                                    (b_plus_idx - 1) *
                                    get_constituent_num_basis(tp_lr_1plus)[1] +
                                    get_constituent_num_basis(tp_lr_1plus)[1] - 1
                                local_dof = bplusb2_idx + local_offset
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)

                                # beta(v) b+'(v) h/p bN1-1(u)
                                local_offset =
                                    get_num_basis(tp_space) + get_num_basis(tp_lr_1plus)
                                val = (h / p) .* beta_bplusprime_coeffs
                                # The number of non-zero values can be
                                # larger than one here, so we add all of them.
                                for local_dof in eachindex(beta_bplusprime_coeffs)
                                    if abs(beta_bplusprime_coeffs[local_dof]) >= 1e-10#beta_bplusprime_coeffs[local_dof] != 0.0
                                        push!(global_extr_rows, global_dof)
                                        push!(
                                            global_extr_cols,
                                            (local_dof - 1) *
                                            get_constituent_num_basis(tp_lr_prod)[1] +
                                            get_constituent_num_basis(tp_lr_prod)[1] - 1 +
                                            local_offset +
                                            offset[patch_idx],
                                        )
                                        push!(global_extr_vals, val[local_dof])
                                    end
                                end
                            else
                                # - dofs, only one term (alpha(v) b-(v) h/p bN1-1(u)).
                                if idx == 1
                                    # - dof bottom row.
                                    b_minus_idx = 1
                                elseif idx == 1 + get_num_basis(Slr_plus) + 1
                                    # - dof top row.
                                    b_minus_idx = get_num_basis(Slr_minus)
                                else
                                    b_minus_idx = idx - idxs_plus_right - 2 + 1
                                end

                                b_min_coeffs = zeros(get_num_basis(Slr_minus))
                                b_min_coeffs[b_minus_idx] = 1.0
                                alpha_bmin_coeffs = compute_product_coefficients(
                                    alpha_tilde_coeffs,
                                    Sgd_lr,
                                    b_min_coeffs,
                                    Slr_minus,
                                    create_product_space(Sgd_lr, Slr_minus),
                                    product_algorithm,
                                )

                                local_offset =
                                    get_num_basis(tp_space) + get_num_basis(tp_lr_1plus)
                                val = (h / p) .* alpha_bmin_coeffs
                                for local_dof in eachindex(alpha_bmin_coeffs)
                                    if abs(alpha_bmin_coeffs[local_dof]) >= 1e-10
                                        push!(global_extr_rows, global_dof)
                                        push!(
                                            global_extr_cols,
                                            (local_dof - 1) *
                                            get_constituent_num_basis(tp_lr_prod)[1] +
                                            get_constituent_num_basis(tp_lr_prod)[1] - 1 +
                                            local_offset +
                                            offset[patch_idx],
                                        )
                                        push!(global_extr_vals, val[local_dof])
                                    end
                                end
                            end
                        end
                    end

                    if dof_part_division == 2
                        # Bottom edge. Note that we start from the left
                        # with counting the dofs on this edge.

                        # In the two-patch case, these are not shared
                        # and have a direct correspondence with the
                        # tensor product basis functions.
                        if patch_idx == 1
                            if global_to_local_dof_dict[global_dof][patch_idx] <=
                                N_outer_ring
                                # Outer edge dofs. 1 to N1-2 in TP space.
                                local_dof = idx + 1
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                                idxs_plus_bottom += 1
                            else
                                # Inner edge dofs. N1 to 2N1-2 in TP space.
                                local_dof = idx - idxs_plus_bottom + 1 + N1
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                            end

                        else
                            # Second patch
                            if global_to_local_dof_dict[global_dof][patch_idx] <=
                                N_outer_ring
                                # Outer edge dofs. 3 to N1 in TP space.
                                local_dof = idx + 2
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                                idxs_plus_bottom += 1
                            else
                                # Inner edge dofs. N1+3 to 2N1 in TP space.
                                local_dof = idx - idxs_plus_bottom + 2 + N1
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                            end
                        end
                    end

                    if dof_part_division == 8
                        # Top edge. Note that we start from the right
                        # with counting the dofs on this edge.

                        # In the two-patch case, these are not shared
                        # and have a direct correspondence with the
                        # tensor product basis functions.
                        if patch_idx == 1
                            if global_to_local_dof_dict[global_dof][patch_idx] <=
                                N_outer_ring
                                # Outer edge dofs. Ntp-N1+2 to Ntp-2 in TP space.
                                local_dof = N1 * N2 - 1 - idx
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                                idxs_plus_top += 1
                            else
                                # Inner edge dofs. Ntp-2N1+3 to Ntp-N1 in TP space.
                                local_dof = N1 * N2 - N1 - 1 - (idx - idxs_plus_top)
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                            end

                        else
                            # Second patch
                            if global_to_local_dof_dict[global_dof][patch_idx] <=
                                N_outer_ring
                                # Outer edge dofs. Ntp-N1+3 to Ntp in TP space.
                                local_dof = N1 * N2 - idx
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                                idxs_plus_top += 1
                            else
                                # Inner edge dofs. Ntp-2N1+4 to Ntp-N1 in TP space.
                                local_dof = N1 * N2 - N1 - 1 - (idx - idxs_plus_top)
                                push!(global_extr_rows, global_dof)
                                push!(global_extr_cols, local_dof + offset[patch_idx])
                                push!(global_extr_vals, 1.0)
                            end
                        end
                    end
                end
            end
        end
    end

    global_extraction_matrix = SparseArrays.sparse(
        global_extr_rows,
        global_extr_cols,
        global_extr_vals,
        max_global_dof,
        local_basis_functions_total,
    )

    # Convert the global extraction matrix to the local (per element)
    # ones that the ExtractionOperator expects.
    extraction_coefficients = Vector{Tuple{Matrix{Float64}}}(undef, num_elements)
    basis_indices = Vector{Indices{1, Vector{Int}, UnitRange{Int}}}(undef, num_elements)
    global_elem_id = 1
    for patch_idx in 1:1:2
        for elem_id in 1:1:elements_per_patch[patch_idx]
            # Obtain the local indices for each constituent space. Also
            # add the offset per space and per patch.
            support_per_space = Vector{Vector{Int}}(undef, 5)
            cumulative_functions_per_space = cumsum([
                get_num_basis(local_space) for
                local_space in patch_spaces[patch_idx][[1, 8, 9, 10, 11]]
            ])
            for (idx, local_space) in enumerate(patch_spaces[patch_idx][[1, 8, 9, 10, 11]])
                support_per_space[idx] = get_support_on_element(local_space, elem_id)

                # Offset per space.
                if idx > 1
                    support_per_space[idx] .+= cumulative_functions_per_space[idx - 1]
                end

                # Offset per patch.
                support_per_space[idx] .+= offset[patch_idx]
            end

            # The local indices are now column indices in the global
            # extraction coefficient matrix. The non-zero row indices
            # are the global indices supported on this element.
            global_columns = reduce(vcat, support_per_space)
            (nz_row_idxs, _, _) = SparseArrays.findnz(
                global_extraction_matrix[:, global_columns]
            )

            # Only the unique indices are needed. unique!(sort!()) is
            # supposed to be more efficient than unique alone.
            nz_row_idxs = unique!(sort!(nz_row_idxs))
            basis_indices[global_elem_id] = Indices(nz_row_idxs, (1:length(nz_row_idxs),))

            # Extract the local extraction coefficients from the global
            # matrix. The transpose is needed to be in line with the
            # convention that [constituent_spaces] * [extraction] = [Approx C1].
            extraction_coefficients[global_elem_id] = (
                Matrix(transpose(global_extraction_matrix[nz_row_idxs, global_columns])),
            )

            global_elem_id += 1
        end
    end

    return ExtractionOperator(
        extraction_coefficients, basis_indices, num_elements, max_global_dof
    )
end

function _get_dirichlet_dofs(
    approx_c1_space::ApproximateC1Space{manifold_dim, 2}
) where {manifold_dim}
    dirichlet_dofs = Int[]

    # This is all done manually for the time being!!!

    # patch 1
    for dof_division in eachindex(approx_c1_space.dof_partition[1])
        if dof_division == 1 ||
            dof_division == 2 ||
            dof_division == 4 ||
            dof_division == 7 ||
            dof_division == 8
            for dof in approx_c1_space.dof_partition[1][dof_division]
                #println("dof: ", dof, " dof_division: ", dof_division, " local dof: ", approx_c1_space.global_to_local_dof_dict[dof][1])
                if approx_c1_space.global_to_local_dof_dict[dof][1] <=
                    approx_c1_space.N_outer_ring_per_patch[1]
                    push!(dirichlet_dofs, dof)
                end
            end
        end
    end

    # patch 2
    for dof_division in eachindex(approx_c1_space.dof_partition[2])
        if dof_division == 2 ||
            dof_division == 3 ||
            dof_division == 6 ||
            dof_division == 8 ||
            dof_division == 9
            for dof in approx_c1_space.dof_partition[2][dof_division]
                if approx_c1_space.global_to_local_dof_dict[dof][2] <=
                    approx_c1_space.N_outer_ring_per_patch[2]
                    push!(dirichlet_dofs, dof)
                end
            end
        end
    end

    # # Shared dofs which are also on the external boundary.
    for (i, dof) in enumerate(approx_c1_space.dof_partition[1][6])
        #println("i: ", i, " dof: ", dof, " local dof: ", approx_c1_space.global_to_local_dof_dict[dof][1], " N_outer_ring: ", approx_c1_space.N_outer_ring_per_patch[1])
        #if approx_c1_space.global_to_local_dof_dict[dof][1] <= approx_c1_space.N_outer_ring_per_patch[1] && (i == 2 || i == get_constituent_num_basis(approx_c1_space.function_spaces[1][4])[2] + 1)
        if approx_c1_space.global_to_local_dof_dict[dof][1] <=
           approx_c1_space.N_outer_ring_per_patch[1] && (
            i <= 2 ||
            i >= get_constituent_num_basis(approx_c1_space.function_spaces[1][4])[2] + 1
        )
            #println("dof: ", dof)
            push!(dirichlet_dofs, dof)
        end
    end

    # for (i, dof) in enumerate(approx_c1_space.dof_partition[1][9])
    #     if approx_c1_space.global_to_local_dof_dict[dof][1] <= approx_c1_space.N_outer_ring_per_patch[1] && i >= 3 && i < 6
    #         #println("dof: ", dof)
    #         push!(dirichlet_dofs, dof)
    #     end
    # end

    # Non-shared dofs in the shared partitions.
    # push!(dirichlet_dofs, approx_c1_space.dof_partition[2][1][3])
    # push!(dirichlet_dofs, approx_c1_space.dof_partition[2][7][1])
    #println("dofs: ", approx_c1_space.dof_partition[2][1][3], " ", approx_c1_space.dof_partition[2][7][1])

    return dirichlet_dofs
end

function _get_dirichlet_and_neumann_dofs(
    approx_c1_space::ApproximateC1Space{manifold_dim, 2}
) where {manifold_dim}
    bc_dofs = Int[]

    # patch 1
    for dof_division in eachindex(approx_c1_space.dof_partition[1])
        if dof_division == 1 ||
            dof_division == 2 ||
            dof_division == 4 ||
            dof_division == 7 ||
            dof_division == 8
            for dof in approx_c1_space.dof_partition[1][dof_division]
                # if approx_c1_space.global_to_local_dof_dict[dof][1] <=
                #     approx_c1_space.N_outer_ring_per_patch[1]
                #     push!(bc_dofs, dof)
                # end
                push!(bc_dofs, dof)
            end
        end

        # if dof_division == 5
        #     all_dofs = approx_c1_space.dof_partition[1][dof_division]
        #     dofs_per_dim = Int(sqrt(length(all_dofs)))
        #     interior_dofs = reshape(all_dofs, (dofs_per_dim, dofs_per_dim))[
        #         2:end, 2:(end - 1)
        #     ]

        #     push!(bc_dofs, (setdiff(all_dofs, interior_dofs))...)
        # end
    end

    # patch 2
    for dof_division in eachindex(approx_c1_space.dof_partition[2])
        if dof_division == 2 ||
            dof_division == 3 ||
            dof_division == 6 ||
            dof_division == 8 ||
            dof_division == 9
            for dof in approx_c1_space.dof_partition[2][dof_division]
                # if approx_c1_space.global_to_local_dof_dict[dof][2] <=
                #     approx_c1_space.N_outer_ring_per_patch[2]
                #     push!(bc_dofs, dof)
                # end
                push!(bc_dofs, dof)
            end
        end

        # if dof_division == 5
        #     all_dofs = approx_c1_space.dof_partition[2][dof_division]
        #     dofs_per_dim = Int(sqrt(length(all_dofs)))
        #     interior_dofs = reshape(all_dofs, (dofs_per_dim, dofs_per_dim))[
        #         1:(end - 1), 2:(end - 1)
        #     ]

        #     push!(bc_dofs, (setdiff(all_dofs, interior_dofs))...)
        # end
    end

    # Shared dofs which are also on the external boundary.
    min_dof = 1e9
    max_dof = 0
    for (i, dof) in enumerate(approx_c1_space.dof_partition[1][6])
        if approx_c1_space.global_to_local_dof_dict[dof][1] <=
           approx_c1_space.N_outer_ring_per_patch[1] && (
            i <= 3 ||
            i >= get_constituent_num_basis(approx_c1_space.function_spaces[1][4])[2] #+ 1
        )
            push!(bc_dofs, dof)
        end
        # if approx_c1_space.global_to_local_dof_dict[dof][1] <=
        #    approx_c1_space.N_outer_ring_per_patch[1] && (
        #     i <= 2 ||
        #     i >= get_constituent_num_basis(approx_c1_space.function_spaces[1][4])[2] + 1
        # )
        #     push!(bc_dofs, dof)
        # end

        if approx_c1_space.global_to_local_dof_dict[dof][1] >
            approx_c1_space.N_outer_ring_per_patch[1]
            min_dof = min(min_dof, dof)
            max_dof = max(max_dof, dof)
        end
    end

    push!(bc_dofs, min_dof)
    push!(bc_dofs, max_dof)

    return bc_dofs
end

# Getters and setters (for internal and external use).
"""
    _get_local_dof_partition(approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}) where {manifold_dim, num_patches}

Get the local dof partitioning for an approximate C1 space.

# Arguments
- `approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}`: The approximate C1 space.

# Returns
- `::Vector{Vector{Vector{Int}}}`: The local dof partition.
"""
function _get_local_dof_partition(
    approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}
) where {manifold_dim, num_patches}
    return approx_c1_space.local_dof_partition
end

"""
    _find_patch(approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}, element_id::Int) where {manifold_dim, num_patches}

Find the patch on which the given element is located.

# Arguments
- `approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}`: The approximate C1 space.
- `element_id::Int`: The element of interest.

# Returns
- `patch_id::Int`: The patch on which the given element is located.

# Throws
- `ArgumentError`: If the given element number is larger than the number of patches in the appropriate C1 space.
"""
function _find_patch(
    approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}, element_id::Int
) where {manifold_dim, num_patches}
    elements_total = 0
    for patch_id in eachindex(approx_c1_space.function_spaces)
        elements_on_patch = get_num_elements(approx_c1_space.function_spaces[patch_id][1])
        elements_total += elements_on_patch
        if element_id <= elements_total
            return patch_id
        end
    end
    throw(
        ArgumentError(
            "The element_id $element_id is too large for the approximate C1 space. It has only $elements_total elements.",
        ),
    )
end

"""
    get_max_local_dim(approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}) where {manifold_dim, num_patches}

Get the maximum local dimension of the approximate C1 space.

# Arguments
- `approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}`: The approximate C1 space.

# Returns
- `max_local_dim::Int`: The maximum local dimension.
"""
function get_max_local_dim(
    approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}
) where {manifold_dim, num_patches}
    max_local_dim = 0
    for patch_idx in 1:1:num_patches
        max_local_dim_patch = 0
        for local_space in approx_c1_space.function_spaces[patch_idx]
            max_local_dim_patch += get_max_local_dim(local_space)
        end
        max_local_dim = max(max_local_dim, max_local_dim_patch)
    end
    return max_local_dim
end

# Specialised evaluate (related) functions.
"""
    get_local_basis(approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_patches}

Compute the local basis functions and their derivatives for an
approximate C1 space on the given element.

# Arguments
- `approx_c1_space::ApproximateC1Space{manifold_dim, num_patches}`: The approximate C1 space.
- `el_id::Int`: Element ID.
- `xi::NTuple{n,Vector{Float64}}`: Tuple of vectors representing evaluation points in each dimension.
- `nderivatives::Int`: Number of derivatives to compute.

# Returns
- `local_basis::Vector{Vector{Matrix{Float64}}}`: The local basis functions and their derivatives.
"""
function get_local_basis(
    approx_c1_space::ApproximateC1Space{manifold_dim, num_patches},
    el_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
    component_id::Int=1,
) where {manifold_dim, num_patches}

    # Find the patch on which the element is located.
    patch_id = _find_patch(approx_c1_space, el_id)

    # Find the local element to which the given `el_id` corresponds.
    local_element_id = el_id
    for patch_i in 1:1:(patch_id - 1)
        local_element_id -= get_num_elements(approx_c1_space.function_spaces[patch_i][1])
    end
    # Get the constituent spaces on this patch and evaluate them. Note
    # that the constituent spaces are all tensor product spaces, so we
    # only have to evaluate them and stack them together.
    tp_eval = evaluate(
        approx_c1_space.function_spaces[patch_id][1], local_element_id, xi, nderivatives
    )[1]
    tp_lr_1plus_eval = evaluate(
        approx_c1_space.function_spaces[patch_id][2], local_element_id, xi, nderivatives
    )[1]
    tp_lr_prod_eval = evaluate(
        approx_c1_space.function_spaces[patch_id][3], local_element_id, xi, nderivatives
    )[1]
    tp_bt_2plus_eval = evaluate(
        approx_c1_space.function_spaces[patch_id][4], local_element_id, xi, nderivatives
    )[1]
    tp_bt_prod_eval = evaluate(
        approx_c1_space.function_spaces[patch_id][5], local_element_id, xi, nderivatives
    )[1]

    # The order of the evaluations is important.
    local_basis = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for i in eachindex(local_basis)
        local_basis[i] = Vector{Vector{Matrix{Float64}}}(undef, size(tp_eval[i]))
        for j in eachindex(local_basis[i])
            local_basis[i][j] = [
                hcat(
                    tp_eval[i][j][1],
                    tp_lr_1plus_eval[i][j][1],
                    tp_lr_prod_eval[i][j][1],
                    tp_bt_2plus_eval[i][j][1],
                    tp_bt_prod_eval[i][j][1],
                ),
            ]
        end
    end

    return local_basis
end

function get_num_elements_per_patch(
    space::ApproximateC1Space{manifold_dim, num_patches}
) where {manifold_dim, num_patches}
    return ntuple(num_patches) do i
        return get_num_elements(space.function_spaces[i][1])
    end
end
