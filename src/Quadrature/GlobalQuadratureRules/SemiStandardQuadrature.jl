struct SemiStandardQuadrature{manifold_dim, Q, Q2, Q3} <: AbstractGlobalQuadratureRule{manifold_dim}
    canonical_qrule::Q
    semi_canonical_qrule_1::Q2
    semi_element_ids_1::Set{Int}
    semi_canonical_qrule_2::Q3
    semi_element_ids_2::Set{Int}
    num_elements::Int
    function SemiStandardQuadrature(
        canonical_qrule::Q, semi_canonical_qrule_1::Q2, semi_element_ids_1::Set{Int}, semi_canonical_qrule_2::Q3, semi_element_ids_2::Set{Int}, num_elements::Int
    ) where {
        manifold_dim,
        Q <: AbstractElementQuadratureRule{manifold_dim},
        Q2 <: AbstractElementQuadratureRule{manifold_dim},
        Q3 <: AbstractElementQuadratureRule{manifold_dim},
    }
        return new{manifold_dim, Q, Q2, Q3}(
            canonical_qrule, semi_canonical_qrule_1, semi_element_ids_1, semi_canonical_qrule_2, semi_element_ids_2, num_elements
        )
    end
end

get_canonical_quadrature_rule(rule::SemiStandardQuadrature) = rule.canonical_qrule
get_semi_canonical_quadrature_rule_1(rule::SemiStandardQuadrature) = rule.semi_canonical_qrule_1
get_semi_canonical_quadrature_rule_2(rule::SemiStandardQuadrature) = rule.semi_canonical_qrule_2
get_num_evaluation_elements(rule::SemiStandardQuadrature) = rule.num_elements
get_num_base_elements(rule::SemiStandardQuadrature) = rule.num_elements
function get_element_idxs(::SemiStandardQuadrature, element_idx::Int)
    return [element_idx]
end

function get_element_quadrature_rule(rule::SemiStandardQuadrature, element_id::Int)
    if element_id in rule.semi_element_ids_1
        return get_semi_canonical_quadrature_rule_1(rule)
    elseif element_id in rule.semi_element_ids_2
        return get_semi_canonical_quadrature_rule_2(rule)
    else
        return get_canonical_quadrature_rule(rule)
    end
end
