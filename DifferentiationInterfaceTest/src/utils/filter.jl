function filter_scenarios(
    scenarios::Vector{<:Scenario};
    input_type::Type,
    output_type::Type,
    first_order::Bool,
    second_order::Bool,
    onearg::Bool,
    twoarg::Bool,
    inplace::Bool,
    outofplace::Bool,
    excluded::Vector{Symbol},
)
    scenarios = filter(s -> (s.x isa input_type && s.y isa output_type), scenarios)
    !first_order && (scenarios = filter(s -> order(s) != 1, scenarios))
    !second_order && (scenarios = filter(s -> order(s) != 2, scenarios))
    !onearg && (scenarios = filter(s -> nb_args(s) != 1, scenarios))
    !twoarg && (scenarios = filter(s -> nb_args(s) != 2, scenarios))
    !inplace && (scenarios = filter(s -> place(s) != :inplace, scenarios))
    !outofplace && (scenarios = filter(s -> place(s) != :outofplace, scenarios))
    scenarios = filter(s -> !(operator(s) in excluded), scenarios)
    # sort for nice printing
    scenarios = sort(scenarios; by=s -> (operator(s), string(s.f)))
    return scenarios
end
