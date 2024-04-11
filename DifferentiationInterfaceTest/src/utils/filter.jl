function filter_scenarios(
    scenarios::Vector{<:AbstractScenario};
    input_type::Type,
    output_type::Type,
    first_order::Bool,
    second_order::Bool,
    excluded::Vector,
)
    scenarios = filter(s -> (s.x isa input_type && s.y isa output_type), scenarios)
    !first_order &&
        (scenarios = filter(s -> isa(s, AbstractSecondOrderScenario), scenarios))
    !second_order &&
        (scenarios = filter(s -> isa(s, AbstractFirstOrderScenario), scenarios))
    for T in excluded
        scenarios = filter(s -> !isa(s, T), scenarios)
    end
    # sort for nice printing
    scenarios = sort(scenarios; by=s -> (string(typeof(s).name.name), string(s.f)))
    return scenarios
end
