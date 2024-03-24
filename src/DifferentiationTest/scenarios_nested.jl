nested_norm(x::Number) = abs2(x)
nested_norm(x::AbstractArray) = sum(nested_norm, x)
nested_norm(x::Tuple) = sum(nested_norm, x)

function nested_scenarios()
    scenarios = [Scenario(nested_norm; x=([1.0, 2.0], [3.0, 4.0, 5.0]))]
    return scenarios
end
