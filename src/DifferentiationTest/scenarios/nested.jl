recursive_norm(x::Number) = abs2(x)
recursive_norm(x::AbstractArray) = sum(abs2, x)
recursive_norm(x) = sum(recursive_norm, fleaves(x))

function nested(x::Number)
    return (a=[x^2, x^3], b=([sin(x);;],), c=1)
end

nested_derivative(x) = (a=[2x, 3x^2], b=([cos(x);;],), c=0)
nested_pushforward(x, dx) = (a=[2x * dx, 3x^2 * dx], b=([cos(x) * dx;;],), c=0)
nested_pullback(x, dy) = (a=[2x * dy, 3x^2 * dy], b=([cos(x) * dy;;],), c=0)
nested_second_derivative(x) = (a=[2, 6x], b=([-sin(x);;],), c=0)

function nested_ref()
    return Reference(;
        derivative=nested_derivative,
        pushforward=nested_pushforward,
        pullback=nested_pullback,
        second_derivative=nested_second_derivative,
    )
end

function nested_immutables(x::Number)
    return merge(nested_immutables(x), (; d=cos(x)))
end

function nested_immutables_derivative(x)
    return merge(nested_derivative(x), (; d=-sin(x)))
end

function nested_immutables_pushforward(x, dx)
    return merge(nested_pushforward(x, dx), (; d=-sin(x) * dx))
end

function nested_immutables_pullback(x, dy)
    return merge(nested_pullback(x, dy), (; d=-sin(x) * dy))
end

function nested_immutables_second_derivative(x)
    return merge(nested_second_derivative(x), (; d=-cos(x)))
end

function nested_immutables_ref()
    return Reference(;
        derivative=nested_immutables_derivative,
        pushforward=nested_immutables_pushforward,
        pullback=nested_immutables_pullback,
        second_derivative=nested_immutables_second_derivative,
    )
end

function nested_scenarios(; immutables=true)
    scenarios = [
        Scenario(nested, 2.0; ref=nested_ref()),
        # Scenario(recursive_norm; x=nested(2.0)),
    ]
    scenarios_immutables = [
        Scenario(nested_immutables, 2.0; ref=nested_immutables_ref()),
        # Scenario(recursive_norm; x=nested_immutables(2.0)),
    ]
    if immutables
        return vcat(scenarios, scenarios_immutables)
    else
        return scenarios
    end
end
