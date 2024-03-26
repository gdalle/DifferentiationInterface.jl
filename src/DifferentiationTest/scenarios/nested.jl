recursive_norm(x::Number) = abs2(x)
recursive_norm(x::AbstractArray) = sum(abs2, x)
recursive_norm(x) = sum(recursive_norm, fleaves(x))

recursive_norm_gradient(x) = fmap(_x -> 2_x, x)
recursive_norm_hvp(x) = fmap(_x -> 2 * one(x), x)
recursive_norm_pushforward(x, dx) = fmap(dot, recursive_norm_gradient(x), dx)
recursive_norm_pullback(x, dy) = fmap(_x -> dy * _x, recursive_norm_gradient(x))

function recursive_norm_ref()
    return Reference(;
        pushforward=recursive_norm_pushforward,
        pullback=recursive_norm_pullback,
        gradient=recursive_norm_gradient,
        hvp=recursive_norm_hvp,
    )
end

function nested(x::Number)
    return (a=[x^2, x^3], b=([sin(x);;],), c=1.0)
end

nested_derivative(x) = (a=[2x, 3x^2], b=([cos(x);;],), c=0.0)
nested_pushforward(x, dx) = (a=[2x * dx, 3x^2 * dx], b=([cos(x) * dx;;],), c=0.0)
nested_pullback(x, dy) = (a=[2x * dy, 3x^2 * dy], b=([cos(x) * dy;;],), c=0.0)
nested_second_derivative(x) = (a=[2, 6x], b=([-sin(x);;],), c=0.0)

function nested_ref()
    return Reference(;
        pushforward=nested_pushforward,
        pullback=nested_pullback,
        derivative=nested_derivative,
        second_derivative=nested_second_derivative,
    )
end

function nested_immutables(x::Number)
    return merge(nested(x), (; d=cos(x)))
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
        Scenario(nested; x=2.0, ref=nested_ref()),
        Scenario(recursive_norm; x=nested(2.0), ref=recursive_norm_ref()),
    ]
    scenarios_immutables = [
        Scenario(nested_immutables; x=2.0, ref=nested_immutables_ref()),
        Scenario(recursive_norm; x=nested_immutables(2.0), ref=recursive_norm_ref()),
    ]
    if immutables
        return vcat(scenarios, scenarios_immutables)
    else
        return scenarios
    end
end
