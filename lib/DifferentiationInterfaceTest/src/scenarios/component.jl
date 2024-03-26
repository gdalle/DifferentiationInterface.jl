## Vector to scalar

function componentvector_to_scalar(x::ComponentVector)::Number
    return sum(sin, x.a) + sum(cos, x.b)
end

componentvector_to_scalar_gradient(x) = ComponentVector(; a=cos.(x.a), b=-sin.(x.b))

function componentvector_to_scalar_pushforward(x, dx)
    return dot(componentvector_to_scalar_gradient(x), dx)
end

function componentvector_to_scalar_pullback(x, dy)
    return componentvector_to_scalar_gradient(x) .* dy
end

function componentvector_to_scalar_ref()
    return Reference(;
        pushforward=componentvector_to_scalar_pushforward,
        pullback=componentvector_to_scalar_pullback,
        gradient=componentvector_to_scalar_gradient,
    )
end

## Gather

const SCALING_CVEC = ComponentVector(; a=collect(1:7), b=collect(8:12))

function component_scenarios_allocating()
    return [
        Scenario(
            make_scalar_to_array(SCALING_CVEC); x=2.0, ref=scalar_to_array_ref(SCALING_CVEC)
        ),
        Scenario(
            componentvector_to_scalar;
            x=ComponentVector{Float64}(; a=collect(1:7), b=collect(8:12)),
            ref=componentvector_to_scalar_ref(),
        ),
    ]
end

function component_scenarios_mutating()
    return [
        Scenario(
            make_scalar_to_array!(SCALING_CVEC);
            x=2.0,
            y=float.(SCALING_CVEC),
            ref=scalar_to_array_ref(SCALING_CVEC),
        ),
    ]
end

function component_scenarios()
    return vcat(component_scenarios_allocating(), component_scenarios_mutating())
end
