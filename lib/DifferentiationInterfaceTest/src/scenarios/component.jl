## Vector to scalar

function comp_to_num(x::ComponentVector)::Number
    return sum(sin, x.a) + sum(cos, x.b)
end

comp_to_num_gradient(x) = ComponentVector(; a=cos.(x.a), b=-sin.(x.b))

function comp_to_num_pushforward(x, dx)
    return dot(comp_to_num_gradient(x), dx)
end

function comp_to_num_pullback(x, dy)
    return comp_to_num_gradient(x) .* dy
end

function comp_to_num_scenarios_allocating(x::ComponentVector)
    return [
        PushforwardScenario(arr_to_num; x=x, ref=comp_to_num_pushforward),
        PullbackScenario(arr_to_num; x=x, ref=comp_to_num_pullback),
        GradientScenario(arr_to_num; x=x, ref=comp_to_num_gradient),
    ]
end

## Gather

const CVEC = ComponentVector(; a=collect(1:4), b=collect(5:6))

"""
    component_scenarios()

Create a vector of [`AbstractScenario`](@ref)s with component array types from [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl).
"""
function component_scenarios()
    return vcat(
        # allocating
        num_to_arr_scenarios_allocating(randn(), CVEC),
        arr_to_num_scenarios_allocating(ComponentVector(; a=randn(4), b=randn(2))),
        # mutating
        num_to_arr_scenarios_mutating(randn(), CVEC),
    )
end
