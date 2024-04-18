## Vector to scalar

function comp_to_num(x::ComponentVector)::Number
    return sum(sin, x.a) + sum(cos, x.b)
end

comp_to_num_gradient(x) = ComponentVector(; a=cos.(x.a), b=-sin.(x.b))

function comp_to_num_pushforward(x, dx)
    g = comp_to_num_gradient(x)
    return dot(g.a, dx.a) + dot(g.b, dx.b)
end

function comp_to_num_pullback(x, dy)
    return comp_to_num_gradient(x) .* dy
end

function comp_to_num_scenarios_onearg(x::ComponentVector)
    # pushforward stays out of place
    scens = AbstractScenario[]
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PullbackScenario(comp_to_num; x=x, ref=comp_to_num_pullback, operator=op),
                GradientScenario(comp_to_num; x=x, ref=comp_to_num_gradient, operator=op),
            ],
        )
    end
    for op in (:outofplace,)
        append!(
            scens,
            [
                PushforwardScenario(
                    comp_to_num; x=x, ref=comp_to_num_pushforward, operator=op
                ),
            ],
        )
    end
    return scens
end

## Gather

const CVEC = ComponentVector(; a=collect(1:4), b=collect(5:6))

"""
    component_scenarios()

Create a vector of [`AbstractScenario`](@ref)s with component array types from [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl).
"""
function component_scenarios()
    x = ComponentVector(; a=randn(4), b=randn(2))
    return vcat(
        # one argument
        num_to_arr_scenarios_onearg(randn(), CVEC),
        comp_to_num_scenarios_onearg(x::ComponentVector),
        # two arguments
        num_to_arr_scenarios_twoarg(randn(), CVEC),
    )
end
