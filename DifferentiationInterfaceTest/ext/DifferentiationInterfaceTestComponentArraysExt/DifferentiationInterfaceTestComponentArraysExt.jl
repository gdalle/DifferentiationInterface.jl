module DifferentiationInterfaceTestComponentArraysExt

using ComponentArrays: ComponentVector
using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Random: AbstractRNG, default_rng

## Vector to scalar

function comp_to_num(x::ComponentVector)::Number
    return sum(sin.(x.a)) + sum(cos.(x.b))
end

comp_to_num_gradient(x) = ComponentVector(; a=cos.(x.a), b=-sin.(x.b))

function comp_to_num_pushforward(x, dx)
    g = comp_to_num_gradient(x)
    return dot(g, dx)
end

function comp_to_num_pullback(x, dy)
    return comp_to_num_gradient(x) .* dy
end

function comp_to_num_scenarios_onearg(x::ComponentVector; dx::AbstractVector, dy::Number)
    nb_args = 1
    f = comp_to_num
    y = f(x)
    dy_from_dx = comp_to_num_pushforward(x, dx)
    dx_from_dy = comp_to_num_pullback(x, dy)
    grad = comp_to_num_gradient(x)

    # pushforward stays out of place
    scens = Scenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PullbackScenario(f; x, y, dy, dx=dx_from_dy, nb_args, place),
                GradientScenario(f; x, y, grad, nb_args, place),
            ],
        )
    end
    for place in (:outofplace,)
        append!(scens, [PushforwardScenario(f; x, y, dx, dy=dy_from_dx, nb_args, place)])
    end
    return scens
end

## Gather

function DIT.component_scenarios(rng::AbstractRNG=default_rng())
    dy_ = rand(rng)

    x_comp = ComponentVector(; a=randn(rng, 4), b=randn(rng, 2))
    dx_comp = ComponentVector(; a=randn(rng, 4), b=randn(rng, 2))

    scens = vcat(
        # one argument
        comp_to_num_scenarios_onearg(x_comp::ComponentVector; dx=dx_comp, dy=dy_),
        # two arguments
    )
    return scens
end

end
