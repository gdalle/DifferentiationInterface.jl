module DifferentiationInterfaceComponentArraysExt

using ComponentArrays: ComponentVector
using DifferentiationInterface.DifferentiationTest: Scenario, SCALING_VEC

const SCALING_CVEC = ComponentVector(; a=collect(1:7), b=collect(8:12))

function scalar_to_componentvector(x::Number)::ComponentVector
    return sin.(SCALING_CVEC .* x) # output size 12
end

function componentvector_to_scalar(x::ComponentVector)::Number
    return sum(sin, x.a) + sum(cos, x.b)
end

function component_scenarios_allocating()
    return [
        Scenario(scalar_to_componentvector, 2.0),
        Scenario(
            componentvector_to_scalar,
            ComponentVector{Float64}(; a=collect(1:7), b=collect(8:12)),
        ),
    ]
end

end
