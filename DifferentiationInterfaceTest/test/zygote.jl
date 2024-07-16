using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using JLArrays: JLArrays
using Zygote: Zygote

## Dense

test_differentiation(AutoZygote(); excluded=[:second_derivative], logging=LOGGING)

## Weird

if VERSION >= v"1.10"
    test_differentiation(
        AutoZygote(), gpu_scenarios(); correctness=true, second_order=false, logging=LOGGING
    )
end

flux_scenarios()
