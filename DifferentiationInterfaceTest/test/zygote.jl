using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using Zygote: Zygote

## Dense

test_differentiation(AutoZygote(); excluded=[:second_derivative], logging=LOGGING)

## Weird

test_differentiation(
    AutoZygote(),
    vcat(component_scenarios(), static_scenarios(), gpu_scenarios());
    correctness=true,
    excluded=[:second_derivative],
    logging=LOGGING,
)
