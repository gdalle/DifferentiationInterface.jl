using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
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

test_differentiation(
    AutoZygote(),
    flux_scenarios();
    isequal=DIT.flux_isequal,
    isapprox=DIT.flux_isapprox,
    rtol=1e-2,
    atol=1e-6,
)
