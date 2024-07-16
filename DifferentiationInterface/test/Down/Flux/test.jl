using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDifferences: FiniteDifferences
using Flux: Flux
using Enzyme: Enzyme
using Zygote: Zygote
using Test

test_differentiation(
    [AutoZygote()],
    flux_scenarios();
    isequal=DIT.flux_isequal,
    isapprox=DIT.flux_isapprox,
    rtol=1e-2,
    atol=1e-6,
)
