using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using FiniteDifferences: FiniteDifferences
using Flux: Flux
using Random
using Tracker: Tracker
using Zygote: Zygote
using Test

Random.seed!(0)

test_differentiation(
    [
        AutoZygote(),
        AutoTracker(),
        # AutoEnzyme()  # TODO: fix
    ],
    DIT.flux_scenarios();
    isequal=DIT.flux_isequal,
    isapprox=DIT.flux_isapprox,
    rtol=1e-2,
    atol=1e-6,
)
