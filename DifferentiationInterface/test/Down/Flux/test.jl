using Pkg
Pkg.add(["FiniteDifferences", "Enzyme", "Flux", "Zygote"])

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using FiniteDifferences: FiniteDifferences
using Flux: Flux
using Random
using Zygote: Zygote
using Test

LOGGING = get(ENV, "CI", "false") == "false"

Random.seed!(0)

test_differentiation(
    [
        AutoZygote(),
        # AutoEnzyme()  # TODO: fix
    ],
    DIT.flux_scenarios();
    isapprox=DIT.flux_isapprox,
    rtol=1e-2,
    atol=1e-6,
    scenario_intact=false,  # TODO: why?
    logging=LOGGING,
)
