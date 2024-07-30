using Pkg
Pkg.add(["Enzyme", "Flux", "Zygote"])

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using Flux: Flux
using Random
using Zygote: Zygote
using Test

Random.seed!(0)

test_differentiation(
    [
        AutoZygote(),
        # AutoEnzyme()  # TODO: fix
    ],
    DIT.flux_scenarios();
    isequal=DIT.flux_isequal,
    isapprox=DIT.flux_isapprox,
    rtol=1e-2,
    atol=1e-6,
)
