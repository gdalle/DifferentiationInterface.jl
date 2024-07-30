using Pkg
Pkg.add(["Lux", "LuxTestUtils", "Zygote"])

using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Lux: Lux
using LuxTestUtils: LuxTestUtils
using Random

Random.seed!(0)

test_differentiation(
    AutoZygote(),
    DIT.lux_scenarios();
    isequal=DIT.lux_isequal,
    isapprox=DIT.lux_isapprox,
    rtol=1.0f-2,
    atol=1.0f-2,
    logging=LOGGING,
)
