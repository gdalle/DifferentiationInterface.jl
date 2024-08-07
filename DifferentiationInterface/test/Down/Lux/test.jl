using Pkg
Pkg.add(["FiniteDiff", "Lux", "LuxTestUtils", "Tracker", "Zygote"])

using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDiff: FiniteDiff
using Lux: Lux
using LuxTestUtils: LuxTestUtils
using Random
using Tracker: Tracker
using Zygote: Zygote

LOGGING = get(ENV, "CI", "false") == "false"

Random.seed!(0)

test_differentiation(
    [AutoZygote(), AutoTracker()],
    DIT.lux_scenarios(Random.Xoshiro(63));
    isequal=DIT.lux_isequal,
    isapprox=DIT.lux_isapprox,
    rtol=1.0f-2,
    atol=1.0f-3,
    logging=LOGGING,
)
