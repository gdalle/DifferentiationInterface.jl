module DifferentiationInterfaceTestLuxExt

using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDifferences: FiniteDifferences
using Lux
using LuxTestUtils
using LuxTestUtils: check_approx
using Random: AbstractRNG, default_rng

function DIT.lux_scenarios(rng::AbstractRNG=default_rng())
    scens = Scenario[]
    return scens
end

end
