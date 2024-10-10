using Pkg
Pkg.add("Tracker")

using DifferentiationInterface, DifferentiationInterfaceTest
using Tracker: Tracker
using Test

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoTracker()]
    @test check_available(backend)
    @test !check_inplace(backend)
end

test_differentiation(
    AutoTracker(),
    default_scenarios(; include_constantified=true);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
