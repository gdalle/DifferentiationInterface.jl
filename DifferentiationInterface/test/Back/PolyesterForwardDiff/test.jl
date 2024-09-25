using Pkg
Pkg.add("PolyesterForwardDiff")

using DifferentiationInterface, DifferentiationInterfaceTest
using PolyesterForwardDiff: PolyesterForwardDiff
using Test

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoPolyesterForwardDiff(; chunksize=1)]
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    AutoPolyesterForwardDiff(; chunksize=1),
    default_scenarios(; include_constantified=true);
    logging=LOGGING,
);
