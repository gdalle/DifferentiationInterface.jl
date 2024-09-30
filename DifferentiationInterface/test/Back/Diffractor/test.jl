using Pkg
Pkg.add("Diffractor")

using DifferentiationInterface, DifferentiationInterfaceTest
using Diffractor: Diffractor
using Test

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoDiffractor()]
    @test check_available(backend)
    @test !check_inplace(backend)
end

test_differentiation(
    AutoDiffractor(), default_scenarios(; linalg=false); second_order=false, logging=LOGGING
);
