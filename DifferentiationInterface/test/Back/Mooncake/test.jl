using Pkg
Pkg.add("Mooncake")

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake
using Test

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoMooncake(; config=nothing)]
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    AutoMooncake(; config=nothing),
    default_scenarios(; include_constantified=false);  # toggle to true for multi-argument
    second_order=false,
    logging=LOGGING,
);
