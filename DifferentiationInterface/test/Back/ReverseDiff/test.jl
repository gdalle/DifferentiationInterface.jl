using Pkg
Pkg.add("ReverseDiff")

using DifferentiationInterface, DifferentiationInterfaceTest
using ReverseDiff: ReverseDiff
using StaticArrays: StaticArrays
using Test

LOGGING = get(ENV, "CI", "false") == "false"

dense_backends = [AutoReverseDiff(; compile=false), AutoReverseDiff(; compile=true)]

for backend in dense_backends
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    dense_backends, default_scenarios(; include_constantified=true); logging=LOGGING
);

test_differentiation(AutoReverseDiff(), static_scenarios(); logging=LOGGING);
