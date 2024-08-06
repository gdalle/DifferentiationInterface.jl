using Pkg
Pkg.add("ReverseDiff")

using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterface: AutoReverseFromPrimitive
using ReverseDiff: ReverseDiff
using StaticArrays: StaticArrays
using Test

LOGGING = get(ENV, "CI", "false") == "false"

dense_backends = [AutoReverseDiff(; compile=false), AutoReverseDiff(; compile=true)]

fromprimitive_backends = [AutoReverseFromPrimitive(AutoReverseDiff())]

for backend in vcat(dense_backends, fromprimitive_backends)
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(vcat(dense_backends, fromprimitive_backends); logging=LOGGING);

test_differentiation(AutoReverseDiff(), static_scenarios(); logging=LOGGING);
