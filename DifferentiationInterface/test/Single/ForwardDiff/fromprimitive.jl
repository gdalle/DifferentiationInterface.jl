using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterface: AutoForwardFromPrimitive
using ForwardDiff: ForwardDiff
using Test

fromprimitive_backends = [AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5))]

for backend in vcat(fromprimitive_backends)
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

## Dense backends

test_differentiation(fromprimitive_backends; logging=LOGGING);

test_differentiation(
    fromprimitive_backends;
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);
