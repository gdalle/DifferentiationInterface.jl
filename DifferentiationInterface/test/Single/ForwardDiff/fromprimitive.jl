using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterface: AutoForwardFromPrimitive, AutoReverseFromPrimitive
using DifferentiationInterfaceTest: add_batchified!
using ForwardDiff: ForwardDiff
using Test

fromprimitive_backends = [ #
    AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
    AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
]

for backend in vcat(fromprimitive_backends)
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

## Dense backends

test_differentiation(
    fromprimitive_backends, add_batchified!(default_scenarios()); logging=LOGGING
);

test_differentiation(
    fromprimitive_backends[1],
    add_batchified!(default_scenarios());
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);
