using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterface: AutoForwardFromPrimitive, AutoReverseFromPrimitive
using ForwardDiff: ForwardDiff
using Test

LOGGING = get(ENV, "CI", "false") == "false"

fromprimitive_backends = [ #
    AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
    AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
]

for backend in vcat(fromprimitive_backends)
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
    @test DifferentiationInterface.pick_batchsize(backend, 100) == 5
end

## Dense backends

test_differentiation(fromprimitive_backends, default_scenarios(); logging=LOGGING);

test_differentiation(
    fromprimitive_backends[1],
    default_scenarios();
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);
