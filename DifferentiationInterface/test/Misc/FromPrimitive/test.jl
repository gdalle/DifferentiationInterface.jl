using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterface: AutoForwardFromPrimitive, AutoReverseFromPrimitive
using DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using Test

LOGGING = get(ENV, "CI", "false") == "false"

fromprimitive_backends = [ #
    AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
    AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
]

fromprimitive_secondorder_backends = [ #
    SecondOrder(
        AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
        AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
    ),
    SecondOrder(
        AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
        AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
    ),
]

for backend in vcat(fromprimitive_backends)
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    fromprimitive_backends, default_scenarios(; include_constantified=true); logging=LOGGING
);

test_differentiation(
    fromprimitive_secondorder_backends,
    default_scenarios(; include_constantified=true);
    first_order=false,
    logging=LOGGING,
);
