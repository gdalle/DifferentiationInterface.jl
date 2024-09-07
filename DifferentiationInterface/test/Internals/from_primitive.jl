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
    @test check_inplace(backend)
end

test_differentiation(fromprimitive_backends, default_scenarios(); logging=LOGGING);
