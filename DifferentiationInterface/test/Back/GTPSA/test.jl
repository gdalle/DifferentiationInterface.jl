using Pkg
Pkg.add("GTPSA")

using DifferentiationInterface, DifferentiationInterfaceTest
using GTPSA: GTPSA
using Test

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoGTPSA()]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(
  AutoGTPSA();
  type_stability=true,
  logging=LOGGING);