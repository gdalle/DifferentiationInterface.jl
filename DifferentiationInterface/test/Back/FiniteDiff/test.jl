using Pkg
Pkg.add("FiniteDiff")

using DifferentiationInterface, DifferentiationInterfaceTest
using FiniteDiff: FiniteDiff
using Test

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoFiniteDiff()]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(AutoFiniteDiff(); excluded=[:second_derivative, :hvp], logging=LOGGING);
