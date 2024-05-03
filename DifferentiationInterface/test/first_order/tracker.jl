using DifferentiationInterface, DifferentiationInterfaceTest
using Tracker: Tracker

backends = [AutoTracker()]

for backend in backends
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test !check_hessian(backend; verbose=false)
end

test_differentiation(backends; second_order=false, logging=LOGGING);
