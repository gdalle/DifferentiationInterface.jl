using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using StochasticAD: StochasticAD
#using SparseConnectivityTracer
#using SparseMatrixColorings
#using ComponentArrays: ComponentArrays
#using StaticArrays: StaticArrays

for backend in [AutoStochasticAD(10)]
    @test check_available(backend)
    @test !check_twoarg(backend)
end

## Dense

test_differentiation(AutoStochasticAD(10); second_order=false, logging=LOGGING)
