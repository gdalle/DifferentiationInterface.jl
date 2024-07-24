using ADTypes
using ComponentArrays: ComponentArrays
using DifferentiationInterface
using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using Flux: Flux
using ForwardDiff: ForwardDiff
using JLArrays: JLArrays
using Random
using SparseConnectivityTracer
using SparseMatrixColorings
using StaticArrays: StaticArrays
using Zygote: Zygote

test_differentiation(
    AutoForwardDiff(),
    vcat(component_scenarios(), static_scenarios());
    correctness=true,
    logging=LOGGING,
)

test_differentiation(
    AutoFiniteDiff(),
    DIT.make_closure.(default_scenarios());
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    AutoZygote(), gpu_scenarios(); correctness=true, second_order=false, logging=LOGGING
)

Random.seed!(0)

test_differentiation(
    AutoZygote(),
    DIT.flux_scenarios();
    isequal=DIT.flux_isequal,
    isapprox=DIT.flux_isapprox,
    rtol=1e-2,
    atol=1e-6,
    logging=LOGGING,
)
