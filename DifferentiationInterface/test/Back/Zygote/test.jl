using Pkg
Pkg.add(["ForwardDiff", "Zygote"])

using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using JLArrays: JLArrays
using StaticArrays: StaticArrays
using Test
using Zygote: Zygote

LOGGING = get(ENV, "CI", "false") == "false"

backends = [AutoZygote()]
second_order_backends = [SecondOrder(AutoForwardDiff(), AutoZygote())]

for backend in vcat(backends, second_order_backends)
    @test check_available(backend)
    @test !check_inplace(backend)
end

## Dense

test_differentiation(
    backends,
    default_scenarios(; include_constantified=true);
    excluded=[:second_derivative],
    logging=LOGGING,
);

test_differentiation(second_order_backends; logging=LOGGING);

test_differentiation(
    backends[1],
    vcat(component_scenarios(), gpu_scenarios());
    excluded=SECOND_ORDER,
    logging=LOGGING,
)

## Sparse

test_differentiation(
    MyAutoSparse.(vcat(backends, second_order_backends)),
    sparse_scenarios(; band_sizes=0:-1);
    sparsity=true,
    logging=LOGGING,
)
