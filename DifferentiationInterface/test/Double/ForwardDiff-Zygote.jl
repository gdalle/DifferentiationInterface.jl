using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer
using SparseMatrixColorings
using Zygote: Zygote
using Test

dense_backends = [SecondOrder(AutoForwardDiff(), AutoZygote())]

sparse_backends = [
    AutoSparse(
        SecondOrder(AutoForwardDiff(), AutoZygote());
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    ),
]

for backend in vcat(dense_backends, sparse_backends)
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test check_hessian(backend)
end

## Dense backends

test_differentiation(dense_backends; first_order=false, logging=LOGGING);

## Sparse backends

test_differentiation(
    sparse_backends, sparse_scenarios(); first_order=false, sparsity=true, logging=LOGGING
);

## Translation

function zygote_breaking_scenarios()
    onearg_scens = filter(default_scenarios()) do scen
        DIT.nb_args(scen) == 1
    end
    bad_onearg_scens = map(onearg_scens) do scen
        function bad_f(x)
            a = Vector{eltype(x)}(undef, 1)
            a[1] = sum(x)
            return scen.f(x)
        end
        wrapped_bad_f = DifferentiateWith(bad_f, AutoForwardDiff())
        bad_scen = DIT.change_function(scen, wrapped_bad_f)
        return bad_scen
    end
    return bad_onearg_scens
end

test_differentiation(
    AutoZygote(), zygote_breaking_scenarios(); second_order=false, logging=LOGGING
)
