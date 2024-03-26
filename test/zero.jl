include("test_imports.jl")

using DifferentiationInterfaceTest: AutoZeroForward, AutoZeroReverse

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())

## Correctness (vs oneself) + type-stability

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(
        backend,
        all_operators(),
        default_scenarios();
        correctness=backend,
        type_stability=true,
    )
end

for backend in [
    SecondOrder(AutoZeroForward(), AutoZeroReverse()),
    SecondOrder(AutoZeroReverse(), AutoZeroForward()),
]
    test_differentiation(
        backend,
        all_operators(),
        default_scenarios();
        correctness=backend,
        type_stability=true,
        first_order=false,
    )
end

## Call count

test_differentiation(
    AutoZeroForward(); correctness=false, call_count=true, excluded=[gradient]
);

test_differentiation(
    AutoZeroReverse(); correctness=false, call_count=true, excluded=[derivative]
);

## Allocations

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    allocations=true,
    excluded=[jacobian],
);

data = test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()]; correctness=false, benchmark=true
);

df = DataFrames.DataFrame(pairs(data)...)

## Weird arrays

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(backend, all_operators(), gpu_scenarios(); correctness=backend)
    # copyto!(col, col) fails on static arrays
    test_differentiation(
        backend,
        all_operators(),
        static_scenarios();
        correctness=backend,
        excluded=[jacobian],
    )
    # stack fails on component vectors
    test_differentiation(
        backend,
        all_operators(),
        component_scenarios();
        correctness=backend,
        excluded=[hessian],
    )
end
