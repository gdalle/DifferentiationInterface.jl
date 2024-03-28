using DifferentiationInterfaceTest: AutoZeroForward, AutoZeroReverse

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())

## Correctness (vs oneself)

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(backend, all_operators(), default_scenarios(); correctness=backend)
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
        first_order=false,
    )
end

## Type stability

test_differentiation(
    AutoZeroForward(); correctness=false, type_stability=true, excluded=[pullback]
)
test_differentiation(
    AutoZeroReverse(); correctness=false, type_stability=true, excluded=[pushforward]
)
test_differentiation(
    [
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
    ];
    correctness=false,
    type_stability=true,
    first_order=false,
)

## Call count

test_differentiation(
    AutoZeroForward(); correctness=false, call_count=true, excluded=[gradient, pullback]
);

test_differentiation(
    AutoZeroReverse();
    correctness=false,
    call_count=true,
    excluded=[derivative, pushforward],
);

## Benchmark

data = benchmark_differentiation([AutoZeroForward(), AutoZeroReverse()]);

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
        excluded=[jacobian, hessian],
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
