using DifferentiationInterfaceTest: AutoZeroForward, AutoZeroReverse

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())

## Correctness (vs oneself)

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(backend, default_scenarios(); correctness=backend, logging=true)
end

for backend in [
    SecondOrder(AutoZeroForward(), AutoZeroReverse()),
    SecondOrder(AutoZeroReverse(), AutoZeroForward()),
]
    test_differentiation(
        backend, default_scenarios(); correctness=backend, first_order=false, logging=true
    )
end

## Type stability

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    type_stability=true,
    logging=true,
)

test_differentiation(
    [
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
    ];
    correctness=false,
    type_stability=true,
    first_order=false,
    logging=true,
)

## Call count

test_differentiation(
    AutoZeroForward();
    correctness=false,
    call_count=true,
    logging=true,
    excluded=[GradientScenario],
);

test_differentiation(
    AutoZeroReverse();
    correctness=false,
    call_count=true,
    logging=true,
    excluded=[DerivativeScenario],
);

## Benchmark

data = benchmark_differentiation([AutoZeroForward(), AutoZeroReverse()]; logging=true);

df = DataFrames.DataFrame(data)

## Weird arrays

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(backend, gpu_scenarios(); correctness=backend, logging=true)
    # copyto!(col, col) fails on static arrays
    test_differentiation(
        backend,
        static_scenarios();
        correctness=backend,
        excluded=[JacobianScenario, HessianScenario],
        logging=true,
    )
    # stack fails on component vectors
    test_differentiation(
        backend,
        component_scenarios();
        correctness=backend,
        excluded=[HessianScenario],
        logging=true,
    )
end
