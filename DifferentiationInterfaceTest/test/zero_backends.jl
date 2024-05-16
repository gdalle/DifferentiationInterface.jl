using DifferentiationInterface
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: AutoZeroForward, AutoZeroReverse

using DataFrames: DataFrames

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())
@test check_twoarg(AutoZeroForward())
@test check_twoarg(AutoZeroReverse())

## Correctness (vs oneself)

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(
        backend,
        default_scenarios();
        correctness=true,
        ref_backend=backend,
        logging=LOGGING == "false",
    )
end

for backend in [
    SecondOrder(AutoZeroForward(), AutoZeroReverse()),
    SecondOrder(AutoZeroReverse(), AutoZeroForward()),
]
    test_differentiation(
        backend,
        default_scenarios();
        correctness=true,
        first_order=false,
        ref_backend=backend,
        logging=LOGGING == "false",
    )
end

## Type stability

if VERSION >= v"1.10"
    test_differentiation(
        [AutoZeroForward(), AutoZeroReverse()];
        correctness=false,
        type_stability=true,
        logging=LOGGING == "false",
    )

    test_differentiation(
        [
            SecondOrder(AutoZeroForward(), AutoZeroReverse()),
            SecondOrder(AutoZeroReverse(), AutoZeroForward()),
        ];
        correctness=false,
        type_stability=true,
        first_order=false,
        logging=LOGGING == "false",
    )
end

## Benchmark

data1 = benchmark_differentiation(
    [AutoZeroForward(), AutoZeroReverse()], default_scenarios(); logging=LOGGING == "false"
);

data2 = benchmark_differentiation(
    [SecondOrder(AutoZeroForward(), AutoZeroReverse())],
    default_scenarios();
    first_order=false,
    logging=LOGGING == "false",
);

df1 = DataFrames.DataFrame(data1)
df2 = DataFrames.DataFrame(data2)
df = vcat(df1, df2)

for col in eachcol(vcat(df1, df2))
    if eltype(col) <: AbstractFloat
        @test !any(isnan, col)
    end
end

struct FakeBackend <: ADTypes.AbstractADType end
ADTypes.mode(::FakeBackend) = ADTypes.ForwardMode()

data3 = benchmark_differentiation([FakeBackend()], default_scenarios(); logging=false);

df3 = DataFrames.DataFrame(data3)

for col in eachcol(df3)
    if eltype(col) <: AbstractFloat
        @test all(isnan, col)
    end
end

## Weird arrays

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(
        backend,
        gpu_scenarios();
        correctness=true,
        ref_backend=backend,
        logging=LOGGING == "false",
    )
    test_differentiation(
        backend,
        static_scenarios();
        correctness=true,
        ref_backend=backend,
        logging=LOGGING == "false",
    )
    # stack fails on component vectors
    test_differentiation(
        backend,
        component_scenarios();
        correctness=true,
        excluded=[HessianScenario],
        ref_backend=backend,
        logging=LOGGING == "false",
    )
end
