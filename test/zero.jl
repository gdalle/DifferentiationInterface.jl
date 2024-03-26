include("test_imports.jl")

using DifferentiationInterface.DifferentiationTest:
    AutoZeroForward, AutoZeroReverse, change_ref

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())

function test_differentiation_selfref(
    backend::ADTypes.AbstractADType,
    operators::Vector{Function},
    scenarios::Vector{<:Scenario};
    kwargs...,
)
    new_ref_scenarios = change_ref.(scenarios, Ref(backend))
    return test_differentiation(backend, operators, new_ref_scenarios; kwargs...)
end

## Correctness (vs oneself) + type-stability

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation_selfref(
        backend, all_operators(), default_scenarios(); type_stability=true
    )
end

for backend in [
    SecondOrder(AutoZeroForward(), AutoZeroReverse()),
    SecondOrder(AutoZeroReverse(), AutoZeroForward()),
]
    test_differentiation_selfref(
        backend,
        all_operators(),
        default_scenarios();
        first_order=false,
        type_stability=true,
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
    test_differentiation_selfref(
        backend, all_operators(), weird_array_scenarios(; gpu=true);
    )
    # copyto!(col, col) fails on static arrays
    test_differentiation_selfref(
        backend, all_operators(), weird_array_scenarios(; static=true); excluded=[jacobian]
    )
    # stack fails on component vectors
    test_differentiation_selfref(
        backend,
        all_operators(),
        weird_array_scenarios(; component=true);
        excluded=[hessian],
    )
end
