"""
    BenchmarkDataRow

Ad-hoc storage type for differentiation benchmarking results.
If you have a vector `rows::Vector{BenchmarkDataRow}`, you can turn it into a `DataFrame` as follows:

```julia
df = DataFrames.DataFrame(rows)
```

#  Fields

These are not part of the public API.

$(TYPEDFIELDS)
"""
@kwdef struct BenchmarkDataRow
    backend::String
    mode::Type
    scenario::Symbol
    operator::Symbol
    func::Symbol
    mutating::Bool
    input_type::Type
    output_type::Type
    input_size::Tuple
    output_size::Tuple
    samples::Int
    time::Float64
    bytes::Float64
    allocs::Float64
    compile_fraction::Float64
    gc_fraction::Float64
    evals::Float64
end

function record!(
    data::Vector{BenchmarkDataRow},
    backend::AbstractADType,
    operator::Function,
    scenario::AbstractScenario,
    bench,
)
    bench_min = minimum(bench)
    row = BenchmarkDataRow(;
        backend=backend_string(backend),
        mode=mode(backend),
        scenario=typeof(scenario).name.name,
        operator=Symbol(operator),
        func=Symbol(scenario.f),
        mutating=ismutating(scenario),
        input_type=typeof(scenario.x),
        output_type=typeof(scenario.y),
        input_size=size(scenario.x),
        output_size=size(scenario.y),
        samples=length(bench.samples),
        time=bench_min.time,
        bytes=bench_min.bytes,
        allocs=bench_min.allocs,
        compile_fraction=bench_min.compile_fraction,
        gc_fraction=bench_min.gc_fraction,
        evals=bench_min.evals,
    )
    return push!(data, row)
end

## Pushforward

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::PushforwardScenario{false};
)
    (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    bench1 = @be mysimilar(y) value_and_pushforward!!(f, _, ba, x, dx, extras)
    bench2 = @be mysimilar(y) pushforward!!(f, _, ba, x, dx, extras)
    record!(data, ba, value_and_pushforward!!, scen, bench1)
    record!(data, ba, pushforward!!, scen, bench2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::PushforwardScenario{true};
)
    (; f, x, y, dx) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    bench1 = @be (mysimilar(y), mysimilar(y)) value_and_pushforward!!(
        f!, _[1], _[2], ba, x, dx, extras
    )
    record!(data, ba, value_and_pushforward!!, scen, bench1)
    return nothing
end

## Pullback

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::PullbackScenario{false};
)
    (; f, x, y, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    bench1 = @be mysimilar(x) value_and_pullback!!(f, _, ba, x, dy, extras)
    bench2 = @be mysimilar(x) pullback!!(f, _, ba, x, dy, extras)
    record!(data, ba, value_and_pullback!!, scen, bench1)
    record!(data, ba, pullback!!, scen, bench2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::PullbackScenario{true}
)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    bench1 = @be (mysimilar(y), mysimilar(x)) value_and_pullback!!(
        f!, _[1], _[2], ba, x, dy, extras
    )
    record!(data, ba, value_and_pullback!!, scen, bench1)
    return nothing
end

## Derivative

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::DerivativeScenario{false};
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    bench1 = @be mysimilar(y) value_and_derivative!!(f, _, ba, x, extras)
    bench2 = @be mysimilar(y) derivative!!(f, _, ba, x, extras)
    record!(data, ba, value_and_derivative!!, scen, bench1)
    record!(data, ba, derivative!!, scen, bench2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::DerivativeScenario{true};
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    bench1 = @be (mysimilar(y), mysimilar(y)) value_and_derivative!!(
        f!, _[1], _[2], ba, x, extras
    )
    record!(data, ba, value_and_derivative!!, scen, bench1)
    return nothing
end

## Gradient

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::GradientScenario{false};
)
    (; f, x) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    bench1 = @be mysimilar(x) value_and_gradient!!(f, _, ba, x, extras)
    bench2 = @be mysimilar(x) gradient!!(f, _, ba, x, extras)
    record!(data, ba, value_and_gradient!!, scen, bench1)
    record!(data, ba, gradient!!, scen, bench2)
    return nothing
end

## Jacobian

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::JacobianScenario{false};
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_template = Matrix{eltype(y)}(undef, length(y), length(x))
    bench1 = @be mysimilar(jac_template) value_and_jacobian!!(f, _, ba, x, extras)
    bench2 = @be mysimilar(jac_template) jacobian!!(f, _, ba, x, extras)
    record!(data, ba, value_and_jacobian!!, scen, bench1)
    record!(data, ba, jacobian!!, scen, bench2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::JacobianScenario{true}
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    jac_template = Matrix{eltype(y)}(undef, length(y), length(x))
    bench1 = @be (mysimilar(y), mysimilar(jac_template)) value_and_jacobian!!(
        f!, _[1], _[2], ba, x, extras
    )
    record!(data, ba, value_and_jacobian!!, scen, bench1)
    return nothing
end

## Second derivative

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::SecondDerivativeScenario{false};
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)
    bench1 = @be mysimilar(y) second_derivative!!(f, _, ba, x, extras)
    record!(data, ba, second_derivative, scen, bench1)
    return nothing
end

## Hessian-vector product

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::HVPScenario{false}
)
    (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x)
    bench1 = @be mysimilar(x) hvp!!(f, _, ba, x, dx, extras)
    record!(data, ba, hvp, scen, bench1)
    return nothing
end

## Hessian

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::HessianScenario{false}
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    hess_template = Matrix{typeof(y)}(undef, length(x), length(x))
    bench1 = @be similar(hess_template) hessian!!(f, _, ba, x, extras)
    record!(data, ba, hessian, scen, bench1)
    return nothing
end
