"""
    BenchmarkData

Ad-hoc storage type for differentiation benchmarking results.
You can turn it into a `DataFrame` as follows:

```julia
df = DataFrames.DataFrame(pairs(benchmark_data)...)
```

#  Fields

These are not part of the public API.

$(TYPEDFIELDS)
"""
@kwdef struct BenchmarkData
    backend::Vector{String} = []
    mode::Vector{Type} = []
    operator::Vector{Function} = []
    variant::Vector{Function} = []
    func::Vector{String} = []
    mutating::Vector{Bool} = []
    input_type::Vector{Type} = []
    output_type::Vector{Type} = []
    input_size::Vector = []
    output_size::Vector = []
    samples::Vector{Int} = []
    time::Vector{Float64} = []
    bytes::Vector{Float64} = []
    allocs::Vector{Float64} = []
    compile_fraction::Vector{Float64} = []
    gc_fraction::Vector{Float64} = []
    evals::Vector{Float64} = []
end

function Base.pairs(data::BenchmarkData)
    ns = fieldnames(BenchmarkData)
    return ns .=> getfield.(Ref(data), ns)
end

function record!(data, tup::NamedTuple)
    for n in fieldnames(typeof(tup))
        push!(getfield(data, n), getfield(tup, n))
    end
end

function record!(
    data::BenchmarkData,
    backend::AbstractADType,
    variant::Function,
    scenario::AbstractScenario,
    bench,
)
    operator = operator(scenario)
    bench_min = minimum(bench)
    tup = (;
        backend=backend_string(backend),
        mode=mode(backend),
        operator=operator,
        variant=variant,
        func=string(scenario.f),
        mutating=is_mutating(scenario),
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
    return record!(data, tup)
end

## Pushforward

function run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    scen::PushforwardScenario{false};
    allocations::Bool,
)
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    bench1 = @be mysimilar(dy) value_and_pushforward!!(f, _, ba, x, dx, extras)
    if allocations && dy isa Number
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, value_and_pushforward!!, scen, bench1)
    return nothing
end

function run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    scen::PushforwardScenario{true};
    allocations::Bool,
)
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    bench1 = @be (mysimilar(y), mysimilar(dy)) value_and_pushforward!!(
        f!, _[1], _[2], ba, x, dx, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, value_and_pushforward!!, scen, bench1)
    return nothing
end

## Pullback

function run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    scen::PullbackScenario{false};
    allocations::Bool,
)
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    bench1 = @be mysimilar(dx) value_and_pullback!!(f, _, ba, x, dy, extras)
    if allocations && dy isa Number
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, value_and_pullback!!, scen, bench1)
    return nothing
end

function run_benchmark!(
    data::BenchmarkData, ba::AbstractADType, scen::PullbackScenario{true}; allocations::Bool
)
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    bench1 = @be (mysimilar(y), mysimilar(dx)) value_and_pullback!!(
        f!, _[1], _[2], ba, x, dy, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, value_and_pullback!!, scen, bench1)
    return nothing
end

## Derivative

function run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    scen::DerivativeScenario{false};
    allocations::Bool,
)
    (; f, x, y, dy) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    bench1 = @be mysimilar(dy) value_and_derivative!!(f, _, ba, x, extras)
    # only test allocations if the output is scalar
    if allocations && y isa Number
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, value_and_derivative!!, scen, bench1)
    return nothing
end

function run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    scen::DerivativeScenario{true};
    allocations::Bool,
)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    bench1 = @be (mysimilar(y), mysimilar(dy)) value_and_derivative!!(
        f!, _[1], _[2], ba, x, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, value_and_derivative!!, scen, bench1)
    return nothing
end

## Gradient

function run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    scen::GradientScenario{false};
    allocations::Bool,
)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    bench1 = @be mysimilar(dx) value_and_gradient!!(f, _, ba, x, extras)
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, value_and_gradient!!, scen, bench1)
    return nothing
end

## Jacobian

function run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    scen::JacobianScenario{false};
    allocations::Bool,
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_template = Matrix{eltype(y)}(undef, length(y), length(x))
    bench1 = @be mysimilar(jac_template) value_and_jacobian!!(f, _, ba, x, extras)
    # never test allocations
    record!(data, ba, value_and_jacobian!!, scen, bench1)
    return nothing
end

function run_benchmark!(
    data::BenchmarkData, ba::AbstractADType, scen::JacobianScenario{true}; allocations::Bool
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    jac_template = Matrix{eltype(y)}(undef, length(y), length(x))
    bench1 = @be (mysimilar(y), mysimilar(jac_template)) value_and_jacobian!!(
        f!, _[1], _[2], ba, x, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, value_and_jacobian!!, scen, bench1)
    return nothing
end

## Second derivative

function run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    scen::SecondDerivativeScenario{false};
    allocations::Bool,
)
    (; f, x, y, dy) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)
    bench1 = @be second_derivative(f, ba, x, extras)
    # only test allocations if the output is scalar
    if allocations && y isa Number
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, second_derivative, scen, bench1)
    return nothing
end

## Hessian-vector product

function run_benchmark!(
    data::BenchmarkData, ba::AbstractADType, scen::HVPScenario{false}; allocations::Bool
)
    (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x)
    bench1 = @be hvp(f, ba, x, dx, extras)
    # no test for now
    record!(data, ba, hvp, scen, bench1)
    return nothing
end

## Hessian

function run_benchmark!(
    data::BenchmarkData, ba::AbstractADType, scen::HessianScenario{false}; allocations::Bool
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    bench1 = @be hessian(f, ba, x, extras)
    # no test for now
    record!(data, ba, hessian, scen, bench1)
    return nothing
end
