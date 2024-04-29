struct CallCounter{F}
    f::F
    count::Base.RefValue{Int}
end

CallCounter(f::F) where {F} = CallCounter{F}(f, Ref(0))

function (cc::CallCounter)(x)
    cc.count[] += 1
    return cc.f(x)
end

function (cc::CallCounter)(y, x)
    cc.count[] += 1
    return cc.f(y, x)
end

function reset_count!(cc::CallCounter)
    count = cc.count[]
    cc.count[] = 0
    return count
end

function failed_bench()
    evals = 0
    time = NaN
    allocs = NaN
    bytes = NaN
    gc_fraction = NaN
    compile_fraction = NaN
    recompile_fraction = NaN
    warmup = NaN
    checksum = NaN
    sample = Sample(
        evals,
        time,
        allocs,
        bytes,
        gc_fraction,
        compile_fraction,
        recompile_fraction,
        warmup,
        checksum,
    )
    return Benchmark([sample])
end

function failed_benchs(k::Integer)
    return ntuple(i -> failed_bench(), k)
end

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
    mode::AbstractMode
    scenario::Symbol
    operator::Symbol
    func::Symbol
    arguments::Int
    input_type::Type
    output_type::Type
    input_size::Tuple
    output_size::Tuple
    calls::Int
    samples::Int
    evals::Int
    time::Float64
    bytes::Float64
    allocs::Float64
    compile_fraction::Float64
    gc_fraction::Float64
end

function record!(
    data::Vector{BenchmarkDataRow},
    backend::AbstractADType,
    scenario::AbstractScenario,
    operator::Symbol,
    bench::Benchmark,
    calls::Integer,
)
    bench_min = minimum(bench)
    row = BenchmarkDataRow(;
        backend=backend_str(backend),
        mode=mode(backend),
        scenario=typeof(scenario).name.name,
        operator=Symbol(operator),
        func=Symbol(scenario.f),
        arguments=nb_args(scenario),
        input_type=typeof(scenario.x),
        output_type=typeof(scenario.y),
        input_size=size(scenario.x),
        output_size=size(scenario.y),
        calls=calls,
        samples=length(bench.samples),
        evals=Int(bench_min.evals),
        time=bench_min.time,
        bytes=bench_min.bytes,
        allocs=bench_min.allocs,
        compile_fraction=bench_min.compile_fraction,
        gc_fraction=bench_min.gc_fraction,
    )
    return push!(data, row)
end

## Pushforward

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::PushforwardScenario{1,:outofplace};
)
    (; f, x, y, dx) = deepcopy(scen)
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pushforward(f, ba, x, dx)
        bench0 = @be prepare_pushforward(f, ba, x, dx) samples = 1 evals = 1
        bench1 = @be value_and_pushforward(f, ba, x, dx, extras)
        bench2 = @be pushforward(f, ba, x, dx, extras)
        # count
        cc = CallCounter(f)
        extras = prepare_pushforward(cc, ba, x, dx)
        calls0 = reset_count!(cc)
        value_and_pushforward(cc, ba, x, dx, extras)
        calls1 = reset_count!(cc)
        pushforward(cc, ba, x, dx, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pushforward, bench0, calls0)
    record!(data, ba, scen, :value_and_pushforward, bench1, calls1)
    record!(data, ba, scen, :pushforward, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::PushforwardScenario{1,:inplace};
)
    (; f, x, y, dx) = deepcopy(scen)
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pushforward(f, ba, x, dx)
        bench0 = @be prepare_pushforward(f, ba, x, dx) samples = 1 evals = 1
        bench1 = @be mysimilar(y) value_and_pushforward!(f, _, ba, x, dx, extras) evals =
            1
        bench2 = @be mysimilar(y) pushforward!(f, _, ba, x, dx, extras) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_pushforward(cc, ba, x, dx)
        calls0 = reset_count!(cc)
        value_and_pushforward!(cc, mysimilar(y), ba, x, dx, extras)
        calls1 = reset_count!(cc)
        pushforward!(cc, mysimilar(y), ba, x, dx, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pushforward, bench0, calls0)
    record!(data, ba, scen, :value_and_pushforward!, bench1, calls1)
    record!(data, ba, scen, :pushforward!, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::PushforwardScenario{2,:outofplace};
)
    (; f, x, y, dx) = deepcopy(scen)
    f! = f
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pushforward(f!, mysimilar(y), ba, x, dx)
        bench0 = @be mysimilar(y) prepare_pushforward(f!, _, ba, x, dx) samples = 1 evals =
            1
        bench1 = @be mysimilar(y) value_and_pushforward(f!, _, ba, x, dx, extras) evals =
            1
        bench2 = @be mysimilar(y) pushforward(f!, _, ba, x, dx, extras) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_pushforward(cc!, mysimilar(y), ba, x, dx)
        calls0 = reset_count!(cc!)
        value_and_pushforward(cc!, mysimilar(y), ba, x, dx, extras)
        calls1 = reset_count!(cc!)
        pushforward(cc!, mysimilar(y), ba, x, dx, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pushforward, bench0, calls0)
    record!(data, ba, scen, :value_and_pushforward, bench1, calls1)
    record!(data, ba, scen, :pushforward, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::PushforwardScenario{2,:inplace};
)
    (; f, x, y, dx) = deepcopy(scen)
    f! = f
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pushforward(f!, mysimilar(y), ba, x, dx)
        bench0 = @be mysimilar(y) prepare_pushforward(f!, _, ba, x, dx) evals = 1 samples =
            1
        bench1 = @be (mysimilar(y), mysimilar(y)) value_and_pushforward!(
            f!, _[1], _[2], ba, x, dx, extras
        ) evals = 1
        bench2 = @be (mysimilar(y), mysimilar(y)) pushforward!(
            f!, _[1], _[2], ba, x, dx, extras
        ) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_pushforward(cc!, mysimilar(y), ba, x, dx)
        calls0 = reset_count!(cc!)
        value_and_pushforward!(cc!, mysimilar(y), mysimilar(y), ba, x, dx, extras)
        calls1 = reset_count!(cc!)
        pushforward!(cc!, mysimilar(y), mysimilar(y), ba, x, dx, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pushforward, bench0, calls0)
    record!(data, ba, scen, :value_and_pushforward!, bench1, calls1)
    record!(data, ba, scen, :pushforward!, bench2, calls2)
    return nothing
end

## Pullback

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::PullbackScenario{1,:outofplace};
)
    (; f, x, y, dy) = deepcopy(scen)
    (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4) = try
        # benchmark
        extras = prepare_pullback(f, ba, x, dy)
        bench0 = @be prepare_pullback(f, ba, x, dy) samples = 1 evals = 1
        bench1 = @be value_and_pullback(f, ba, x, dy, extras)
        bench2 = @be pullback(f, ba, x, dy, extras)
        bench3 = @be value_and_pullback_split(f, ba, x, extras)
        bench4 = @be last(value_and_pullback_split(f, ba, x, extras)) _(dy)
        # count
        cc = CallCounter(f)
        extras = prepare_pullback(cc, ba, x, dy)
        calls0 = reset_count!(cc)
        value_and_pullback(cc, ba, x, dy, extras)
        calls1 = reset_count!(cc)
        pullback(cc, ba, x, dy, extras)
        calls2 = reset_count!(cc)
        _, pullbackfunc = value_and_pullback_split(cc, ba, x, extras)
        calls3 = reset_count!(cc)
        pullbackfunc(dy)
        calls4 = reset_count!(cc)
        (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4)
    catch e
        bench0, bench1, bench2, bench3, bench4 = failed_benchs(5)
        calls0, calls1, calls2, calls3, calls4 = -1, -1, -1, -1, -1
        (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4)
    end
    # record
    record!(data, ba, scen, :prepare_pullback, bench0, calls0)
    record!(data, ba, scen, :value_and_pullback, bench1, calls1)
    record!(data, ba, scen, :pullback, bench2, calls2)
    record!(data, ba, scen, :value_and_pullback_split, bench3, calls3)
    record!(data, ba, scen, :pullbackfunc, bench4, calls4)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::PullbackScenario{1,:inplace};
)
    (; f, x, y, dy) = deepcopy(scen)
    (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4) = try
        # benchmark
        extras = prepare_pullback(f, ba, x, dy)
        bench0 = @be prepare_pullback(f, ba, x, dy) samples = 1 evals = 1
        bench1 = @be mysimilar(x) value_and_pullback!(f, _, ba, x, dy, extras) evals = 1
        bench2 = @be mysimilar(x) pullback!(f, _, ba, x, dy, extras) evals = 1
        bench3 = @be value_and_pullback!_split(f, ba, x, extras)
        bench4 = @be (mysimilar(x), last(value_and_pullback!_split(f, ba, x, extras))) _[2](
            _[1], dy
        ) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_pullback(cc, ba, x, dy)
        calls0 = reset_count!(cc)
        value_and_pullback!(cc, mysimilar(x), ba, x, dy, extras)
        calls1 = reset_count!(cc)
        pullback!(cc, mysimilar(x), ba, x, dy, extras)
        calls2 = reset_count!(cc)
        _, pullbackfunc! = value_and_pullback!_split(cc, ba, x, extras)
        calls3 = reset_count!(cc)
        pullbackfunc!(mysimilar(x), dy)
        calls4 = reset_count!(cc)
        (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4)
    catch e
        bench0, bench1, bench2, bench3, bench4 = failed_benchs(5)
        calls0, calls1, calls2, calls3, calls4 = -1, -1, -1, -1, -1
        (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4)
    end
    # record
    record!(data, ba, scen, :prepare_pullback, bench0, calls0)
    record!(data, ba, scen, :value_and_pullback!, bench1, calls1)
    record!(data, ba, scen, :pullback!, bench2, calls2)
    record!(data, ba, scen, :value_and_pullback!_split, bench3, calls3)
    record!(data, ba, scen, :pullbackfunc!, bench4, calls4)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::PullbackScenario{2,:outofplace},
)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4) = try
        # benchmark
        extras = prepare_pullback(f!, mysimilar(y), ba, x, dy)
        bench0 = @be mysimilar(y) prepare_pullback(f!, _, ba, x, dy) samples = 1 evals =
            1
        bench1 = @be mysimilar(y) value_and_pullback(f!, _, ba, x, dy, extras) evals = 1
        bench2 = @be mysimilar(y) pullback(f!, _, ba, x, dy, extras) evals = 1
        bench3 = @be value_and_pullback_split(f!, y, ba, x, extras)
        bench4 = @be last(value_and_pullback_split(f!, y, ba, x, extras)) _(y, dy) evals =
            1
        # count
        cc! = CallCounter(f!)
        extras = prepare_pullback(cc!, mysimilar(y), ba, x, dy)
        calls0 = reset_count!(cc!)
        value_and_pullback(cc!, mysimilar(y), ba, x, dy, extras)
        calls1 = reset_count!(cc!)
        pullback(cc!, mysimilar(y), ba, x, dy, extras)
        calls2 = reset_count!(cc!)
        _, pullbackfunc = value_and_pullback_split(cc!, y, ba, x, extras)
        calls3 = reset_count!(cc!)
        pullbackfunc(y, dy)
        calls4 = reset_count!(cc!)
        (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4)
    catch e
        bench0, bench1, bench2, bench3, bench4 = failed_benchs(5)
        calls0, calls1, calls2, calls3, calls4 = -1, -1, -1, -1, -1
        (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4)
    end
    # record
    record!(data, ba, scen, :prepare_pullback, bench0, calls0)
    record!(data, ba, scen, :value_and_pullback, bench1, calls1)
    record!(data, ba, scen, :pullback, bench2, calls2)
    record!(data, ba, scen, :value_and_pullback_split, bench3, calls3)
    record!(data, ba, scen, :pullbackfunc, bench4, calls4)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::PullbackScenario{2,:inplace}
)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4) = try
        # benchmark
        extras = prepare_pullback(f!, mysimilar(y), ba, x, dy)
        bench0 = @be mysimilar(y) prepare_pullback(f!, _, ba, x, dy) samples = 1 evals =
            1
        bench1 = @be (mysimilar(y), mysimilar(x)) value_and_pullback!(
            f!, _[1], _[2], ba, x, dy, extras
        ) evals = 1
        bench2 = @be (mysimilar(y), mysimilar(x)) pullback!(
            f!, _[1], _[2], ba, x, dy, extras
        ) evals = 1
        bench3 = @be value_and_pullback!_split(f!, y, ba, x, extras)
        bench4 = @be (mysimilar(x), last(value_and_pullback!_split(f!, y, ba, x, extras))) _[2](y, _[1], dy) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_pullback(cc!, mysimilar(y), ba, x, dy)
        calls0 = reset_count!(cc!)
        value_and_pullback!(cc!, mysimilar(y), mysimilar(x), ba, x, dy, extras)
        calls1 = reset_count!(cc!)
        pullback!(cc!, mysimilar(y), mysimilar(x), ba, x, dy, extras)
        calls2 = reset_count!(cc!)
        _, pullbackfunc! = value_and_pullback!_split(cc!, y, ba, x, extras)
        calls3 = reset_count!(cc!)
        pullbackfunc!(y, mysimilar(x), dy)
        calls4 = reset_count!(cc!)
        (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4)
    catch e
        bench0, bench1, bench2, bench3, bench4 = failed_benchs(5)
        calls0, calls1, calls2, calls3, calls4 = -1, -1, -1, -1, -1
        (; bench0, bench1, bench2, bench3, bench4, calls0, calls1, calls2, calls3, calls4)
    end
    # record
    record!(data, ba, scen, :prepare_pullback, bench0, calls0)
    record!(data, ba, scen, :value_and_pullback!, bench1, calls1)
    record!(data, ba, scen, :pullback!, bench2, calls2)
    record!(data, ba, scen, :value_and_pullback!_split, bench3, calls3)
    record!(data, ba, scen, :pullbackfunc!, bench4, calls4)
    return nothing
end

## Derivative

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::DerivativeScenario{1,:outofplace};
)
    (; f, x, y) = deepcopy(scen)
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_derivative(f, ba, x)
        bench0 = @be prepare_derivative(f, ba, x) samples = 1 evals = 1
        bench1 = @be value_and_derivative(f, ba, x, extras)
        bench2 = @be derivative(f, ba, x, extras)
        # count
        cc = CallCounter(f)
        extras = prepare_derivative(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_derivative(cc, ba, x, extras)
        calls1 = reset_count!(cc)
        derivative(cc, ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_derivative, bench0, calls0)
    record!(data, ba, scen, :value_and_derivative, bench1, calls1)
    record!(data, ba, scen, :derivative, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::DerivativeScenario{1,:inplace};
)
    (; f, x, y) = deepcopy(scen)
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_derivative(f, ba, x)
        bench0 = @be prepare_derivative(f, ba, x) samples = 1 evals = 1
        bench1 = @be mysimilar(y) value_and_derivative!(f, _, ba, x, extras) evals = 1
        bench2 = @be mysimilar(y) derivative!(f, _, ba, x, extras) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_derivative(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_derivative!(cc, mysimilar(y), ba, x, extras)
        calls1 = reset_count!(cc)
        derivative!(cc, mysimilar(y), ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_derivative, bench0, calls0)
    record!(data, ba, scen, :value_and_derivative!, bench1, calls1)
    record!(data, ba, scen, :derivative!, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::DerivativeScenario{2,:outofplace};
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_derivative(f!, mysimilar(y), ba, x)
        bench0 = @be mysimilar(y) prepare_derivative(f!, _, ba, x) samples = 1 evals = 1
        bench1 = @be mysimilar(y) value_and_derivative(f!, _, ba, x, extras) evals = 1
        bench2 = @be mysimilar(y) derivative(f!, _, ba, x, extras) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_derivative(cc!, mysimilar(y), ba, x)
        calls0 = reset_count!(cc!)
        value_and_derivative(cc!, mysimilar(y), ba, x, extras)
        calls1 = reset_count!(cc!)
        derivative(cc!, mysimilar(y), ba, x, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_derivative, bench0, calls0)
    record!(data, ba, scen, :value_and_derivative, bench1, calls1)
    record!(data, ba, scen, :derivative, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::DerivativeScenario{2,:inplace};
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_derivative(f!, mysimilar(y), ba, x)
        bench0 = @be mysimilar(y) prepare_derivative(f!, _, ba, x) samples = 1 evals = 1
        bench1 = @be (mysimilar(y), mysimilar(y)) value_and_derivative!(
            f!, _[1], _[2], ba, x, extras
        ) evals = 1
        bench2 = @be (mysimilar(y), mysimilar(y)) derivative!(f!, _[1], _[2], ba, x, extras) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_derivative(cc!, mysimilar(y), ba, x)
        calls0 = reset_count!(cc!)
        value_and_derivative!(cc!, mysimilar(y), mysimilar(y), ba, x, extras)
        calls1 = reset_count!(cc!)
        derivative!(cc!, mysimilar(y), mysimilar(y), ba, x, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_derivative, bench0, calls0)
    record!(data, ba, scen, :value_and_derivative!, bench1, calls1)
    record!(data, ba, scen, :derivative!, bench2, calls2)
    return nothing
end

## Gradient

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::GradientScenario{1,:outofplace};
)
    (; f, x) = deepcopy(scen)
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_gradient(f, ba, x)
        bench0 = @be prepare_gradient(f, ba, x) samples = 1 evals = 1
        bench1 = @be value_and_gradient(f, ba, x, extras)
        bench2 = @be gradient(f, ba, x, extras)
        # count
        cc = CallCounter(f)
        extras = prepare_gradient(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_gradient(cc, ba, x, extras)
        calls1 = reset_count!(cc)
        gradient(cc, ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_gradient, bench0, calls0)
    record!(data, ba, scen, :value_and_gradient, bench1, calls1)
    record!(data, ba, scen, :gradient, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::GradientScenario{1,:inplace};
)
    (; f, x) = deepcopy(scen)
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_gradient(f, ba, x)
        bench0 = @be prepare_gradient(f, ba, x) samples = 1 evals = 1
        bench1 = @be mysimilar(x) value_and_gradient!(f, _, ba, x, extras) evals = 1
        bench2 = @be mysimilar(x) gradient!(f, _, ba, x, extras) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_gradient(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_gradient!(cc, mysimilar(x), ba, x, extras)
        calls1 = reset_count!(cc)
        gradient!(cc, mysimilar(x), ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_gradient, bench0, calls0)
    record!(data, ba, scen, :value_and_gradient!, bench1, calls1)
    record!(data, ba, scen, :gradient!, bench2, calls2)
    return nothing
end

## Jacobian

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::JacobianScenario{1,:outofplace};
)
    (; f, x, y) = deepcopy(scen)
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_jacobian(f, ba, x)
        bench0 = @be prepare_jacobian(f, ba, x) samples = 1 evals = 1
        bench1 = @be value_and_jacobian(f, ba, x, extras)
        bench2 = @be jacobian(f, ba, x, extras)
        # count
        cc = CallCounter(f)
        extras = prepare_jacobian(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_jacobian(cc, ba, x, extras)
        calls1 = reset_count!(cc)
        jacobian(cc, ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_jacobian, bench0, calls0)
    record!(data, ba, scen, :value_and_jacobian, bench1, calls1)
    record!(data, ba, scen, :jacobian, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::JacobianScenario{1,:inplace};
)
    (; f, x, y) = deepcopy(scen)
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        jac_template = mysimilar(jacobian(f, ba, x))
        # benchmark
        extras = prepare_jacobian(f, ba, x)
        bench0 = @be prepare_jacobian(f, ba, x) samples = 1 evals = 1
        bench1 = @be mysimilar(jac_template) value_and_jacobian!(f, _, ba, x, extras) evals =
            1
        bench2 = @be mysimilar(jac_template) jacobian!(f, _, ba, x, extras) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_jacobian(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_jacobian!(cc, mysimilar(jac_template), ba, x, extras)
        calls1 = reset_count!(cc)
        jacobian!(cc, mysimilar(jac_template), ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_jacobian, bench0, calls0)
    record!(data, ba, scen, :value_and_jacobian!, bench1, calls1)
    record!(data, ba, scen, :jacobian!, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::JacobianScenario{2,:outofplace},
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_jacobian(f!, mysimilar(y), ba, x)
        bench0 = @be mysimilar(y) prepare_jacobian(f!, _, ba, x) samples = 1 evals = 1
        bench1 = @be mysimilar(y) value_and_jacobian(f!, _, ba, x, extras) evals = 1
        bench2 = @be mysimilar(y) jacobian(f!, _, ba, x, extras) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_jacobian(cc!, mysimilar(y), ba, x)
        calls0 = reset_count!(cc!)
        value_and_jacobian(cc!, mysimilar(y), ba, x, extras)
        calls1 = reset_count!(cc!)
        jacobian(cc!, mysimilar(y), ba, x, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_jacobian, bench0, calls0)
    record!(data, ba, scen, :value_and_jacobian, bench1, calls1)
    record!(data, ba, scen, :jacobian, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::JacobianScenario{2,:inplace}
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        jac_template = mysimilar(jacobian(f!, mysimilar(y), ba, x))
        # benchmark
        extras = prepare_jacobian(f!, mysimilar(y), ba, x)
        bench0 = @be mysimilar(y) prepare_jacobian(f!, _, ba, x) samples = 1 evals = 1
        bench1 = @be (mysimilar(y), mysimilar(jac_template)) value_and_jacobian!(
            f!, _[1], _[2], ba, x, extras
        ) evals = 1
        bench2 = @be (mysimilar(y), mysimilar(jac_template)) jacobian!(
            f!, _[1], _[2], ba, x, extras
        ) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_jacobian(cc!, y, ba, x)
        calls0 = reset_count!(cc!)
        value_and_jacobian!(cc!, mysimilar(y), mysimilar(jac_template), ba, x, extras)
        calls1 = reset_count!(cc!)
        jacobian!(cc!, mysimilar(y), mysimilar(jac_template), ba, x, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_jacobian, bench0, calls0)
    record!(data, ba, scen, :value_and_jacobian!, bench1, calls1)
    record!(data, ba, scen, :jacobian!, bench2, calls2)
    return nothing
end

## Second derivative

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::SecondDerivativeScenario{1,:outofplace};
)
    (; f, x, y) = deepcopy(scen)
    (; bench0, bench1, calls0, calls1) = try
        # benchmark
        extras = prepare_second_derivative(f, ba, x)
        bench0 = @be prepare_second_derivative(f, ba, x) samples = 1 evals = 1
        bench1 = @be second_derivative(f, ba, x, extras)
        # count
        cc = CallCounter(f)
        extras = prepare_second_derivative(cc, ba, x)
        calls0 = reset_count!(cc)
        second_derivative(cc, ba, x, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_second_derivative, bench0, calls0)
    record!(data, ba, scen, :second_derivative, bench1, calls1)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow},
    ba::AbstractADType,
    scen::SecondDerivativeScenario{1,:inplace};
)
    (; f, x, y) = deepcopy(scen)
    (; bench0, bench1, calls0, calls1) = try
        # benchmark
        extras = prepare_second_derivative(f, ba, x)
        bench0 = @be prepare_second_derivative(f, ba, x) samples = 1 evals = 1
        bench1 = @be mysimilar(y) second_derivative!(f, _, ba, x, extras) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_second_derivative(cc, ba, x)
        calls0 = reset_count!(cc)
        second_derivative!(cc, mysimilar(y), ba, x, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_second_derivative, bench0, calls0)
    record!(data, ba, scen, :second_derivative!, bench1, calls1)
    return nothing
end

## Hessian-vector product

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::HVPScenario{1,:outofplace}
)
    (; f, x, y, dx) = deepcopy(scen)
    (; bench0, bench1, calls0, calls1) = try
        # benchmark
        extras = prepare_hvp(f, ba, x, dx)
        bench0 = @be prepare_hvp(f, ba, x, dx) samples = 1 evals = 1
        bench1 = @be hvp(f, ba, x, dx, extras)
        # count
        cc = CallCounter(f)
        extras = prepare_hvp(cc, ba, x, dx)
        calls0 = reset_count!(cc)
        hvp(cc, ba, x, dx, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_hvp, bench0, calls0)
    record!(data, ba, scen, :hvp, bench1, calls1)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::HVPScenario{1,:inplace}
)
    (; f, x, y, dx) = deepcopy(scen)
    (; bench0, bench1, calls0, calls1) = try
        # benchmark
        extras = prepare_hvp(f, ba, x, dx)
        bench0 = @be prepare_hvp(f, ba, x, dx) samples = 1 evals = 1
        bench1 = @be mysimilar(x) hvp!(f, _, ba, x, dx, extras) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_hvp(cc, ba, x, dx)
        calls0 = reset_count!(cc)
        hvp!(cc, mysimilar(x), ba, x, dx, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_hvp, bench0, calls0)
    record!(data, ba, scen, :hvp!, bench1, calls1)
    return nothing
end

## Hessian

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::HessianScenario{1,:outofplace}
)
    (; f, x, y) = deepcopy(scen)
    (; bench0, bench1, calls0, calls1) = try
        # benchmark
        extras = prepare_hessian(f, ba, x)
        bench0 = @be prepare_hessian(f, ba, x) samples = 1 evals = 1
        bench1 = @be hessian(f, ba, x, extras)
        # count
        cc = CallCounter(f)
        extras = prepare_hessian(cc, ba, x)
        calls0 = reset_count!(cc)
        hessian(cc, ba, x, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_hessian, bench0, calls0)
    record!(data, ba, scen, :hessian, bench1, calls1)
    return nothing
end

function run_benchmark!(
    data::Vector{BenchmarkDataRow}, ba::AbstractADType, scen::HessianScenario{1,:inplace}
)
    (; f, x, y) = deepcopy(scen)
    (; bench0, bench1, calls0, calls1) = try
        hess_template = Matrix{typeof(y)}(undef, length(x), length(x))
        # benchmark
        extras = prepare_hessian(f, ba, x)
        bench0 = @be prepare_hessian(f, ba, x) samples = 1 evals = 1
        bench1 = @be mysimilar(hess_template) hessian!(f, _, ba, x, extras) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_hessian(cc, ba, x)
        calls0 = reset_count!(cc)
        hessian!(cc, mysimilar(hess_template), ba, x, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_hessian, bench0, calls0)
    record!(data, ba, scen, :hessian!, bench1, calls1)
    return nothing
end
