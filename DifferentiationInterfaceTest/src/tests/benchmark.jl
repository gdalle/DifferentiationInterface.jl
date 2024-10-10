struct CallCounter{F}
    f::F
    count::Base.RefValue{Int}
end

CallCounter(f::F) where {F} = CallCounter{F}(f, Ref(0))

function (cc::CallCounter)(x, args...)
    cc.count[] += 1
    return cc.f(x, args...)
end

function (cc::CallCounter)(y, x, args...)
    cc.count[] += 1
    return cc.f(y, x, args...)
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

failed_benchs(k::Integer) = ntuple(i -> failed_bench(), k)
failed_calls(k::Integer) = ntuple(i -> -1, k)

"""
    DifferentiationBenchmarkDataRow

Ad-hoc storage type for differentiation benchmarking results.

#  Fields

$(TYPEDFIELDS)

See the documentation of [Chairmarks.jl](https://github.com/LilithHafner/Chairmarks.jl) for more details on the measurement fields.
"""
Base.@kwdef struct DifferentiationBenchmarkDataRow
    "backend used for benchmarking"
    backend::AbstractADType
    "scenario used for benchmarking"
    scenario::Scenario
    "differentiation operator used for benchmarking, e.g. `:gradient` or `:hessian`"
    operator::Symbol
    "whether the operator had been prepared"
    prepared::Union{Nothing,Bool}
    "number of calls to the differentiated function for one call to the operator"
    calls::Int
    "number of benchmarking samples taken"
    samples::Int
    "number of evaluations used for averaging in each sample"
    evals::Int
    "minimum runtime over all samples, in seconds"
    time::Float64
    "minimum number of allocations over all samples"
    allocs::Float64
    "minimum memory allocated over all samples, in bytes"
    bytes::Float64
    "minimum fraction of time spent in garbage collection over all samples, between 0.0 and 1.0"
    gc_fraction::Float64
    "minimum fraction of time spent compiling over all samples, between 0.0 and 1.0"
    compile_fraction::Float64
end

function record!(
    data::Vector{DifferentiationBenchmarkDataRow};
    backend::AbstractADType,
    scenario::Scenario,
    operator::String,
    prepared::Union{Nothing,Bool},
    bench::Benchmark,
    calls::Integer,
)
    bench_min = minimum(bench)
    row = DifferentiationBenchmarkDataRow(;
        backend=backend,
        scenario=scenario,
        operator=Symbol(operator),
        prepared=prepared,
        calls=calls,
        samples=length(bench.samples),
        evals=Int(bench_min.evals),
        time=bench_min.time,
        allocs=bench_min.allocs,
        bytes=bench_min.bytes,
        gc_fraction=bench_min.gc_fraction,
        compile_fraction=bench_min.compile_fraction,
    )
    return push!(data, row)
end
