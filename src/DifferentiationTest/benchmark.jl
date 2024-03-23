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
    operator::Function,
    scenario::Scenario,
    bench,
)
    bench_min = minimum(bench)
    tup = (;
        backend=backend_string(backend),
        mode=mode(backend),
        operator=operator,
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
