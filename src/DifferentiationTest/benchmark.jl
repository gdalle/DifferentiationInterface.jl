run_benchmark(args...; kwargs...) = error("Please load Chairmarks.jl")

"""
    BenchmarkData

#  Fields

$(TYPEDFIELDS)
"""
@kwdef struct BenchmarkData
    backend::Vector = []
    mode::Vector = []
    operator::Vector = []
    func::Vector = []
    mutating::Vector = []
    input_type::Vector = []
    output_type::Vector = []
    input_size::Vector = []
    output_size::Vector = []
    samples::Vector = []
    time::Vector = []
    bytes::Vector = []
    allocs::Vector = []
    compile_fraction::Vector = []
    gc_fraction::Vector = []
    evals::Vector = []
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
    scenario::Scenario,
    operator::Symbol,
    bench,
)
    bench_min = minimum(bench)
    tup = (;
        backend=backend_string(backend),
        mode=mode(backend),
        operator=operator,
        func=string(scenario.f),
        mutating=scenario.mutating,
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
