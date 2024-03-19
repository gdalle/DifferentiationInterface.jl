module DifferentiationInterfaceChairmarksDataFramesExt

using Chairmarks: Benchmark
using DataFrames: DataFrame, Not, select!
using DifferentiationInterface
import DifferentiationInterface.DifferentiationTest as DT

NAMES =
    Base.get_extension(
        DifferentiationInterface, :DifferentiationInterfaceChairmarksExt
    ).NAMES

function parse_benchmark_results_aux(bench::Benchmark, level; names, aggregators)
    data = DataFrame()
    data[!, :samples] = [length(bench.samples)]
    for agg in aggregators
        agg_bench = agg(bench)
        data[!, Symbol("time_$agg")] = [agg_bench.time]
        data[!, Symbol("bytes_$agg")] = [agg_bench.bytes]
        data[!, Symbol("allocs_$agg")] = [agg_bench.allocs]
        data[!, Symbol("compile_fraction_$agg")] = [agg_bench.compile_fraction]
        data[!, Symbol("gc_fraction_$agg")] = [agg_bench.gc_fraction]
    end
    return data
end

function parse_benchmark_results_aux(results, level; names, aggregators)
    level_symbol = Symbol(get(names, level, "level_$level"))
    data = DataFrame()
    for k in keys(results)
        v = results[k]
        subdata = parse_benchmark_results_aux(v, level + 1; names, aggregators)
        subdata[!, level_symbol] = fill(k, size(subdata, 1))
        append!(data, subdata; promote=true)
    end
    select!(data, level_symbol, Not(level_symbol))
    return data
end

function DT.parse_benchmark(results; names::Vector{Symbol}=NAMES, aggregators=[minimum])
    return parse_benchmark_results_aux(results, 1; names, aggregators)
end

end
