module DifferentiationInterfaceChairmarksDataFramesExt

using Chairmarks: Benchmark
using DataFrames: DataFrame, Not, select!
import DifferentiationInterface.DifferentiationTest as DT

function parse_benchmark_results_aux(bench::Benchmark, level; aggregators=[minimum])
    data = DataFrame()
    for agg in aggregators
        agg_bench = agg(bench)
        data[!, Symbol("time_$agg")] = [agg_bench.time]
        data[!, Symbol("bytes_$agg")] = [agg_bench.bytes]
        data[!, Symbol("allocs_$agg")] = [agg_bench.allocs]
        data[!, Symbol("compile_fraction_$agg")] = [agg_bench.compile_fraction]
        data[!, Symbol("gc_fraction_$agg")] = [agg_bench.gc_fraction]
    end
    data[!, :samples] = [length(bench.samples)]
    return data
end

function parse_benchmark_results_aux(results::Dict, level)
    data = DataFrame()
    level_symbol = Symbol(string("level_$level"))
    for (key, val) in pairs(results)
        subdata = parse_benchmark_results_aux(val, level + 1)
        subdata[!, level_symbol] = fill(key, size(subdata, 1))
        append!(data, subdata)
    end
    select!(data, level_symbol, Not(level_symbol))
    return data
end

function DT.parse_benchmark(results::Dict)
    return parse_benchmark_results_aux(results, 1)
end

end
