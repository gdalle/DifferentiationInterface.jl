module DifferentiationInterfaceChairmarksDataFramesExt

using Chairmarks: Benchmark
using DataFrames: DataFrame, Not, select!
using DifferentiationInterface
import DifferentiationInterface.DifferentiationTest as DT

function parse_benchmark_results_aux(bench::Benchmark, level; colnames, aggregators)
    data = DataFrame()
    data[!, :samples] = [length(bench.samples)]
    for agg in aggregators
        agg_bench = agg(bench)
        data[!, Symbol("time_$agg")] = [agg_bench.time]
        data[!, Symbol("bytes_$agg")] = [agg_bench.bytes]
        data[!, Symbol("allocs_$agg")] = [agg_bench.allocs]
        data[!, Symbol("compile_fraction_$agg")] = [agg_bench.compile_fraction]
        data[!, Symbol("gc_fraction_$agg")] = [agg_bench.gc_fraction]
        data[!, Symbol("evals_$agg")] = [agg_bench.evals]
    end
    return data
end

function parse_benchmark_results_aux(results, level; colnames, aggregators)
    level_symbol = Symbol(get(colnames, level, "level_$level"))
    data = DataFrame()
    for k in keys(results)
        v = results[k]
        subdata = parse_benchmark_results_aux(v, level + 1; colnames, aggregators)
        if !isempty(subdata)
            subdata[!, level_symbol] = fill(k, size(subdata, 1))
            append!(data, subdata; promote=true, cols=:setequal)
        end
    end
    if !isempty(data)
        select!(data, level_symbol, Not(level_symbol))
    end
    return data
end

function DT.parse_benchmark(
    results;
    colnames::Vector{Symbol}=[
        :backend,
        :operator,
        :function,
        :mutating,
        :input_type,
        :output_type,
        :input_size,
        :output_size,
        :operator_variant,
    ],
    aggregators=[minimum],
)
    return parse_benchmark_results_aux(results, 1; colnames, aggregators)
end

end
