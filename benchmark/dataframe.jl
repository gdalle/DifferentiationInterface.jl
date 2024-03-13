using BenchmarkTools
using CSV
using DataFrames
using Statistics

function parse_benchmark_results_aux(
    result::BenchmarkTools.Trial, level=nothing; aggregators=[minimum, median]
)
    data = DataFrame()
    for agg in aggregators
        data[!, Symbol("time_$agg")] = [agg(result.times)]
        data[!, Symbol("memory_$agg")] = [agg(result.memory)]
        data[!, Symbol("allocs_$agg")] = [agg(result.allocs)]
        data[!, Symbol("gctime_$agg")] = [agg(result.gctimes)]
    end
    data[!, :samples] = [length(result.times)]
    data[!, :params_evals] = [result.params.evals]
    data[!, :params_samples] = [result.params.samples]
    data[!, :params_seconds] = [result.params.seconds]
    return data
end

function parse_benchmark_results_aux(results::BenchmarkGroup, level=1)
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

function parse_benchmark_results(results::BenchmarkGroup; path=nothing)
    data = parse_benchmark_results_aux(results)
    if !isnothing(path)
        open(path, "w") do file
            CSV.write(file, data)
        end
    end
    return data
end
