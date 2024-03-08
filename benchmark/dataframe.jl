using BenchmarkTools
using CSV
using DataFrames
using Statistics

function parse_benchmark_results_aux(result::BenchmarkTools.Trial, level=nothing)
    data = DataFrame(
        :samples => [length(result.times)],
        :time_median => [median(result.times)],
        :memory_median => [median(result.memory)],
        :allocs_median => [median(result.allocs)],
    )
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
