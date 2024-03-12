using BenchmarkTools

include("benchmarks.jl")

# Run benchmarks locally
# results = BenchmarkTools.run(SUITE; verbose=true)

# Compare commits locally
# using BenchmarkCI; BenchmarkCI.judge(baseline="origin/main"); BenchmarkCI.displayjudgement()

# Parse into dataframe
# include("dataframe.jl")
# data = parse_benchmark_results(results; path=joinpath(@__DIR__, "results.csv"))
