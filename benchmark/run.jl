using BenchmarkTools

include("benchmarks.jl")
include("dataframe.jl")

# Run benchmarks locally
results = BenchmarkTools.run(SUITE; verbose=true)

# Parse into dataframe
data = parse_benchmark_results(results; path=joinpath(@__DIR__, "results.csv"))

# Compare commits locally
# using BenchmarkCI
# BenchmarkCI.judge(baseline="origin/main")
# BenchmarkCI.displayjudgement()
