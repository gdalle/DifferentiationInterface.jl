using BenchmarkTools

include("benchmarks.jl")

# Run benchmarks locally
# BenchmarkTools.tune!(SUITE; verbose=true)
results = BenchmarkTools.run(SUITE; verbose=true)

# Parse into dataframe
include("dataframe.jl")
data = parse_benchmark_results(results; path=joinpath(@__DIR__, "results.csv"))

# Compare commits locally
# using BenchmarkCI
# BenchmarkCI.judge(baseline="origin/main")
# BenchmarkCI.displayjudgement()
