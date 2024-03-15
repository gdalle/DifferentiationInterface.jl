using BenchmarkTools

include("utils.jl")

SUITE = include("benchmarks.jl")

some_backends = [
    AutoForwardDiff(),
    AutoForwardDiff(; chunksize=2),
    AutoReverseDiff(),
    AutoReverseDiff(; compile=true),
]
SUITE = make_suite(some_backends)

# Run benchmarks locally
BenchmarkTools.tune!(SUITE; verbose=true)
results = BenchmarkTools.run(SUITE; verbose=true)

# Parse into dataframe
include("dataframe.jl")
data = parse_benchmark_results(results; path=joinpath(@__DIR__, "results.csv"))

# Compare commits locally
# using BenchmarkCI
# BenchmarkCI.judge(baseline="origin/main")
# BenchmarkCI.displayjudgement()
