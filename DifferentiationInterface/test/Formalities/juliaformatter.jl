using DifferentiationInterface
using JuliaFormatter: JuliaFormatter
using Test

@test JuliaFormatter.format(DifferentiationInterface; verbose=false, overwrite=false)
