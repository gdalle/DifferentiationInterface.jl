using Pkg
Pkg.develop(
    Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "..", "DifferentiationInterfaceTest"))
)

BACKENDS_INCOMPATIBLE_WITH_LTS = [
    "Diffractor", "FastDifferentiation", "PolyesterForwardDiff", "Symbolics", "Tapir"
]
@static if VERSION >= v"1.10"
    Pkg.add(BACKENDS_INCOMPATIBLE_WITH_LTS)
end

using ADTypes
using DifferentiationInterface
using SparseConnectivityTracer: SparseConnectivityTracer
using Test

function MyAutoSparse(backend::AbstractADType)
    coloring_algorithm = DifferentiationInterface.GreedyColoringAlgorithm()
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector()
    return AutoSparse(backend; sparsity_detector, coloring_algorithm)
end

LOGGING = get(ENV, "CI", "false") == "false"

## Main tests

@testset verbose = true "DifferentiationInterface.jl" begin
    @static if VERSION >= v"1.10"
        @info "Testing formalities"
        @testset verbose = true "Formal tests" begin
            include("formal.jl")
        end
    end

    @testset verbose = true "$folder" for folder in ["Single", "Double", "Internals"]
        folder_path = joinpath(@__DIR__, folder)
        @testset verbose = true "$(file[1:end-3])" for file in readdir(folder_path)
            if (
                VERSION < v"1.10" &&
                any(contains(file, name) for name in BACKENDS_INCOMPATIBLE_WITH_LTS)
            )
                @info "Skipping $folder - $(file[1:end-3])"
            else
                @info "Testing $folder - $(file[1:end-3])"
                include(joinpath(folder_path, file))
            end
        end
    end
end;
