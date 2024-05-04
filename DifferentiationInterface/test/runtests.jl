using Pkg
Pkg.develop(
    Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "..", "DifferentiationInterfaceTest"))
)

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
            @info "Testing $folder - $(file[1:end-3])"
            if !(
                contains(file, "Diffractor") ||
                contains(file, "FastDifferentiation") ||
                contains(file, "PolyesterForwardDiff") ||
                contains(file, "Symbolics") ||
                contains(file, "Tapir"),
            )
                include(joinpath(folder_path, file))
            end
        end
    end
end;
