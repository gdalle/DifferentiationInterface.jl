using Pkg

Pkg.develop(
    Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "..", "DifferentiationInterfaceTest"))
)

using ADTypes
using DifferentiationInterface
using Test

function MyAutoSparse(backend::AbstractADType)
    coloring_algorithm = DifferentiationInterface.GreedyColoringAlgorithm()
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector()
    return AutoSparse(backend; sparsity_detector, coloring_algorithm)
end

LOGGING = get(ENV, "CI", "false") == "false"

## Main tests

@testset verbose = true "DifferentiationInterface.jl" begin
    @testset verbose = true "Formal tests" begin
        include("formal.jl")
    end

    @testset verbose = true "$folder" for folder in [
        "first_order", "second_order", "sparse", "translation", "efficiency", "internals"
    ]
        folder_path = joinpath(@__DIR__, folder)
        @testset verbose = true "$file" for file in readdir(folder_path)
            include(joinpath(folder_path, file))
        end
    end
end;
