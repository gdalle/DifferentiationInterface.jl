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
    @info "Testing formalities"
    @testset verbose = true "Formal tests" begin
        include("formal.jl")
    end

    @testset verbose = true "$folder" for folder in ["Single", "Double", "Internals"]
        folder_path = joinpath(@__DIR__, folder)
        @testset verbose = true "$(file[1:end-3])" for file in readdir(folder_path)
            @info "Testing $folder - $(file[1:end-3])"
            include(joinpath(folder_path, file))
        end
    end
end;
