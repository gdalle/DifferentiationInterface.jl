ADDITIONAL_BACKENDS_10 = [
    "Diffractor", "FastDifferentiation", "PolyesterForwardDiff", "Symbolics", "Tapir"
]

## Weird Pkg mumbo-jumbo

using Pkg

push!(Base.LOAD_PATH, Base.active_project())

Pkg.activate(; temp=true)

Pkg.develop([
    Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "..", "DifferentiationInterface")),
    Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "..", "DifferentiationInterfaceTest")),
])

@static if VERSION >= v"1.10"
    Pkg.add(ADDITIONAL_BACKENDS_10)
end

## Actual stuff that matters

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
                any(contains(file, name) for name in ADDITIONAL_BACKENDS_10)
            )
                @info "Skipping $folder - $(file[1:end-3])"
            else
                @info "Testing $folder - $(file[1:end-3])"
                include(joinpath(folder_path, file))
            end
        end
    end
end;
