using ADTypes
using DifferentiationInterface
using Pkg
using SparseConnectivityTracer
using Test

push!(Base.LOAD_PATH, Base.active_project())
Pkg.activate(; temp=true)

DI_PATH = joinpath(@__DIR__, "..", "..", "DifferentiationInterface")
DIT_PATH = joinpath(@__DIR__, "..", "..", "DifferentiationInterfaceTest")
if isdir(DI_PATH)
    Pkg.develop(; path=DI_PATH)
end
if isdir(DIT_PATH)
    Pkg.develop(; path=DIT_PATH)
end

BACKENDS_1_6 = [
    "FiniteDifferences",  #
    "ForwardDiff",
    "ReverseDiff",
    "Tracker",
    "Zygote",
]

BACKENDS_1_10 = [
    "Diffractor",  # 
    "Enzyme",
    "FiniteDiff",
    "FastDifferentiation",
    "PolyesterForwardDiff",
    "Symbolics",
    "Tapir",
]

@static if VERSION >= v"1.10"
    Pkg.add(vcat(BACKENDS_1_6, BACKENDS_1_10))
else
    Pkg.add(vcat(BACKENDS_1_6))
end

function MyAutoSparse(backend::AbstractADType)
    coloring_algorithm = GreedyColoringAlgorithm()
    sparsity_detector = TracerSparsityDetector()
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
            endswith(file, ".jl") || continue
            if (
                VERSION < v"1.10" && any(
                    part == backend for part in split(file[1:(end - 3)], '-') for
                    backend in BACKENDS_1_10
                )
            )
                @info "Skipping $folder - $(file[1:end-3])"
            else
                @info "Testing $folder - $(file[1:end-3])"
                include(joinpath(folder_path, file))
            end
        end
    end
end;
