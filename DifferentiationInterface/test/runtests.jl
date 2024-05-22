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

function MyAutoSparse(backend::AbstractADType)
    coloring_algorithm = GreedyColoringAlgorithm()
    sparsity_detector = TracerSparsityDetector()
    return AutoSparse(backend; sparsity_detector, coloring_algorithm)
end

LOGGING = get(ENV, "CI", "false") == "false"

GROUP = get(ENV, "JULIA_DI_TEST_GROUP", "All")

## Main tests

@testset verbose = true "DifferentiationInterface.jl" begin
    if GROUP == "Formalities" || GROUP == "All"
        @testset "$file" for file in readdir(joinpath(@__DIR__, "Formalities"))
            @info "Testing $(file[end-3:end])"
            include(joinpath(@__DIR__, "Formalities", file))
        end
    end

    if GROUP == "Internals" || GROUP == "All"
        @testset "$file" for file in readdir(joinpath(@__DIR__, "Internals"))
            @info "Testing $(file[end-3:end])"
            include(joinpath(@__DIR__, "Internals", file))
        end
    end

    if GROUP == "All"
        # do stuff
        nothing
    elseif startswith(GROUP, "Single")
        backend1_str = split(GROUP, '/')[2]
        @info "Testing Single/$backend_str"
        if VERSION >= v"1.10" || backend1_str in BACKENDS_1_6
            Pkg.add(backend1_str)
            include(joinpath(@__DIR__, "Single", "$backend1_str.jl"))
        end
    elseif startswith(GROUP, "Double")
        backend1_str, backend2_str = split(split(GROUP, '/')[2], '-')
        @info "Testing Single/$backend1_str-$backend2_str"
        if VERSION >= v"1.10" ||
            (backend1_str in BACKENDS_1_6 && backend2_str in BACKENDS_1_6)
            Pkg.add([backend1_str, backend2_str])
            include(joinpath(@__DIR__, "Double", "$backend1_str-$backend2_str.jl"))
        end
    end
end;
