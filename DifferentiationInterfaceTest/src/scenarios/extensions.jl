"""
    static_scenarios(rng=Random.default_rng())

Create a vector of [`Scenario`](@ref)s with static array types from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).

!!! warning
    This function requires StaticArrays.jl to be loaded (it is implemented in a package extension).
"""
function static_scenarios end

"""
    component_scenarios(rng=Random.default_rng())

Create a vector of [`Scenario`](@ref)s with component array types from [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl).

!!! warning
    This function requires ComponentArrays.jl to be loaded (it is implemented in a package extension).
"""
function component_scenarios end

"""
    gpu_scenarios(rng=Random.default_rng())

Create a vector of [`Scenario`](@ref)s with GPU array types from [JLArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl/tree/master/lib/JLArrays).

!!! warning
    This function requires JLArrays.jl to be loaded (it is implemented in a package extension).
"""
function gpu_scenarios end
