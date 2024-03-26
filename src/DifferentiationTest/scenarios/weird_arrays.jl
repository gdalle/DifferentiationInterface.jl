"""
    weird_array_scenarios()

Create a vector of [`Scenario`](@ref)s involving weird types for testing differentiation.
"""
function weird_array_scenarios(; static=false, component=false, gpu=false)
    scenarios = Scenario[]
    if static
        ext = get_extension(
            DifferentiationInterface, :DifferentiationInterfaceStaticArraysExt
        )
        @assert !isnothing(ext)
        append!(scenarios, ext.static_scenarios())
    end
    if component
        ext = get_extension(
            DifferentiationInterface, :DifferentiationInterfaceComponentArraysExt
        )
        @assert !isnothing(ext)
        append!(scenarios, ext.component_scenarios())
    end
    if gpu
        ext = get_extension(DifferentiationInterface, :DifferentiationInterfaceJLArraysExt)
        @assert !isnothing(ext)
        append!(scenarios, ext.gpu_scenarios())
    end
    return scenarios
end
