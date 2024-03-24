"""
    weird_array_scenarios()

Create a vector of [`Scenario`](@ref)s involving weird types for testing differentiation.
"""
function weird_array_scenarios(; static=true, component=true, gpu=true)
    scenarios = Scenario[]
    if static
        ext = get_extension(
            DifferentiationInterface, :DifferentiationInterfaceStaticArraysExt
        )
        append!(scenarios, ext.static_scenarios_allocating())
    end
    if component
        ext = get_extension(
            DifferentiationInterface, :DifferentiationInterfaceComponentArraysExt
        )
        append!(scenarios, ext.component_scenarios_allocating())
    end
    if gpu
        ext = get_extension(DifferentiationInterface, :DifferentiationInterfaceJLArraysExt)
        append!(scenarios, ext.gpu_scenarios_allocating())
    end
    return scenarios
end
