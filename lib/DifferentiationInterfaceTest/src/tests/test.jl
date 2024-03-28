"""
    test_differentiation(backends, [operators, scenarios]; [kwargs...])

Test a list of `backends` for a list of `operators` on a list of `scenarios`.

# Default arguments

- `operators::Vector{Function}`: the list `[pushforward, pullback,derivative, gradient, jacobian, second_derivative, hvp, hessian]`
- `scenarios::Vector{Scenario}`: the output of [`default_scenarios()`](@ref)

# Keyword arguments

Testing:

- `correctness=true`: whether to compare the differentiation results with the theoretical values specified in each scenario. If a backend object like `correctness=AutoForwardDiff()` is passed instead of a boolean, the results will be compared using that reference backend as the ground truth. 
- `call_count=false`: whether to check that the function is called the right number of times
- `type_stability=false`: whether to check type stability with JET.jl (thanks to `@test_opt`)
- `detailed=false`: whether to print a detailed or condensed test log

Filtering:

- `input_type=Any`: restrict scenario inputs to subtypes of this
- `output_type=Any`: restrict scenario outputs to subtypes of this
- `allocating=true`: consider operators for allocating functions
- `mutating=true`: consider operators for mutating functions
- `first_order=true`: consider first order operators
- `second_order=true`: consider second order operators
- `excluded=Symbol[]`: list of excluded operators

Options:

- `logging=true`: whether to log progress
- `isapprox=isapprox`: function used to compare objects, with the standard signature `isapprox(x, y; atol, rtol)`
- `atol=0`: absolute precision for correctness testing (when comparing to the reference outputs)
- `rtol=1e-3`: relative precision for correctness testing (when comparing to the reference outputs)
"""
function test_differentiation(
    backends::Vector{<:AbstractADType},
    operators::Vector{<:Function}=all_operators(),
    scenarios::Vector{<:Scenario}=default_scenarios();
    # testing
    correctness::Union{Bool,AbstractADType}=true,
    type_stability::Bool=false,
    call_count::Bool=false,
    detailed=false,
    # filtering
    input_type::Type=Any,
    output_type::Type=Any,
    allocating=true,
    mutating=true,
    first_order=true,
    second_order=true,
    excluded::Vector{<:Function}=Function[],
    # options
    logging=false,
    isapprox=isapprox,
    atol=0,
    rtol=1e-3,
)
    operators = filter_operators(operators; first_order, second_order, excluded)
    scenarios = filter_scenarios(scenarios; input_type, output_type, allocating, mutating)

    if correctness isa AbstractADType
        scenarios = change_ref.(scenarios, Ref(correctness))
    end

    title =
        "Differentiation tests -" *
        (correctness != false ? " correctness" : "") *
        (call_count ? " calls" : "") *
        (type_stability ? " types" : "")

    @testset verbose = true "$title" begin
        @testset verbose = detailed "$(backend_string(backend))" for backend in backends
            @testset verbose = detailed "$op" for op in operators
                @testset "$scen" for scen in filter(scenarios) do scen
                    compatible(backend, op, scen)
                end
                    logging &&
                        @info "Testing: $(backend_string(backend)) - $op - $(string(scen))"
                    correctness != false && @testset "Correctness" begin
                        test_correctness(backend, op, scen; isapprox, atol, rtol)
                    end
                    call_count && @testset "Call count" begin
                        test_call_count(backend, op, scen)
                    end
                    type_stability && @testset "Type stability" begin
                        test_jet(backend, op, scen)
                    end
                end
            end
        end
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Shortcut for a single backend.
"""
function test_differentiation(backend::AbstractADType, args...; kwargs...)
    return test_differentiation([backend], args...; kwargs...)
end
