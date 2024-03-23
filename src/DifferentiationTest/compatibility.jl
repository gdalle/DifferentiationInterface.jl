## Backend-operator

function compatible(::AbstractADType, ::Function)
    return true
end

function compatible(backend::AbstractADType, ::typeof(value_and_pushforward))
    return Bool(supports_pushforward(backend))
end

function compatible(backend::AbstractADType, ::typeof(value_and_pullback))
    return Bool(supports_pullback(backend))
end

## Backend-scenario

function compatible(::AbstractADType, ::Scenario{false})
    return true
end

function compatible(backend::AbstractADType, ::Scenario{true})
    return Bool(supports_mutation(backend))
end

## Operator-scenario

function compatible(::Function, ::Scenario)
    return true
end

function compatible(::typeof(value_and_derivative), scen::Scenario)
    return scen.x isa Number
end

function compatible(::typeof(value_and_gradient), scen::Scenario{mutating}) where {mutating}
    return scen.y isa Number && !mutating
end

function compatible(::typeof(value_and_jacobian), scen::Scenario)
    return scen.x isa AbstractArray && scen.y isa AbstractArray
end

## Triplet

function compatible(backend::AbstractADType, op::Function, scen::Scenario)
    return compatible(backend, op) && compatible(backend, scen) && compatible(op, scen)
end
