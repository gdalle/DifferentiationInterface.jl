## Backend-operator

function compatible(::AbstractADType, ::Function)
    return true
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

function compatible(::typeof(derivative), scen::Scenario)
    return scen.x isa Number
end

function compatible(::typeof(second_derivative), scen::Scenario{mutating}) where {mutating}
    return scen.x isa Number && !mutating
end

function compatible(
    ::Union{typeof(gradient),typeof(hvp)}, scen::Scenario{mutating}
) where {mutating}
    return scen.y isa Number && !mutating
end

function compatible(::typeof(hessian), scen::Scenario{mutating}) where {mutating}
    return scen.y isa Number && !mutating && scen.x isa AbstractArray
end

function compatible(::typeof(jacobian), scen::Scenario)
    return scen.x isa AbstractArray && scen.y isa AbstractArray
end

## Triplet

function compatible(backend::AbstractADType, op::Function, scen::Scenario)
    return compatible(backend, op) && compatible(backend, scen) && compatible(op, scen)
end
