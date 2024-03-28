## Backend-scenario

function compatible(backend::AbstractADType, scen::AbstractScenario)
    if ismutating(scen)
        return Bool(supports_mutation(backend))
    end
    return true
end

function compatible(
    backend::AbstractADType, ::PushforwardScenario{mutating}
) where {mutating}
    if mutating
        return Bool(supports_mutation(backend)) && Bool(supports_pushforward(backend))
    end
    return Bool(supports_pushforward(backend))
end

function compatible(backend::AbstractADType, ::PullbackScenario{mutating}) where {mutating}
    if mutating
        return Bool(supports_mutation(backend)) && Bool(supports_pullback(backend))
    end
    return Bool(supports_pullback(backend))
end
