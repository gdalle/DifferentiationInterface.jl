## Backend-scenario

function compatible(backend::AbstractADType, scen::AbstractScenario)
    if ismutating(scen)
        return Bool(supports_mutation(backend))
    end
    return true
end
