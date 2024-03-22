"""
    value_and_derivative!(f, der, backend, x, [extras]) -> (y, der)
    value_and_derivative!(f!, y, der, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative!(f::F, der, backend::AbstractADType, x) where {F}
    return value_and_pushforward!(f, der, backend, x, one(x))
end

function value_and_derivative!(f!::F, y, der, backend::AbstractADType, x) where {F}
    return value_and_pushforward!(f!, y, der, backend, x, one(x))
end

"""
    value_and_derivative(f, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative(f::F, backend::AbstractADType, x) where {F}
    return value_and_derivative_aux(f, backend, x, one(x))
end
