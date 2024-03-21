"""
    value_and_gradient!(f, grad, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient!(f::F, grad, backend::AbstractADType, x) where {F}
    return value_and_pullback!(f, grad, backend, x, one(eltype(x)))
end

"""
    value_and_gradient(f, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient(f::F, backend::AbstractADType, x) where {F}
    return value_and_pullback(f, backend, x, one(eltype(x)))
end
