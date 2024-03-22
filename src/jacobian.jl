"""
    value_and_jacobian!(f, jac, backend, x, [extras]) -> (y, jac)
    value_and_jacobian!(f!, y, jac, backend, x, [extras]) -> (y, jac)
"""
function value_and_jacobian!(f::F, jac, backend::AbstractADType, x) where {F}
    return value_and_jacobian_aux!(f, jac, backend, x, supports_pushforward(backend))
end

function value_and_jacobian!(f!::F, y, jac, backend::AbstractADType, x) where {F}
    return value_and_jacobian_aux!(f!, y, jac, backend, x, supports_pushforward(backend))
end

## Forward mode

function value_and_jacobian_aux!(
    f::F, _jac, backend, x::Number, ::PushforwardSupported
) where {F}
    y, jac::Number = value_and_derivative(f, backend, x)
    return y, jac
end

function value_and_jacobian_aux!(f::F, jac, backend, x, ::PushforwardSupported) where {F}
    y = f(x)
    for (k, j) in enumerate(CartesianIndices(x))
        dx_j = basisarray(backend, x, j)
        jac_col_j = reshape(view(jac, :, k), size(y))
        value_and_pushforward!(f, jac_col_j, backend, x, dx_j)
    end
    return y, jac
end

function value_and_jacobian_aux!(
    f!::F, y, jac, backend, x, ::PushforwardSupported
) where {F}
    y = f(x)
    for (k, j) in enumerate(CartesianIndices(x))
        dx_j = basisarray(backend, x, j)
        jac_col_j = reshape(view(jac, :, k), size(y))
        value_and_pushforward!(f!, y, jac_col_j, backend, x, dx_j)
    end
    return y, jac
end

## Reverse mode

function value_and_jacobian_aux!(
    f::F, _jac, backend, x::Number, ::PushforwardNotSupported
) where {F}
    y, jac::Number = value_and_gradient(f, backend, x)
    return y, jac
end

function value_and_jacobian_aux!(f::F, jac, backend, x, ::PushforwardNotSupported) where {F}
    y = f(x)
    for (k, i) in enumerate(CartesianIndices(y))
        dy_i = basisarray(backend, y, i)
        jac_row_i = reshape(view(jac, k, :), size(x))
        value_and_pullback!(f, jac_row_i, backend, x, dy_i)
    end
    return y, jac
end

function value_and_jacobian_aux!(
    f!::F, y, jac, backend, x, ::PushforwardNotSupported
) where {F}
    for (k, i) in enumerate(CartesianIndices(y))
        dy_i = basisarray(backend, y, i)
        jac_row_i = reshape(view(jac, k, :), size(x))
        value_and_pullback!(f!, y, jac_row_i, backend, x, dy_i)
    end
    return y, jac
end

"""
    value_and_jacobian(f, backend, x, [extras]) -> (y, jac)
"""
function value_and_jacobian(f::F, backend::AbstractADType, x) where {F}
    return value_and_jacobian_aux(f, backend, x, supports_pushforward(backend))
end

## Forward mode

function value_and_jacobian_aux(f::F, backend, x::Number, ::PushforwardSupported) where {F}
    y, jac::Number = value_and_derivative(f, backend, x)
    return y, jac
end

function value_and_jacobian_aux(
    f::F, backend, x::AbstractArray, ::PushforwardSupported
) where {F}
    y = f(x)
    jac = stack(CartesianIndices(x); dims=2) do j
        dx_j = basisarray(backend, x, j)
        jac_col_j = last(value_and_pushforward(f, backend, x, dx_j))
        vec(jac_col_j)
    end
    return y, jac
end

## Reverse mode

function value_and_jacobian_aux(
    f::F, backend, x::Number, ::PushforwardNotSupported
) where {F}
    y, jac::Number = value_and_gradient(f, backend, x)
    return y, jac
end

function value_and_jacobian_aux(
    f::F, backend, x::AbstractArray, ::PushforwardNotSupported
) where {F}
    y = f(x)
    jac = stack(CartesianIndices(y); dims=1) do i
        dy_i = basisarray(backend, y, i)
        jac_row_i = last(value_and_pullback(f, backend, x, dy_i))
        vec(jac_row_i)
    end
    return y, jac
end
