struct DeferredGradient{F,M<:Mode}
    f::F
    mode::M
end

function (def_grad::DeferredGradient{F,<:ForwardMode})(z) where {F}
    return error("Not implemented yet")
end

function (def_grad::DeferredGradient{F,<:ReverseMode})(z) where {F}
    @compat (; f, mode) = def_grad
    grad = make_zero(z)
    autodiff_deferred(mode, Const(f), Active, Duplicated(z, grad))
    return grad
end

struct DeferredDerivative{F,M<:Mode}
    f::F
    mode::M
end

function (def_der::DeferredDerivative{F,<:ForwardMode})(z) where {F}
    @compat (; f, mode) = def_der
    return only(autodiff_deferred(mode, Const(f), DuplicatedNoNeed, Duplicated(z, one(z))))
end

function (def_der::DeferredDerivative{F,<:ReverseMode})(z) where {F}
    return error("Not implemented yet")
end
