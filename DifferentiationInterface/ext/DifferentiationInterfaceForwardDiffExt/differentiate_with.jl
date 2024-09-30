function (dw::DifferentiateWith)(x::Dual{T,V,N}) where {T,V,N}
    @compat (; f, backend) = dw
    xval = myvalue(T, x)
    tx = mypartials(T, Val(N), x)
    y, ty = DI.value_and_pushforward(f, backend, xval, tx)
    return make_dual(T, y, ty)
end

function (dw::DifferentiateWith)(x::AbstractArray{Dual{T,V,N}}) where {T,V,N}
    @compat (; f, backend) = dw
    xval = myvalue(T, x)
    tx = mypartials(T, Val(N), x)
    y, ty = DI.value_and_pushforward(f, backend, xval, tx)
    return make_dual(T, y, ty)
end
