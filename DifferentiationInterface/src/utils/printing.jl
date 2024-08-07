function package_name(b::AbstractADType)
    s = string(b)
    k = findfirst('(', s)
    if isnothing(k)
        throw(ArgumentError("Cannot parse backend into package"))
    else
        return s[5:(k - 1)]
    end
end

package_name(b::AutoSparse) = package_name(dense_ad(b))

function document_preparation(operator_name::AbstractString; same_point=false)
    if same_point
        return "To improve performance via operator preparation, refer to [`prepare_$(operator_name)`](@ref) and [`prepare_$(operator_name)_same_point`](@ref)."
    else
        return "To improve performance via operator preparation, refer to [`prepare_$(operator_name)`](@ref)."
    end
end
