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
