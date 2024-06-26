function package_name(b::AbstractADType)
    s = string(b)
    return s[5:(findfirst('(', s) - 1)]
end

package_name(b::AutoSparse) = package_name(dense_ad(b))
