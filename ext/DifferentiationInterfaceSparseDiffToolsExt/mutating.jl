struct SparseDiffToolsMutatingJacobianExtras{C} <: JacobianExtras
    cache::C
end

for AutoSparse in SPARSE_BACKENDS
    @eval begin
        ## Jacobian

        function DI.prepare_jacobian(
            f!, backend::$AutoSparse, y::AbstractArray, x::AbstractArray
        )
            cache = sparse_jacobian_cache(
                backend, SymbolicsSparsityDetection(), f!, similar(y), x
            )
            return SparseDiffToolsMutatingJacobianExtras(cache)
        end

        function DI.value_and_jacobian!!(
            f!,
            y,
            jac,
            backend::$AutoSparse,
            x,
            extras::SparseDiffToolsMutatingJacobianExtras,
        )
            sparse_jacobian!(jac, backend, extras.cache, f!, y, x)
            f!(y, x)
            return y, jac
        end
    end
end
