for AutoSparse in SPARSE_BACKENDS
    @eval begin
        ## Jacobian

        function DI.prepare_jacobian(
            f!, backend::$AutoSparse, y::AbstractArray, x::AbstractArray
        )
            return sparse_jacobian_cache(
                backend, SymbolicsSparsityDetection(), f!, similar(y), x
            )
        end

        function DI.value_and_jacobian!!(f!, y, jac, backend::$AutoSparse, x, cache)
            sparse_jacobian!(jac, backend, cache, f!, y, x)
            f!(y, x)
            return y, jac
        end
    end
end
