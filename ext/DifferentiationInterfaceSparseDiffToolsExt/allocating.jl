for AutoSparse in SPARSE_BACKENDS
    @eval begin

        ## Jacobian

        function DI.prepare_jacobian(f, backend::$AutoSparse, x::AbstractArray)
            return sparse_jacobian_cache(
                backend, SymbolicsSparsityDetection(), f, x; fx=f(x)
            )
        end

        function DI.value_and_jacobian!!(f, jac, backend::$AutoSparse, x, cache)
            sparse_jacobian!(jac, backend, cache, f, x)
            return f(x), jac
        end

        function DI.jacobian!!(f, jac, backend::$AutoSparse, x, cache)
            sparse_jacobian!(jac, backend, cache, f, x)
            return jac
        end

        function DI.value_and_jacobian(f, backend::$AutoSparse, x, cache)
            return f(x), sparse_jacobian(backend, cache, f, x)
        end

        function DI.jacobian(f, backend::$AutoSparse, x, cache)
            return sparse_jacobian(backend, cache, f, x)
        end
    end
end
