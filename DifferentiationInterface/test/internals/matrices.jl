@testset "$(typeof(A))" for A in (
    rand(0:1, 10, 10),
    transpose(rand(0:1, 10, 10)),
    sprand(100, 100, 0.1),
    transpose(sprand(100, 100, 0.1)),
)
    A_colmajor = DI.col_major(A)
    A_rowmajor = DI.row_major(A)

    @test A_colmajor == A
    @test A_rowmajor == A
end

@testset "$(typeof(A))" for A in (rand(0:1, 10, 10), sprand(100, 100, 0.1))
    A_colmajor = DI.col_major(A)
    A_rowmajor = DI.row_major(A)

    for i in axes(A, 1)
        @test DI.nz_in_row(A_rowmajor, i) == DI.nz_in_row(Matrix(A_rowmajor), i)
    end
    for j in axes(A, 2)
        @test DI.nz_in_col(A_colmajor, j) == DI.nz_in_col(Matrix(A_colmajor), j)
    end
end
