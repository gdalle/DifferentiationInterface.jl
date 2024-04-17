using DifferentiationInterface: pick_chunksize, DEFAULT_CHUNKSIZE

@test pick_chunksize.(1:DEFAULT_CHUNKSIZE) == 1:DEFAULT_CHUNKSIZE
@test all(
    pick_chunksize.((DEFAULT_CHUNKSIZE + 1):(5DEFAULT_CHUNKSIZE)) .<= DEFAULT_CHUNKSIZE
)
@test all(
    pick_chunksize.((DEFAULT_CHUNKSIZE + 1):(5DEFAULT_CHUNKSIZE)) .>= DEFAULT_CHUNKSIZE / 2
)
