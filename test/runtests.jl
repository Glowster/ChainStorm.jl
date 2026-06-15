using ChainStorm
using BatchedTransformations: Translation, Rotation
using ForwardBackward: tensor
using LinearAlgebra
using Test

_mean(x) = sum(x) / length(x)

@testset "ChainStorm.jl" begin
    L, B = 5, 2
    locs = randn(Float32, 3, 1, L, B)
    rots = zeros(Float32, 3, 3, L, B)
    for b in 1:B, l in 1:L
        rots[:, :, l, b] .= Matrix{Float32}(I, 3, 3)
    end
    padmask = Bool[
        1 1
        1 1
        1 1
        0 1
        0 1
    ]
    aas = ones(Int, L, B)
    aas[.!padmask] .= 100
    chainids = ones(Int, L, B)
    resinds = repeat(reshape(collect(1:L), :, 1), 1, B)
    batch = (; chainids, resinds, padmask, aas, locs, rots)

    ts = training_sample(batch)
    xhat = ChainStorm.zero_state(batch)
    hatframes = Translation(tensor(xhat[1])) ∘ Rotation(tensor(xhat[2]))

    l_loc, l_rot = losses(hatframes, ts)
    @test isa(l_loc, Float32)
    @test isa(l_rot, Float32)

    per, per_loc, per_rot = per_sample_losses(hatframes, ts)
    @test isapprox(_mean(per), l_loc + l_rot; rtol = 1f-5, atol = 1f-5)
    @test isapprox(_mean(per_loc), l_loc; rtol = 1f-5, atol = 1f-5)
    @test isapprox(_mean(per_rot), l_rot; rtol = 1f-5, atol = 1f-5)

    weighted_loc, weighted_rot = losses(hatframes, ts; loc_weight = 0.1f0, rot_weight = 0.5f0)
    weighted_per, weighted_per_loc, weighted_per_rot = per_sample_losses(hatframes, ts; loc_weight = 0.1f0, rot_weight = 0.5f0)
    @test isapprox(_mean(weighted_per), weighted_loc + weighted_rot; rtol = 1f-5, atol = 1f-5)
    @test isapprox(_mean(weighted_per_loc), weighted_loc; rtol = 1f-5, atol = 1f-5)
    @test isapprox(_mean(weighted_per_rot), weighted_rot; rtol = 1f-5, atol = 1f-5)
end

include("equivariance_tests.jl")
