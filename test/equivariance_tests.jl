using BatchedTransformations: Rotation, Translation, linear, translation
using ChainStorm
using LinearAlgebra
using Onion: Framemover
using Random
using Test

function _rotx(theta::Real)
    c, s = cos(Float32(theta)), sin(Float32(theta))
    return Float32[1 0 0; 0 c -s; 0 s c]
end

function _roty(theta::Real)
    c, s = cos(Float32(theta)), sin(Float32(theta))
    return Float32[c 0 s; 0 1 0; -s 0 c]
end

function _rotz(theta::Real)
    c, s = cos(Float32(theta)), sin(Float32(theta))
    return Float32[c -s 0; s c 0; 0 0 1]
end

_rot(theta::Real) = _rotz(theta) * _roty(0.7f0 * theta) * _rotx(-0.3f0 * theta)

function _make_batch(L::Int, B::Int; masked::Bool = false)
    locs = randn(Float32, 3, 1, L, B)
    rots = zeros(Float32, 3, 3, L, B)
    for b in 1:B, l in 1:L
        rots[:, :, l, b] .= _rot(0.17f0 * l + 0.31f0 * b)
    end

    padmask = trues(L, B)
    if masked
        padmask[end, 1] = false
        L > 4 && (padmask[end - 1, 1] = false)
    end

    aas = [mod1(l + 2b, 20) for l in 1:L, b in 1:B]
    chainids = ones(Int, L, B)
    resinds = repeat(reshape(collect(1:L), :, 1), 1, B)
    return (; chainids, resinds, padmask, aas, locs, rots)
end

function _frames_from_arrays(locs, rots)
    return Translation(locs) ∘ Rotation(rots)
end

function _global_transform_arrays(locs, rots, global_rots, global_trans)
    locs_out = similar(locs)
    rots_out = similar(rots)
    _, _, L, B = size(locs)
    for b in 1:B, l in 1:L
        R = global_rots[:, :, b]
        locs_out[:, 1, l, b] .= R * locs[:, 1, l, b] .+ global_trans[:, b]
        rots_out[:, :, l, b] .= R * rots[:, :, l, b]
    end
    return locs_out, rots_out
end

function _transform_batch(batch, global_rots, global_trans)
    locs, rots = _global_transform_arrays(batch.locs, batch.rots, global_rots, global_trans)
    return merge(batch, (; locs, rots))
end

function _transform_frames(frames, global_rots, global_trans)
    locs = values(translation(frames))
    rots = values(linear(frames))
    locs_t, rots_t = _global_transform_arrays(locs, rots, global_rots, global_trans)
    return _frames_from_arrays(locs_t, rots_t)
end

function _output_arrays(frames)
    return Array(values(translation(frames))), Array(values(linear(frames)))
end

function _assert_frames_close(actual, expected; atol = 8f-4, rtol = 8f-4)
    actual_locs, actual_rots = _output_arrays(actual)
    expected_locs, expected_rots = _output_arrays(expected)
    @test isapprox(actual_locs, expected_locs; atol, rtol)
    @test isapprox(actual_rots, expected_rots; atol, rtol)
end

function _make_global_transform(B::Int; rotation::Bool = true, translation::Bool = true)
    global_rots = zeros(Float32, 3, 3, B)
    global_trans = zeros(Float32, 3, B)
    for b in 1:B
        global_rots[:, :, b] .= rotation ? _rot(0.43f0 + 0.29f0 * b) : Matrix{Float32}(I, 3, 3)
        translation && (global_trans[:, b] .= Float32[0.8f0 * b, -0.4f0 + 0.1f0 * b, 1.2f0 - 0.2f0 * b])
    end
    return global_rots, global_trans
end

function _run_model_equivariance_case(; self_conditioned::Bool, masked::Bool, rotation::Bool, translation::Bool)
    Random.seed!(2026)
    L, B = 6, 2
    model = ChainStormV1(24, 2, 1)
    batch = _make_batch(L, B; masked)
    prev_batch = merge(batch, (;
        locs = batch.locs .+ 0.13f0 .* randn(Float32, size(batch.locs)),
        rots = copy(batch.rots),
    ))
    sc_batch = merge(batch, (;
        locs = batch.locs .+ 0.07f0 .* randn(Float32, size(batch.locs)),
        rots = copy(batch.rots),
    ))

    Xt = ChainStorm.compound_state(batch)
    prev_frames = _frames_from_arrays(prev_batch.locs, prev_batch.rots)
    sc_frames = self_conditioned ? _frames_from_arrays(sc_batch.locs, sc_batch.rots) : nothing

    t = Float32[0.17 0.73]
    aas = batch.aas
    chainids = batch.chainids
    resinds = batch.resinds
    disto = randn(Float32, 64, L, L, B)
    delta_ts = Float32[1f-9 100f-9]
    temps = Float32[320 450]

    out = model(t, Xt, aas, chainids, resinds, disto, prev_frames, delta_ts, temps; sc_frames)

    global_rots, global_trans = _make_global_transform(B; rotation, translation)
    batch_t = _transform_batch(batch, global_rots, global_trans)
    prev_frames_t = _transform_frames(prev_frames, global_rots, global_trans)
    sc_frames_t = self_conditioned ? _transform_frames(sc_frames, global_rots, global_trans) : nothing

    out_t = model(
        t,
        ChainStorm.compound_state(batch_t),
        aas,
        chainids,
        resinds,
        disto,
        prev_frames_t,
        delta_ts,
        temps;
        sc_frames = sc_frames_t,
    )

    expected = _transform_frames(out, global_rots, global_trans)
    _assert_frames_close(out_t, expected)
end

@testset "Framemover SE(3) equivariance" begin
    Random.seed!(2026)
    L, B, dim = 5, 2, 24
    batch = _make_batch(L, B)
    frames = _frames_from_arrays(batch.locs, batch.rots)
    x = randn(Float32, dim, L, B)
    mover = Framemover(dim)
    t = Float32[0.2 0.8]

    global_rots, global_trans = _make_global_transform(B; rotation = true, translation = true)
    moved = mover(frames, x; t)
    moved_t = mover(_transform_frames(frames, global_rots, global_trans), x; t)
    expected = _transform_frames(moved, global_rots, global_trans)
    _assert_frames_close(moved_t, expected; atol = 2f-5, rtol = 2f-5)
end

@testset "ChainStormV1 SE(3) equivariance" begin
    @testset "translation only" begin
        _run_model_equivariance_case(
            self_conditioned = false,
            masked = false,
            rotation = false,
            translation = true,
        )
    end

    @testset "rotation only" begin
        _run_model_equivariance_case(
            self_conditioned = false,
            masked = false,
            rotation = true,
            translation = false,
        )
    end

    @testset "batched rigid transforms with masks" begin
        _run_model_equivariance_case(
            self_conditioned = false,
            masked = true,
            rotation = true,
            translation = true,
        )
    end

    @testset "self-conditioned batched rigid transforms" begin
        _run_model_equivariance_case(
            self_conditioned = true,
            masked = true,
            rotation = true,
            translation = true,
        )
    end
end
