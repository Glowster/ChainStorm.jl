ipa(l, f, x, pf, c, m) = l(f, x, pair_feats = pf, cond = c, mask = m)
crossipa(l, f1, f2, x, pf, c, m) = l(f1, f2, x, pair_feats = pf, cond = c, mask = m)

# @eval Flux begin

# function loadmodel!(dst, src; filter = _ -> true, cache = Base.IdSet())
#     ldsts = _filter_children(filter, Functors.children(dst))
#     lsrcs = _filter_children(filter, Functors.children(src))
#     keys_ldsts = keys(ldsts)
#     keys_lsrcs = keys(lsrcs)
#     #collect(keys_ldsts) == collect(keys_lsrcs) || throw(ArgumentError("Tried to load $(keys_lsrcs) into $(keys_ldsts) but the structures do not match."))
    
#     for k in keys_lsrcs
#         lsrc, ldst = lsrcs[k], ldsts[k]
#         if ldst in cache # we already loaded this parameter before
#         _tie_check(ldst, lsrc)
#         elseif Functors.isleaf(ldst) # our first time loading this leaf
#         push!(cache, ldst)
#         loadleaf!(ldst, lsrc)
#         else # this isn't a leaf
#         loadmodel!(ldst, lsrc; filter, cache)
#         end
#     end
    
#     return dst
# end

# end

struct ChainStormV1{L}
    layers::L
end
Flux.@layer ChainStormV1
function ChainStormV1(dim::Int = 384, depth::Int = 6, f_depth::Int = 6)
    layers = (;
        depth = depth,
        f_depth = f_depth,
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        cond_t_encoding = Dense(dim => dim, bias=false),
        cond_delta_t_encoding = Dense(dim => dim, bias=false),
        cond_temp_encoding = Dense(dim => dim, bias=false),
        AApre_t_encoding = Dense(dim => dim, bias=false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias=false),
        disto_project = Dense(64 => 32, bias=false),
        AAencoder = Dense(21 => dim, bias=false),
        selfcond_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth],
        prev_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth],
        prev_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth], #TODO not self cond, rename this
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = AdaLN(dim, dim), ln2 = AdaLN(dim, dim)) for _ in 1:depth], #TODO not selfcond, rename this
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        #AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),
        cond_t_encoding_2 = Dense(dim => dim, bias=false),
        cond_delta_t_encoding_2 = Dense(dim => dim, bias=false),
        cond_temp_encoding_2 = Dense(dim => dim, bias=false),
        AApre_t_encoding_2 = Dense(dim => dim, bias=false),
        #pair_rff_2 = RandomFourierFeatures(2 => 64, 1f0),
        pair_project_2 = Dense(64 => 32, bias=false),
        #disto_project_2 = Dense(64 => 32, bias=false),
        AAencoder_2 = Dense(21 => dim, bias=false),
        #selfcond_crossipa_2 = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth],
        #selfcond_selfipa_2 = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth],
        modelpred_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth],
        modelpred_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth], #TODO not self cond, rename this
        ipa_blocks_2 = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = AdaLN(dim, dim), ln2 = AdaLN(dim, dim)) for _ in 1:depth], #TODO not selfcond, rename this
        framemovers_2 = [Framemover(dim) for _ in 1:f_depth],
    )
    return ChainStormV1(layers)
end



#function (fc::ChainStormV1)(t, Xt, chainids, resinds; sc_frames = nothing)
function (fc::ChainStormV1)(t, Xt, aas, chainids, resinds, disto_gram, Xtprev_frames, delta_ts, temps; Xt_2 = nothing, delta_ts_2 = nothing, sc_frames = nothing)#, sc_frames_2 = nothing)

    
    l = fc.layers
    delta_ts = 2f6 .* delta_ts
    temps = 2f-3 .* temps
    pmask = Flux.Zygote.@ignore self_att_padding_mask(Xt[1].lmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))

    pair_feats = l.pair_project(pre_z) + l.disto_project(disto_gram)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    deltat_rff = Flux.Zygote.@ignore l.t_rff(delta_ts)
    temp_rff = Flux.Zygote.@ignore l.t_rff(temps)
    cond = reshape(l.cond_t_encoding(t_rff)+l.cond_delta_t_encoding(deltat_rff)+l.cond_temp_encoding(temp_rff), :, 1, size(t,2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))
    AA_one_hots = tensor(Flux.onehotbatch(aas, 1:21))
    #AA_one_hots = tensor(Xt[3])

    x = l.AAencoder(AA_one_hots .+ 0)
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, pmask)
            f1, f2 = mod(i, 2) == 0 ? (frames, sc_frames) : (sc_frames, frames)
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], f1, f2, x, pair_feats, cond, pmask)
        end

        x = Flux.Zygote.checkpointed(crossipa, l.prev_selfipa[i], Xtprev_frames, Xtprev_frames, x, pair_feats, cond, pmask)
        f1, f2 = mod(floor(i/2), 2) == 0 ? (frames, Xtprev_frames) : (Xtprev_frames, frames)
        x = Flux.Zygote.checkpointed(crossipa, l.prev_crossipa[i], f1, f2, x, pair_feats, cond, pmask)

        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = t)
        end
    end
    #aa_logits = l.AAdecoder(x .+ reshape(l.AApre_t_encoding(t_rff), :, 1, size(t,2)))   
    #return frames, aa_logits
    if (Xt_2 === nothing) != (delta_ts_2 === nothing)
      error("Xt_2 and delta_ts_2 must be provided together")
    end

    if Xt_2 !== nothing && delta_ts_2 !== nothing 
        delta_ts_2 = 2f6 .* delta_ts_2

        pair_feats_2 = l.pair_project_2(pre_z) #+ l.disto_project_2(disto_gram)
        #t_rff = Flux.Zygote.@ignore l.t_rff(t)
        deltat_rff_2 = Flux.Zygote.@ignore l.t_rff(delta_ts_2)
        # temp_rff = Flux.Zygote.@ignore l.t_rff(temps)
        cond_2 = reshape(l.cond_t_encoding_2(t_rff)+l.cond_delta_t_encoding_2(deltat_rff_2)+l.cond_temp_encoding_2(temp_rff), :, 1, size(t,2))
        frames_2 = Translation(tensor(Xt_2[1])) ∘ Rotation(tensor(Xt_2[2]))
        #AA_one_hots_2 = tensor(Flux.onehotbatch(aas, 1:21))
        #AA_one_hots = tensor(Xt[3])

        x_2 = l.AAencoder_2(AA_one_hots .+ 0)
        for i in 1:l.depth
            # if sc_frames_2 !== nothing
            #     x_2 = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa_2[i], sc_frames_2, sc_frames_2, x_2, pair_feats_2, cond_2, pmask)
            #     f1, f2 = mod(i, 2) == 0 ? (frames_2, sc_frames_2) : (sc_frames_2, frames_2)
            #     x_2 = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa_2[i], f1, f2, x_2, pair_feats_2, cond_2, pmask)
            # end

            x_2 = Flux.Zygote.checkpointed(crossipa, l.modelpred_selfipa[i], frames, frames, x_2, pair_feats_2, cond_2, pmask)
            f1, f2 = mod(floor(i/2), 2) == 0 ? (frames_2, frames) : (frames, frames_2)
            x_2 = Flux.Zygote.checkpointed(crossipa, l.modelpred_crossipa[i], f1, f2, x_2, pair_feats_2, cond_2, pmask)

            x_2 = Flux.Zygote.checkpointed(ipa, l.ipa_blocks_2[i], frames_2, x_2, pair_feats_2, cond_2, pmask)
            if i > l.depth - l.f_depth
                frames_2 = l.framemovers_2[i - l.depth + l.f_depth](frames_2, x_2, t = t)
            end
        end

        return frames, frames_2
    else
        return frames
    end
end