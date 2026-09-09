/* Atmosphere path tracer: four-way min(t_gs, t_mesh, t_vox, t_gas).
 * Volume math matches htrdr hit_filter (Woodcock + 1D gas). GS/mesh are
 * surfaces only — they never contribute cloud k. */

#pragma once

#include <3dgptir/kernels/cuda/atmosphere.cuh>
#include <3dgptir/kernels/cuda/pathTracer.cuh>
#include <3dgptir/pipelineParameters.h>

static constexpr float kAtmSurfEps = 2e-3f;

static __device__ __forceinline__ void atm_f3_to_arr(const float3& v, float o[3]) {
    o[0] = v.x;
    o[1] = v.y;
    o[2] = v.z;
}

static __device__ __forceinline__ float3 atm_arr_to_f3(const float a[3]) {
    return make_float3(a[0], a[1], a[2]);
}

struct AtmAccelHit {
    float t;
    int kind; /* 0 miss, 1 mesh, 2 vol */
    int prim;
    int event;
    float temperature;
};

static constexpr int kAtmMeshSlabMaxSteps = 32;
static constexpr int kAtmProfSlots = 26;

static __device__ int atm_mesh_repeat_ok() {
    return (params.meshRepeatUpp[0] - params.meshRepeatLow[0] > 1.f) &&
           (params.meshRepeatUpp[1] - params.meshRepeatLow[1] > 1.f);
}

/* Z-slab of the -R mesh tile. 0 = ray never hits terrain AABB. */
static __device__ int atm_mesh_z_interval(
    const float org[3],
    const float dir[3],
    float tmax,
    float* t_enter,
    float* t_exit) {
    const float z0 = params.meshRepeatLow[2];
    const float z1 = params.meshRepeatUpp[2];
    *t_enter = 0.f;
    *t_exit = tmax;
    if (z1 - z0 < 1.f) {
        return 1;
    }
    if (fabsf(dir[2]) < 1e-20f) {
        if (org[2] < z0 || org[2] > z1) {
            return 0;
        }
        return 1;
    }
    float ta = (z0 - org[2]) / dir[2];
    float tb = (z1 - org[2]) / dir[2];
    if (ta > tb) {
        const float tmp = ta;
        ta = tb;
        tb = tmp;
    }
    const float t0 = fmaxf(ta, 0.f);
    const float t1 = fminf(tb, tmax);
    if (t0 >= t1) {
        return 0;
    }
    *t_enter = t0;
    *t_exit = t1;
    return 1;
}

static __device__ void atm_ser_bin_paths(int doMesh, int inVolume) {
#if OPTIX_VERSION >= 80000
    if (params.atmSerReorder) {
        /* 1 = mesh (打山), 2 = volume (透云), 3 = neither (不打山) */
        const unsigned key = doMesh ? 1u : (inVolume ? 2u : 3u);
        optixReorder(key, 2u);
    }
#else
    (void)doMesh;
    (void)inVolume;
#endif
}

static __device__ AtmAccelHit atm_accel_miss() {
    AtmAccelHit hit;
    hit.t = ATM_INF;
    hit.kind = 0;
    hit.prim = -1;
    hit.event = ATM_EVENT_NONE;
    hit.temperature = 0.f;
    return hit;
}

/* Slim mesh closest-hit: t, kind, prim. No PTIR 32-payload continuation. */
static __device__ AtmAccelHit atm_trace_mesh(
    OptixTraversableHandle handle,
    const float org[3],
    const float dir[3],
    float tmin,
    float tmax,
    unsigned rayFlags) {
    AtmAccelHit hit = atm_accel_miss();
    if (!handle || tmax <= tmin) {
        return hit;
    }
    unsigned int p0 = __float_as_uint(ATM_INF);
    unsigned int p1 = 0;
    unsigned int p2 = 0xFFFFFFFFu;
    optixTrace(
        handle,
        make_float3(org[0], org[1], org[2]),
        make_float3(dir[0], dir[1], dir[2]),
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        rayFlags | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        ATM_SBT_MESH,
        1,
        0,
        p0,
        p1,
        p2);
    if (p1 == 1u) {
        hit.t = __uint_as_float(p0);
        hit.kind = 1;
        hit.prim = static_cast<int>(p2);
    }
    return hit;
}

/* Volume IS+CH: t, kind, prim, event, temp, plus RNG seed in p5/p6. */
static __device__ AtmAccelHit atm_trace_vol(
    OptixTraversableHandle handle,
    const float org[3],
    const float dir[3],
    float tmin,
    float tmax,
    Sampler& sampler) {
    AtmAccelHit hit = atm_accel_miss();
    if (!handle || tmax <= tmin) {
        return hit;
    }
    unsigned int p0 = __float_as_uint(ATM_INF);
    unsigned int p1 = 0;
    unsigned int p2 = 0xFFFFFFFFu;
    unsigned int p3 = 0;
    unsigned int p4 = 0;
    unsigned int p5 = static_cast<unsigned int>(sampler.state);
    unsigned int p6 = static_cast<unsigned int>(sampler.state >> 32ull);
    optixTrace(
        handle,
        make_float3(org[0], org[1], org[2]),
        make_float3(dir[0], dir[1], dir[2]),
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        0u,
        ATM_SBT_VOL,
        1,
        0,
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6);
    if (p1 == 2u) {
        hit.t = __uint_as_float(p0);
        hit.kind = 2;
        hit.prim = static_cast<int>(p2);
        hit.event = static_cast<int>(p3);
        hit.temperature = __uint_as_float(p4);
    }
    return hit;
}

/* htrdr_slab_trace_ray / -R: fold origin into the mesh tile, OptiX the GAS,
 * miss → step to the next XY neighbour (max 32). Cloud period is voxels, not this AABB. */
static __device__ AtmAccelHit atm_trace_mesh_slab(
    const float org[3],
    const float dir[3],
    float tmin,
    float tmax,
    unsigned rayFlags,
    Sampler& sampler) {
    (void)sampler;
    AtmAccelHit miss;
    miss.t = ATM_INF;
    miss.kind = 0;
    miss.prim = -1;
    miss.event = ATM_EVENT_NONE;
    miss.temperature = 0.f;
    if (!params.meshHandle || tmax <= tmin) {
        return miss;
    }
    if (!atm_mesh_repeat_ok()) {
        return atm_trace_mesh(params.meshHandle, org, dir, tmin, tmax, rayFlags);
    }

    const float lo[3] = {params.meshRepeatLow[0], params.meshRepeatLow[1], params.meshRepeatLow[2]};
    const float hi[3] = {params.meshRepeatUpp[0], params.meshRepeatUpp[1], params.meshRepeatUpp[2]};
    const float sx = hi[0] - lo[0];
    const float sy = hi[1] - lo[1];

    float t_min_z = tmin;
    float t_max_z = tmax;
    if (fabsf(dir[2]) < 1e-20f) {
        if (org[2] < lo[2] || org[2] > hi[2]) {
            return miss;
        }
    } else {
        float ta = (lo[2] - org[2]) / dir[2];
        float tb = (hi[2] - org[2]) / dir[2];
        if (ta > tb) {
            const float tmp = ta;
            ta = tb;
            tb = tmp;
        }
        t_min_z = fmaxf(ta, tmin);
        t_max_z = fminf(tb, tmax);
        if (t_min_z > t_max_z) {
            return miss;
        }
    }

    const float posx = org[0] + t_min_z * dir[0];
    const float posy = org[1] + t_min_z * dir[1];
    int xy0 = (int)floorf((posx - lo[0]) / sx);
    int xy1 = (int)floorf((posy - lo[1]) / sy);
    const int incr0 = dir[0] < 0.f ? -1 : 1;
    const int incr1 = dir[1] < 0.f ? -1 : 1;

    float cell0 = lo[0] + (float)xy0 * sx;
    float cell1 = lo[1] + (float)xy1 * sy;
    float t_max0 = (fabsf(dir[0]) < 1e-20f)
                       ? ATM_INF
                       : (((dir[0] < 0.f ? cell0 : cell0 + sx) - org[0]) / dir[0]);
    float t_max1 = (fabsf(dir[1]) < 1e-20f)
                       ? ATM_INF
                       : (((dir[1] < 0.f ? cell1 : cell1 + sy) - org[1]) / dir[1]);
    const float t_max2 = t_max_z;
    const float t_delta0 = (fabsf(dir[0]) < 1e-20f) ? ATM_INF : ((incr0 < 0 ? -sx : sx) / dir[0]);
    const float t_delta1 = (fabsf(dir[1]) < 1e-20f) ? ATM_INF : ((incr1 < 0 ? -sy : sy) / dir[1]);

    /* Clip each OptiX query to this XY cell ∩ Z-slab. Unclipped tmax=path_tmax
     * made upward / grazing misses walk the 8.3M GAS up to 32 times. */
    float t_prev = t_min_z;
    for (int istep = 0; istep < kAtmMeshSlabMaxSteps; ++istep) {
        const float t_cell = fminf(fminf(t_max0, t_max1), t_max_z);
        const float qmin = fmaxf(tmin, t_prev);
        const float qmax = fminf(tmax, t_cell);
        if (qmax > qmin) {
            const float org_cs[3] = {
                org[0] - (float)xy0 * sx,
                org[1] - (float)xy1 * sy,
                org[2]};
            const AtmAccelHit h = atm_trace_mesh(
                params.meshHandle, org_cs, dir, qmin, qmax, rayFlags);
            if (h.kind == 1 && h.t <= qmax && h.t >= qmin) {
                return h;
            }
        }

        const int iaxis = (t_max0 < t_max1) ? ((t_max0 < t_max2) ? 0 : 2)
                                            : ((t_max1 < t_max2) ? 1 : 2);
        if (iaxis == 2) {
            break;
        }
        if (iaxis == 0) {
            if (t_max0 >= tmax) {
                break;
            }
            t_max0 += t_delta0;
            xy0 += incr0;
        } else {
            if (t_max1 >= tmax) {
                break;
            }
            t_max1 += t_delta1;
            xy1 += incr1;
        }
        t_prev = t_cell;
    }
    return miss;
}

static __device__ struct AtmMeshHit atm_mesh_closest_slab(
    const struct AtmMesh* mesh,
    const float org[3],
    const float dir[3],
    float tmax) {
    if (!mesh || mesh->ntri <= 0) {
        struct AtmMeshHit none;
        atm_mesh_hit_clear(&none);
        return none;
    }
    if (!atm_mesh_repeat_ok()) {
        return atm_mesh_closest_hit(mesh, org, dir, tmax);
    }

    const float lo[3] = {params.meshRepeatLow[0], params.meshRepeatLow[1], params.meshRepeatLow[2]};
    const float hi[3] = {params.meshRepeatUpp[0], params.meshRepeatUpp[1], params.meshRepeatUpp[2]};
    const float sx = hi[0] - lo[0];
    const float sy = hi[1] - lo[1];
    struct AtmMeshHit miss;
    atm_mesh_hit_clear(&miss);

    float t_min_z = 0.f;
    float t_max_z = tmax;
    if (fabsf(dir[2]) < 1e-20f) {
        if (org[2] < lo[2] || org[2] > hi[2]) {
            return miss;
        }
    } else {
        float ta = (lo[2] - org[2]) / dir[2];
        float tb = (hi[2] - org[2]) / dir[2];
        if (ta > tb) {
            const float tmp = ta;
            ta = tb;
            tb = tmp;
        }
        t_min_z = fmaxf(ta, 0.f);
        t_max_z = fminf(tb, tmax);
        if (t_min_z > t_max_z) {
            return miss;
        }
    }

    const float posx = org[0] + t_min_z * dir[0];
    const float posy = org[1] + t_min_z * dir[1];
    int xy0 = (int)floorf((posx - lo[0]) / sx);
    int xy1 = (int)floorf((posy - lo[1]) / sy);
    const int incr0 = dir[0] < 0.f ? -1 : 1;
    const int incr1 = dir[1] < 0.f ? -1 : 1;
    float cell0 = lo[0] + (float)xy0 * sx;
    float cell1 = lo[1] + (float)xy1 * sy;
    float t_max0 = (fabsf(dir[0]) < 1e-20f)
                       ? ATM_INF
                       : (((dir[0] < 0.f ? cell0 : cell0 + sx) - org[0]) / dir[0]);
    float t_max1 = (fabsf(dir[1]) < 1e-20f)
                       ? ATM_INF
                       : (((dir[1] < 0.f ? cell1 : cell1 + sy) - org[1]) / dir[1]);
    const float t_max2 = t_max_z;
    const float t_delta0 = (fabsf(dir[0]) < 1e-20f) ? ATM_INF : ((incr0 < 0 ? -sx : sx) / dir[0]);
    const float t_delta1 = (fabsf(dir[1]) < 1e-20f) ? ATM_INF : ((incr1 < 0 ? -sy : sy) / dir[1]);

    float t_prev = t_min_z;
    for (int istep = 0; istep < kAtmMeshSlabMaxSteps; ++istep) {
        const float t_cell = fminf(fminf(t_max0, t_max1), t_max_z);
        const float qmax = fminf(tmax, t_cell);
        if (qmax > t_prev) {
            const float org_cs[3] = {
                org[0] - (float)xy0 * sx,
                org[1] - (float)xy1 * sy,
                org[2]};
            struct AtmMeshHit h = atm_mesh_closest_hit(mesh, org_cs, dir, qmax);
            if (h.t < qmax && h.t >= t_prev) {
                return h;
            }
        }
        const int iaxis = (t_max0 < t_max1) ? ((t_max0 < t_max2) ? 0 : 2)
                                            : ((t_max1 < t_max2) ? 1 : 2);
        if (iaxis == 2) {
            break;
        }
        if (iaxis == 0) {
            if (t_max0 >= tmax) {
                break;
            }
            t_max0 += t_delta0;
            xy0 += incr0;
        } else {
            if (t_max1 >= tmax) {
                break;
            }
            t_max1 += t_delta1;
            xy1 += incr1;
        }
        t_prev = t_cell;
    }
    return miss;
}

static __device__ void atm_mesh_from_accel(
    const struct AtmMesh* mesh,
    const AtmAccelHit& a,
    const float dir[3],
    struct AtmMeshHit* hit) {
    atm_mesh_hit_clear(hit);
    if (a.kind != 1 || a.prim < 0) {
        return;
    }
    hit->t = a.t;
    hit->tri = a.prim;
    hit->albedo[0] = 0.22f;
    hit->albedo[1] = 0.16f;
    hit->albedo[2] = 0.10f;
    hit->n[0] = 0.f;
    hit->n[1] = 0.f;
    hit->n[2] = 1.f;
    hit->temperature = 300.f;
    if (mesh && mesh->indices && mesh->vertices && mesh->ntri > 0 && a.prim < mesh->ntri) {
        atm_mesh_fill_hit(mesh, a.prim, a.t, hit);
        if (atm_dot3(hit->n, dir) > 0.f) {
            hit->n[0] = -hit->n[0];
            hit->n[1] = -hit->n[1];
            hit->n[2] = -hit->n[2];
        }
    }
}

static __device__ void atm_vol_from_accel(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    const AtmAccelHit& a,
    struct AtmHit* hit) {
    atm_hit_clear(hit);
    if (!hit || a.kind != 2 || a.event == ATM_EVENT_NONE) {
        return;
    }
    hit->t = a.t;
    hit->event = a.event;
    hit->pos[0] = org[0] + dir[0] * a.t;
    hit->pos[1] = org[1] + dir[1] * a.t;
    hit->pos[2] = org[2] + dir[2] * a.t;
    int ijk[3];
    atm_world_to_ijk(&atm->voxels, hit->pos, ijk, atm->voxel_z_iface);
    hit->ka = atm_fetch_ka(&atm->voxels, ijk[0], ijk[1], ijk[2]);
    hit->ks = atm_fetch_ks(&atm->voxels, ijk[0], ijk[1], ijk[2]);
    hit->kext = hit->ka + hit->ks;
    hit->temperature = a.temperature > 0.f ? a.temperature : atm_fetch_temp(&atm->voxels, ijk[0], ijk[1], ijk[2]);
}

static __device__ void atm_merge_gas(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tmax,
    Sampler& sampler,
    const struct AtmHit& vox,
    struct AtmHit* out) {
    const int walk_ks = atm->spectral_mode != ATM_SPECTRAL_LW;
    struct AtmHit gas;
    atm_gas_free_path_into(&atm->gas, org, dir, tmax, sampler, walk_ks, &gas);
    if (gas.event != ATM_EVENT_NONE && gas.t <= vox.t && gas.t < tmax) {
        *out = gas;
        return;
    }
    if (vox.event != ATM_EVENT_NONE && vox.t < tmax) {
        *out = vox;
        return;
    }
    atm_hit_clear(out);
    out->t = tmax;
}

/* First GS surface along the ray. Stochastic α is the object model, not cloud k. */
static __device__ float atm_gs_closest_surface(
    const Ray& ray,
    float tmin,
    float tmax,
    Sampler& sampler,
    Interaction& outIx) {
    outIx = Interaction();
    if (!params.atmosphere.gs_enabled || params.handle == 0 || params.particleDensity == nullptr) {
        return ATM_INF;
    }

    constexpr float epsT = 1e-9f;
    const float2 minMaxT = intersectAABB(params.aabb, ray);
    float startT = fmaxf(tmin, minMaxT.x - epsT);
    const float endT = fminf(tmax, minMaxT.y) + epsT;
    if (startT >= endT) {
        return ATM_INF;
    }

    RayPayload hitPayload;
    while (startT < endT) {
        trace(hitPayload, ray, startT + epsT, endT);
        if (hitPayload[0].particleId == RayHit::InvalidParticleId) {
            break;
        }

        float batchEndT = startT;
#pragma unroll
        for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; ++i) {
            const RayHit rayHit = hitPayload[i];
            if (rayHit.particleId == RayHit::InvalidParticleId) {
                break;
            }
            if (rayHit.distance > endT) {
                break;
            }
            batchEndT = fmaxf(batchEndT, rayHit.distance);
            if (isSelfOcclusionHit(rayHit, ray, params.particleShadingNormal)) {
                continue;
            }
            const float alpha = shadowHitAlpha<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
                ray, rayHit.particleId);
            if (alpha > 0.0f && sampler.next_1d() < alpha) {
                float3 particlePosition;
                float3 particleScale;
                float33 particleRotation;
                float particleDensity;
                fetchParticleDensity(
                    rayHit.particleId,
                    params.particleDensity,
                    particlePosition,
                    particleScale,
                    particleRotation,
                    particleDensity);
                float3 albedo;
                float roughness, metallic;
                fetchParticleMaterial(rayHit.particleId, params.particleMaterial, albedo, roughness, metallic);
                float3 n = make_float3(0.f, 0.f, 1.f);
                if (params.particleShadingNormal) {
                    n = make_float3(
                        params.particleShadingNormal[rayHit.particleId * 3 + 0],
                        params.particleShadingNormal[rayHit.particleId * 3 + 1],
                        params.particleShadingNormal[rayHit.particleId * 3 + 2]);
                }
                if (dot(n, n) < 1e-12f) {
                    n = -ray.direction;
                }
                if (dot(n, ray.direction) > 0.f) {
                    n = -n;
                }
                const float3 pos = ray.origin + ray.direction * rayHit.distance;
                outIx = Interaction(pos, n, Material(albedo, roughness, metallic), true);
                params.particleVisibility[rayHit.particleId] = 1;
                return rayHit.distance;
            }
        }
        if (batchEndT <= startT) {
            break;
        }
        startT = batchEndT;
    }
    return ATM_INF;
}

static __device__ __noinline__ void atm_apply_gas_after_vol(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tmax,
    Sampler& sampler,
    struct AtmHit* vol) {
    if (vol->event == ATM_EVENT_NONE) {
        /* IS missed the volume AABB: still walk gas + Z-slab voxels. */
        atm_next_volume_event_into(atm, org, dir, tmax, sampler, vol);
    }
}

static __device__ int atm_ground_occluded(
    const float org[3],
    const float dir[3],
    float tmax) {
    /* htrdr atmosphere_ground_trace_ray: triangles only. Volume NEE must not
     * use GS shadows (sun from cloud is upward; GS false hits zero scatter). */
    float meshEnter = 0.f;
    float meshExit = tmax;
    if (!atm_mesh_z_interval(org, dir, tmax, &meshEnter, &meshExit)) {
        return 0;
    }
    if (params.meshHandle) {
        Sampler rng;
        rng.init(1ull, 1ull);
        const AtmAccelHit h = atm_trace_mesh_slab(
            org,
            dir,
            fmaxf(2.f, meshEnter),
            fminf(tmax, meshExit),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            rng);
        return (h.kind == 1 && h.t < tmax && h.t > 2.f) ? 1 : 0;
    }
    if (params.atmosphere.mesh.ntri > 0) {
        const struct AtmMeshHit mh = atm_mesh_closest_slab(&params.atmosphere.mesh, org, dir, tmax);
        if (mh.t < tmax && mh.t > 2.f) {
            return 1;
        }
    }
    return 0;
}

static __device__ int atm_surface_occluded(
    const float org[3],
    const float dir[3],
    float tmax,
    Sampler& sampler) {
    if (atm_ground_occluded(org, dir, tmax)) {
        return 1;
    }
    if (params.atmosphere.gs_enabled) {
        const Ray ray(make_float3(org[0], org[1], org[2]), make_float3(dir[0], dir[1], dir[2]));
        return traceShadowMonteCarloOccluded<true>(ray, sampler, 0.0f, tmax) ? 1 : 0;
    }
    return 0;
}

/* Surface NEE: GS+mesh then analog exp(-∫Kext). */
static __device__ float atm_sun_visibility(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tmax,
    Sampler& sampler) {
    if (atm_surface_occluded(org, dir, tmax, sampler)) {
        return 0.f;
    }
    return atm_volume_transmittance(atm, org, dir, tmax);
}

/* Volume NEE: mesh only, then analog exp(-∫Kext). Do not clip tmax at z=850. */
static __device__ float atm_sun_visibility_volume(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tmax) {
    if (atm_ground_occluded(org, dir, tmax)) {
        return 0.f;
    }
    return atm_volume_transmittance(atm, org, dir, tmax);
}

static __device__ int atm_lambert_bounce(
    const Interaction& ix,
    Sampler& sampler,
    float3& throughput,
    float org[3],
    float dir[3]) {
    const float R = fmaxf(ix.material.albedo.x, fmaxf(ix.material.albedo.y, ix.material.albedo.z));
    if (R <= 1e-8f) {
        return 0;
    }
    float n[3] = {ix.shadingnormal.x, ix.shadingnormal.y, ix.shadingnormal.z};
    atm_normalize3(n);
    float wo[3];
    const float u1 = sampler.next_1d();
    const float u2 = sampler.next_1d();
    const float r = sqrtf(fmaxf(u1, 0.f));
    const float phi = 6.28318530718f * u2;
    const float z = sqrtf(fmaxf(1.f - u1, 0.f));
    float T[3], B[3];
    if (fabsf(n[2]) < 0.999f) {
        T[0] = n[1];
        T[1] = -n[0];
        T[2] = 0.f;
    } else {
        T[0] = 1.f;
        T[1] = 0.f;
        T[2] = 0.f;
    }
    atm_normalize3(T);
    B[0] = n[1] * T[2] - n[2] * T[1];
    B[1] = n[2] * T[0] - n[0] * T[2];
    B[2] = n[0] * T[1] - n[1] * T[0];
    const float cp = cosf(phi);
    const float sp = sinf(phi);
    wo[0] = r * cp * T[0] + r * sp * B[0] + z * n[0];
    wo[1] = r * cp * T[1] + r * sp * B[1] + z * n[1];
    wo[2] = r * cp * T[2] + r * sp * B[2] + z * n[2];
    atm_normalize3(wo);
    throughput = throughput * R;
    org[0] = ix.position.x + n[0] * 2.f + wo[0] * 0.05f;
    org[1] = ix.position.y + n[1] * 2.f + wo[1] * 0.05f;
    org[2] = ix.position.z + n[2] * 2.f + wo[2] * 0.05f;
    dir[0] = wo[0];
    dir[1] = wo[1];
    dir[2] = wo[2];
    return 1;
}

static __device__ void atm_add_sun_nee_surface(
    const struct AtmParams* atm,
    const float org[3],
    const float3& rayDir,
    const Interaction& ix,
    const float3& throughput,
    Sampler& sampler,
    float3& radiance,
    float3& direct) {
    if (atm->spectral_mode == ATM_SPECTRAL_LW || atm->sun.radiance <= 0.f) {
        return;
    }
    float wo[3];
    float pdf = 1.f;
    atm_sample_sun(&atm->sun, sampler, wo, &pdf);
    const float3 L = atm_arr_to_f3(wo);
    if (dot(ix.shadingnormal, L) <= 0.f) {
        return;
    }
    const float ndl = fmaxf(0.f, ix.shadingnormal.x * L.x + ix.shadingnormal.y * L.y + ix.shadingnormal.z * L.z);
    float3 fcos;
    if (atm->path_albedo >= 0.f) {
        /* htrdr mrumtl lambertian: R/π * n·L. Disney F0=0.04 is not sand. */
        fcos = ix.material.albedo * (ndl * (1.f / 3.14159265f));
    } else {
        const float3 lambert = ix.material.albedo * (ndl * (1.f / 3.14159265f));
        float scatterPdf = 0.f;
        const float3 brdfTimesCos = eval_material_fast_brdf_light_sample(rayDir, ix, L, scatterPdf);
        (void)scatterPdf;
        fcos = (brdfTimesCos.x + brdfTimesCos.y + brdfTimesCos.z) > 1e-8f ? brdfTimesCos : lambert;
    }
    /* Offset along shading normal so 8M heightfield NEE does not self-hit. */
    float p[3] = {
        org[0] + ix.shadingnormal.x * 2.f + wo[0] * 0.05f,
        org[1] + ix.shadingnormal.y * 2.f + wo[1] * 0.05f,
        org[2] + ix.shadingnormal.z * 2.f + wo[2] * 0.05f};
    const float vis = atm_sun_visibility(atm, p, wo, atm->path_tmax, sampler);
    if (vis <= 0.f) {
        return;
    }
    const float3 contrib = throughput * fcos * (atm->sun.radiance * vis / fmaxf(pdf, 1e-12f));
    radiance += contrib;
    direct += contrib;
}

static __device__ void atm_add_sky_surface(
    const struct AtmParams* atm,
    const Interaction& ix,
    const float3& throughput,
    float3& radiance,
    float3& direct) {
    if (atm->spectral_mode == ATM_SPECTRAL_LW || atm->sun.radiance <= 0.f) {
        return;
    }
    /* htrdr has no constant skylight on surfaces. 0.08*sun was for mono-550. */
    if (atm->cie_enabled) {
        return;
    }
    const float sky = atm->sun.radiance * 0.08f;
    const float3 contrib = throughput * ix.material.albedo * (sky * (1.f / 3.14159265f));
    radiance += contrib;
    direct += contrib;
}

static __device__ void atm_add_sun_nee_volume(
    const struct AtmParams* atm,
    const float org[3],
    const float wi[3],
    int voxelScatter,
    const float3& throughput,
    Sampler& sampler,
    float3& radiance,
    float3& direct) {
    if (atm->spectral_mode == ATM_SPECTRAL_LW || atm->sun.radiance <= 0.f) {
        return;
    }
    float wo[3];
    float pdf = 1.f;
    atm_sample_sun(&atm->sun, sampler, wo, &pdf);
    /* htrdr: wo_ssf = -dir (toward camera); ssf_phase_eval then uses
     * cos = dot(-wo_ssf, sun) = dir · sun. Keep mu = path_dir · sun. */
    const float mu = atm_dot3(wi, wo);
    const float phase = voxelScatter ? atm_hg_phase(atm->mie_g, mu) : atm_rayleigh_phase(mu);
    float p[3] = {org[0] + wo[0] * kAtmSurfEps, org[1] + wo[1] * kAtmSurfEps, org[2] + wo[2] * kAtmSurfEps};
    const float vis = atm_sun_visibility_volume(atm, p, wo, atm->path_tmax);
    if (vis <= 0.f) {
        return;
    }
    const float3 contrib = throughput * (phase * atm->sun.radiance * vis / fmaxf(pdf, 1e-12f));
    radiance += contrib;
    direct += contrib;
}

static __device__ void writeAtmosphereOutputs(
    const uint3& idx,
    const float3& radiance,
    const float3& direct,
    const float3& indirect,
    float hitT,
    int escaped,
    int firstEvent) {
    params.rayRadiance[idx.z][idx.y][idx.x][0] = radiance.x;
    params.rayRadiance[idx.z][idx.y][idx.x][1] = radiance.y;
    params.rayRadiance[idx.z][idx.y][idx.x][2] = radiance.z;
    params.rayPbr[idx.z][idx.y][idx.x][0] = radiance.x;
    params.rayPbr[idx.z][idx.y][idx.x][1] = radiance.y;
    params.rayPbr[idx.z][idx.y][idx.x][2] = radiance.z;
    params.rayLight[idx.z][idx.y][idx.x][0] = direct.x;
    params.rayLight[idx.z][idx.y][idx.x][1] = direct.y;
    params.rayLight[idx.z][idx.y][idx.x][2] = direct.z;
    params.rayPbrComponents[idx.z][idx.y][idx.x][0][0] = direct.x;
    params.rayPbrComponents[idx.z][idx.y][idx.x][0][1] = direct.y;
    params.rayPbrComponents[idx.z][idx.y][idx.x][0][2] = direct.z;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][0] = indirect.x;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][1] = indirect.y;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][2] = indirect.z;
    params.rayDensity[idx.z][idx.y][idx.x][0] = escaped ? 0.f : 1.f;
    params.rayHitDistance[idx.z][idx.y][idx.x][0] = hitT;
    params.rayHitDistance[idx.z][idx.y][idx.x][1] = hitT;
    params.rayHitDistanceSecondMoment[idx.z][idx.y][idx.x][0] = hitT * hitT;
    params.rayDepthDistortion[idx.z][idx.y][idx.x][0] = 0.f;
    params.rayNormal[idx.z][idx.y][idx.x][0] = 0.f;
    params.rayNormal[idx.z][idx.y][idx.x][1] = 0.f;
    params.rayNormal[idx.z][idx.y][idx.x][2] = 0.f;
    params.rayShadingNormal[idx.z][idx.y][idx.x][0] = 0.f;
    params.rayShadingNormal[idx.z][idx.y][idx.x][1] = 0.f;
    params.rayShadingNormal[idx.z][idx.y][idx.x][2] = 0.f;
    params.rayMaterial[idx.z][idx.y][idx.x][0] = 0.f;
    params.rayMaterial[idx.z][idx.y][idx.x][1] = 0.f;
    params.rayMaterial[idx.z][idx.y][idx.x][2] = 0.f;
    params.rayMaterial[idx.z][idx.y][idx.x][3] = 0.f;
    params.rayMaterial[idx.z][idx.y][idx.x][4] = 0.f;
    params.rayHitsCount[idx.z][idx.y][idx.x][0] = static_cast<float>(firstEvent);
}

static __device__ __forceinline__ unsigned long long atm_cycles_since(long long t0) {
    return static_cast<unsigned long long>(clock64() - t0);
}

static __device__ void atm_next_volume_event_into_timed(
    const AtmParams* atm,
    const float org[3],
    const float dir[3],
    float t_surface,
    Sampler& sampler,
    struct AtmHit* out,
    unsigned long long* cGas,
    unsigned long long* cWc) {
    atm_hit_clear(out);
    const float tmax = t_surface < atm->path_tmax ? t_surface : atm->path_tmax;
    out->t = tmax;
    const int walk_ks = atm->spectral_mode != ATM_SPECTRAL_LW;
    const float omega_arg = atm->mie_omega > 0.f ? atm->mie_omega : -1.f;
    struct AtmHit gas;
    struct AtmHit vox;
    long long t0 = clock64();
    atm_gas_free_path_into(&atm->gas, org, dir, tmax, sampler, walk_ks, &gas);
    *cGas += atm_cycles_since(t0);
    t0 = clock64();
    atm_woodcock_voxels_into(
        &atm->voxels,
        org,
        dir,
        tmax,
        sampler,
        omega_arg,
        walk_ks,
        &vox,
        atm_path_ks_scale(atm),
        atm_path_ka_scale(atm),
        atm->voxel_z_iface);
    *cWc += atm_cycles_since(t0);
    if (gas.event != ATM_EVENT_NONE && gas.t <= vox.t && gas.t < tmax) {
        *out = gas;
        return;
    }
    if (vox.event != ATM_EVENT_NONE && vox.t < tmax) {
        *out = vox;
        return;
    }
}

static __device__ void atm_run_volume_event(
    const AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tClip,
    Sampler& sampler,
    int cieOn,
    int prof,
    struct AtmHit* vol,
    unsigned long long* cGas,
    unsigned long long* cWc) {
    if (params.volHandle && !cieOn) {
        const AtmAccelHit a = atm_trace_vol(
            params.volHandle, org, dir, 1e-4f, tClip, sampler);
        atm_vol_from_accel(atm, org, dir, a, vol);
        atm_apply_gas_after_vol(atm, org, dir, tClip, sampler, vol);
    } else if (prof) {
        atm_next_volume_event_into_timed(atm, org, dir, tClip, sampler, vol, cGas, cWc);
    } else {
        atm_next_volume_event_into(atm, org, dir, tClip, sampler, vol);
    }
}

static __device__ void traceAtmospherePath(const uint3& idx, Sampler& sampler) {
    const AtmParams* atm_base = &params.atmosphere;
    Ray ray(params.rayWorldOrigin(idx), params.rayWorldDirection(idx));

    float org0[3], dir0[3];
    atm_f3_to_arr(ray.origin, org0);
    atm_f3_to_arr(ray.direction, dir0);
    atm_normalize3(dir0);

    const int cieOn = (atm_base->cie_enabled && atm_base->cie_cdf && atm_base->cie_nbins > 0) ? 1 : 0;
    const int cieSplit = (cieOn && atm_base->cie_split_launch) ? 1 : 0;
    const int cieSame = (cieOn && !cieSplit) ? 1 : 0;
    int cie_ch0 = 0;
    int nch = 1;
    if (cieSplit) {
        unsigned filmZ = idx.z;
        if (params.raySeedFromBuffer) {
            filmZ = static_cast<unsigned int>(params.raySeedIdx[idx.z][idx.y][idx.x][2]);
        }
        unsigned spp = 0u;
        if (atm_base->cie_spp > 0) {
            spp = static_cast<unsigned int>(atm_base->cie_spp);
        } else {
            const uint3 launchDim = optixGetLaunchDimensions();
            spp = launchDim.z / 3u;
        }
        cie_ch0 = (spp > 0u) ? static_cast<int>(filmZ / spp) : 0;
        if (cie_ch0 < 0) {
            cie_ch0 = 0;
        }
        if (cie_ch0 > 2) {
            cie_ch0 = 2;
        }
    }
    float xyz[3] = {0.f, 0.f, 0.f};
    float firstHitTKeep = 0.f;
    int firstEventKeep = ATM_EVENT_ESCAPE;
    int escapedKeep = 1;
    float xyzFirst[3] = {0.f, 0.f, 0.f};

    const int maxB = atm_base->max_bounces > 0 ? atm_base->max_bounces : (int)params.maxBounces;
    const float tmax0 = atm_base->path_tmax > 0.f ? atm_base->path_tmax : 1.0e5f;
    const int isLW = atm_base->spectral_mode == ATM_SPECTRAL_LW;

    float3 radianceOut = make_float3(0.f, 0.f, 0.f);
    float3 directOut = make_float3(0.f, 0.f, 0.f);
    float3 indirectOut = make_float3(0.f, 0.f, 0.f);
    unsigned long long cMesh = 0ull;
    unsigned long long cWc = 0ull;
    unsigned long long cGas = 0ull;
    unsigned long long cNee = 0ull;
    unsigned long long nCamT = 0ull, nCamHg = 0ull, nCamHe = 0ull, cCam = 0ull;
    unsigned long long nPostT = 0ull, nPostHg = 0ull, nPostHe = 0ull, cPost = 0ull;
    unsigned long long nCdT = 0ull, nCdHg = 0ull, nCdHe = 0ull, cCd = 0ull;
    unsigned long long nSurfT = 0ull, nSurfHe = 0ull, cSurf = 0ull;
    unsigned long long nOthT = 0ull, nOthHe = 0ull, cOth = 0ull;
    unsigned long long nCamDown = 0ull, nCamUp = 0ull;
    const int prof = params.atmBounceCycles != nullptr;
    long long tPath0 = 0;
    if (prof) {
        tPath0 = clock64();
    }

    for (int cie_i = 0; cie_i < nch; ++cie_i) {
    const int cie_ch = cieSplit ? cie_ch0 : cie_i;
    float org[3] = {org0[0], org0[1], org0[2]};
    float dir[3] = {dir0[0], dir0[1], dir0[2]};
    AtmParams atm_ch = *atm_base;
    float wlen_pdf = 1.f;
    if (cieOn) {
        int bind_ch = cie_ch;
        if (cieSame) {
            float r_ch = sampler.next_1d();
            bind_ch = (int)(r_ch * 3.f);
            if (bind_ch < 0) {
                bind_ch = 0;
            }
            if (bind_ch > 2) {
                bind_ch = 2;
            }
        }
        const float r0 = sampler.next_1d();
        const float r1 = sampler.next_1d();
        const float r_quad = sampler.next_1d();
        atm_cie_bind_path(&atm_ch, bind_ch, r0, r1, r_quad, &wlen_pdf);
        const float wlen_nm = atm_ch.wavelength_m * 1.0e9f;
        atm_ch.path_albedo = atm_mrumtl_fetch_R(&atm_ch, wlen_nm, sampler.next_1d());
    } else {
        atm_ch.path_albedo = -1.f;
    }
    const AtmParams* atm = &atm_ch;

    float3 throughput = make_float3(1.f, 1.f, 1.f);
    float3 radiance = make_float3(0.f, 0.f, 0.f);
    float3 direct = make_float3(0.f, 0.f, 0.f);
    float3 indirect = make_float3(0.f, 0.f, 0.f);
    float3 firstVolNee = make_float3(0.f, 0.f, 0.f);
    float firstHitT = 0.f;
    int recordedHit = 0;
    int firstEvent = ATM_EVENT_ESCAPE;
    int escaped = 1;
    int hadCloudScatter = 0;
    int hadSurfBounce = 0;

    /* htrdr SW: if the *camera* dir is in the solar cone, add L_sun * Tr(Kext)
     * before the walk. After a scatter/bounce, htrdr only uses NEE — a miss
     * does not add the disk (except specular, which CIE sand is not). */
    if (!isLW && atm_dir_in_sun_cone(&atm->sun, dir) && atm->sun.radiance > 0.f) {
        const float vis = atm_sun_visibility(atm, org, dir, tmax0, sampler);
        if (vis > 0.f) {
            const float3 contrib = throughput * (atm->sun.radiance * vis);
            radiance += contrib;
            direct += contrib;
        }
    }

    for (int bounce = 0; bounce < maxB; ++bounce) {
        const float tmax = tmax0;
        const int above_terrain = (org[2] > 850.f);
        long long tSec = 0;
        struct AtmMeshHit meshHit;
        atm_mesh_hit_clear(&meshHit);
        struct AtmHit vol;
        atm_hit_clear(&vol);
        vol.t = tmax;
        float t_surf = tmax;

        Interaction gsIx;
        const Ray gsRay(make_float3(org[0], org[1], org[2]), make_float3(dir[0], dir[1], dir[2]));
        float t_gs = ATM_INF;
        /* GS skipped above the mountain so Cu can bounce (104659). Mesh is
         * htrdr ground — always trace, including --ground mesh heightfield. */
        if (!above_terrain) {
            t_gs = atm_gs_closest_surface(gsRay, 0.f, tmax, sampler, gsIx);
        }

        /* Mesh only while the ray can still hit the terrain AABB. Upward from
         * above the tile already misses the Z-slab (no 8.3M GAS). */
        float meshEnter = 0.f;
        float meshExit = tmax;
        const int meshMaybe = atm_mesh_z_interval(org, dir, tmax, &meshEnter, &meshExit);
        const int probeVol = meshMaybe && meshEnter > 1e-3f;
        /* First ground hit keeps the 8.3M closest-hit. Later bounces skip mesh:
         * sand R≈0.16 so 2nd ground hit is ~R², and 68% of those queries miss. */
        const int allowMesh = (!above_terrain || !atm->gs_enabled) && !hadSurfBounce;
        const float meshT0 = fmaxf(1e-4f, meshEnter);
        const float meshT1 = fminf(tmax, meshExit);
        int didMesh = 0;
        unsigned long long meshCycThis = 0ull;
        int doMesh = meshMaybe && allowMesh;
        const float tProbe = fminf(tmax, meshEnter);

        if (!doMesh) {
            atm_run_volume_event(atm, org, dir, tmax, sampler, cieOn, prof, &vol, &cGas, &cWc);
            t_surf = fminf(t_gs, tmax);
        } else if (probeVol) {
            atm_run_volume_event(atm, org, dir, tProbe, sampler, cieOn, prof, &vol, &cGas, &cWc);
            if (vol.event != ATM_EVENT_NONE && vol.t < tProbe) {
                doMesh = 0;
                t_surf = fminf(t_gs, tmax);
            }
        }

        /* All lanes hit reorder so mesh / volume / skip land in separate warps. */
        atm_ser_bin_paths(doMesh, vol.event != ATM_EVENT_NONE);

        if (doMesh) {
            if (prof) {
                tSec = clock64();
            }
            if (!params.meshHandle) {
                meshHit = atm_mesh_closest_slab(&atm->mesh, org, dir, meshT1);
            } else {
                const AtmAccelHit a = atm_trace_mesh_slab(
                    org, dir, meshT0, meshT1, OPTIX_RAY_FLAG_DISABLE_ANYHIT, sampler);
                atm_mesh_from_accel(&atm->mesh, a, dir, &meshHit);
            }
            if (prof) {
                meshCycThis = atm_cycles_since(tSec);
                cMesh += meshCycThis;
            }
            didMesh = 1;
            t_surf = fminf(fminf(t_gs, meshHit.t), tmax);
            if (probeVol) {
                const float tRemain = t_surf - tProbe;
                if (tRemain > 1e-4f) {
                    float org2[3] = {
                        org[0] + dir[0] * tProbe,
                        org[1] + dir[1] * tProbe,
                        org[2] + dir[2] * tProbe};
                    struct AtmHit vol2;
                    atm_hit_clear(&vol2);
                    atm_run_volume_event(atm, org2, dir, tRemain, sampler, cieOn, prof, &vol2, &cGas, &cWc);
                    if (vol2.event != ATM_EVENT_NONE && vol2.t < tRemain) {
                        vol = vol2;
                        vol.t += tProbe;
                    }
                }
            } else {
                atm_run_volume_event(atm, org, dir, t_surf, sampler, cieOn, prof, &vol, &cGas, &cWc);
            }
        }

        /* GS-only: stop the volume at the Cu band so the walk can enter GS.
         * Mesh ground must not clip t_surf — gas/voxels compete all the way to mountain.obj. */
        if (atm->gs_enabled && above_terrain && dir[2] < -1e-5f) {
            const float t_band = (850.f - org[2]) / dir[2];
            if (t_band > 0.f) {
                t_surf = fminf(t_surf, t_band);
            }
        }

        const int vol_first = (vol.event != ATM_EVENT_NONE && vol.t < t_surf);
        const int gs_first = (!vol_first && t_gs < meshHit.t && t_gs < tmax);
        const int mesh_first = (!vol_first && !gs_first && meshHit.t < tmax);

        if (prof) {
            if (bounce == 0) {
                if (dir[2] < 0.f) {
                    nCamDown += 1ull;
                } else {
                    nCamUp += 1ull;
                }
            }
            if (didMesh) {
                const int geomHit = (meshHit.t < tmax) ? 1 : 0;
                if (bounce == 0) {
                    nCamT += 1ull;
                    cCam += meshCycThis;
                    if (geomHit) {
                        nCamHg += 1ull;
                    }
                    if (mesh_first) {
                        nCamHe += 1ull;
                    }
                } else {
                    nPostT += 1ull;
                    cPost += meshCycThis;
                    if (geomHit) {
                        nPostHg += 1ull;
                    }
                    if (mesh_first) {
                        nPostHe += 1ull;
                    }
                }
                if (hadCloudScatter && !hadSurfBounce && dir[2] < 0.f) {
                    nCdT += 1ull;
                    cCd += meshCycThis;
                    if (geomHit) {
                        nCdHg += 1ull;
                    }
                    if (mesh_first) {
                        nCdHe += 1ull;
                    }
                } else if (hadSurfBounce) {
                    nSurfT += 1ull;
                    cSurf += meshCycThis;
                    if (mesh_first) {
                        nSurfHe += 1ull;
                    }
                } else if (bounce != 0) {
                    nOthT += 1ull;
                    cOth += meshCycThis;
                    if (mesh_first) {
                        nOthHe += 1ull;
                    }
                }
            }
        }

        if (!recordedHit) {
            if (vol_first) {
                firstHitT = vol.t;
                firstEvent = vol.event;
            } else if (gs_first) {
                firstHitT = t_gs;
                firstEvent = ATM_EVENT_HIT_SURFACE;
            } else if (mesh_first) {
                firstHitT = meshHit.t;
                firstEvent = ATM_EVENT_HIT_SURFACE;
            } else {
                firstHitT = tmax;
                firstEvent = ATM_EVENT_ESCAPE;
            }
            recordedHit = 1;
        }

        if (vol_first) {
            escaped = 0;
            const float Tr_abs = atm_absorption_transmittance(atm, org, dir, vol.t);
            if (Tr_abs <= 0.f) {
                break;
            }
            throughput *= Tr_abs;

            const int voxelAbsorb = (vol.event == ATM_EVENT_ABSORB_VOXEL);
            const int gasAbsorb = (vol.event == ATM_EVENT_ABSORB_GAS);
            if (gasAbsorb || voxelAbsorb) {
                if (isLW) {
                    const float B = atm_planck_monochromatic(atm->wavelength_m, vol.temperature);
                    radiance += throughput * B;
                    if (bounce == 0) {
                        direct += throughput * B;
                    } else {
                        indirect += throughput * B;
                    }
                }
                break;
            }

            /* Competing risks already chose the component (htrdr: one Ks walk then
             * Bernoulli once). A second ks_g/ks_tot roll would make
             * P(HG)=(ks_p/ks_tot)^2. In this cloud that flip is ~0.05% of events
             * — not the 12% core gap; still wrong, so do not re-Bernoulli. */
            const int voxelScatter = (vol.event == ATM_EVENT_SCATTER_GAS) ? 0 : 1;
            float3 volNee = make_float3(0.f, 0.f, 0.f);
            if (prof) {
                tSec = clock64();
            }
            atm_add_sun_nee_volume(atm, vol.pos, dir, voxelScatter, throughput, sampler, radiance, volNee);
            if (prof) {
                cNee += atm_cycles_since(tSec);
            }
            if (bounce == 0) {
                firstVolNee += volNee;
                direct += volNee;
            } else {
                indirect += volNee;
            }

            float wo[3];
            if (voxelScatter) {
                atm_sample_hg(atm->mie_g, sampler.next_1d(), sampler.next_1d(), dir, wo);
            } else {
                atm_sample_rayleigh(sampler, dir, wo);
            }
            org[0] = vol.pos[0];
            org[1] = vol.pos[1];
            org[2] = vol.pos[2];
            dir[0] = wo[0];
            dir[1] = wo[1];
            dir[2] = wo[2];
            if (voxelScatter) {
                hadCloudScatter = 1;
            }
        } else if (gs_first) {
            escaped = 0;
            hadSurfBounce = 1;
            const float Tr_abs = atm_absorption_transmittance(atm, org, dir, t_gs);
            if (Tr_abs <= 0.f) {
                break;
            }
            throughput *= Tr_abs;
            if (atm->cie_enabled && atm->path_albedo >= 0.f) {
                gsIx.material.albedo = make_float3(atm->path_albedo, atm->path_albedo, atm->path_albedo);
            }
            if (isLW) {
                const float B = atm_planck_monochromatic(atm->wavelength_m, 288.15f);
                const float rho = fmaxf(gsIx.material.albedo.x, fmaxf(gsIx.material.albedo.y, gsIx.material.albedo.z));
                radiance += throughput * ((1.f - atm_clampf(rho, 0.f, 1.f)) * B);
            }
            float gsPos[3];
            atm_f3_to_arr(gsIx.position, gsPos);
            atm_add_sun_nee_surface(atm, gsPos, gsRay.direction, gsIx, throughput, sampler, radiance, bounce == 0 ? direct : indirect);
            atm_add_sky_surface(atm, gsIx, throughput, radiance, bounce == 0 ? direct : indirect);

            if (atm->path_albedo >= 0.f) {
                if (!atm_lambert_bounce(gsIx, sampler, throughput, org, dir)) {
                    break;
                }
            } else {
            pathPayload path(1u, (unsigned int)bounce, (unsigned int)maxB);
            path.pathThroughput = throughput;
            path.currentRayPayload.ray = gsRay;
            path.currentRayPayload.interaction = gsIx;
            sampleBrdfNextDirection(path, sampler);
            if (!path.active || (path.pathThroughput.x + path.pathThroughput.y + path.pathThroughput.z) <= 0.f) {
                break;
            }
            throughput = path.pathThroughput;
            atm_f3_to_arr(path.currentRayPayload.ray.origin, org);
            atm_f3_to_arr(path.currentRayPayload.ray.direction, dir);
            atm_normalize3(dir);
            }
        } else if (mesh_first) {
            escaped = 0;
            hadSurfBounce = 1;
            const float Tr_abs = atm_absorption_transmittance(atm, org, dir, meshHit.t);
            if (Tr_abs <= 0.f) {
                break;
            }
            throughput *= Tr_abs;
            float p[3] = {
                org[0] + dir[0] * meshHit.t,
                org[1] + dir[1] * meshHit.t,
                org[2] + dir[2] * meshHit.t};
            float ar = meshHit.albedo[0];
            float ag = meshHit.albedo[1];
            float ab = meshHit.albedo[2];
            if (atm->cie_enabled && atm->path_albedo >= 0.f) {
                ar = ag = ab = atm->path_albedo;
            }
            if (isLW) {
                const float B = atm_planck_monochromatic(atm->wavelength_m, meshHit.temperature);
                const float rho = fmaxf(ar, fmaxf(ag, ab));
                radiance += throughput * ((1.f - atm_clampf(rho, 0.f, 1.f)) * B);
            }
            const Material mat(make_float3(ar, ag, ab), 1.f, 0.f);
            const Interaction ix(
                make_float3(p[0], p[1], p[2]),
                make_float3(meshHit.n[0], meshHit.n[1], meshHit.n[2]),
                mat,
                true);
            const float3 rayDir = make_float3(dir[0], dir[1], dir[2]);
            atm_add_sun_nee_surface(atm, p, rayDir, ix, throughput, sampler, radiance, bounce == 0 ? direct : indirect);
            atm_add_sky_surface(atm, ix, throughput, radiance, bounce == 0 ? direct : indirect);

            if (atm->path_albedo >= 0.f) {
                if (!atm_lambert_bounce(ix, sampler, throughput, org, dir)) {
                    break;
                }
            } else {
            pathPayload path(1u, (unsigned int)bounce, (unsigned int)maxB);
            path.pathThroughput = throughput;
            path.currentRayPayload.ray = Ray(ix.position, rayDir);
            path.currentRayPayload.interaction = ix;
            sampleBrdfNextDirection(path, sampler);
            if (!path.active || (path.pathThroughput.x + path.pathThroughput.y + path.pathThroughput.z) <= 0.f) {
                break;
            }
            throughput = path.pathThroughput;
            atm_f3_to_arr(path.currentRayPayload.ray.origin, org);
            atm_f3_to_arr(path.currentRayPayload.ray.direction, dir);
            atm_normalize3(dir);
            }
        } else {
            /* Above the mountain: no volume hit this segment — step into the
             * terrain band and keep walking (do not treat as space escape). */
            if (atm->gs_enabled && above_terrain && dir[2] < -1e-5f) {
                const float t_band = (850.f - org[2]) / dir[2];
                if (t_band > 1e-3f && t_band < tmax) {
                    const float Tr_abs = atm_absorption_transmittance(atm, org, dir, t_band);
                    if (Tr_abs <= 0.f) {
                        break;
                    }
                    throughput *= Tr_abs;
                    org[0] += dir[0] * t_band;
                    org[1] += dir[1] * t_band;
                    org[2] = 850.f;
                    continue;
                }
            }
            /* htrdr: no scatter and no surface → break. Sun after a bounce
             * is NEE only; do not add the disk on miss. */
            escaped = 1;
            break;
        }

        const float tmaxW = fmaxf(fmaxf(throughput.x, throughput.y), throughput.z);
        if (tmaxW < 1e-6f) {
            break;
        }
        /* htrdr SW: RR only on surfaces (BSDF reflectivity). Never RR volume. */
        if (!vol_first && bounce >= 8) {
            const float q = fmaxf(0.05f, 1.f - tmaxW);
            if (sampler.next_1d() < q) {
                break;
            }
            throughput = throughput / fmaxf(1.f - q, 1e-3f);
        }
    }

    if (cieOn) {
        const float L = (radiance.x + radiance.y + radiance.z) * (1.f / 3.f);
        const float L0 = (firstVolNee.x + firstVolNee.y + firstVolNee.z) * (1.f / 3.f);
        if (cieSame) {
            const float wlen_nm = atm_ch.wavelength_m * 1.0e9f;
            atm_cie_xyz_from_L(atm_base, wlen_nm, L, xyz);
            atm_cie_xyz_from_L(atm_base, wlen_nm, L0, xyzFirst);
        } else {
            xyz[cie_ch] = L / fmaxf(wlen_pdf, 1e-20f);
            xyzFirst[cie_ch] = L0 / fmaxf(wlen_pdf, 1e-20f);
        }
        if (cieSame || cie_ch == 0 || cieSplit) {
            firstHitTKeep = firstHitT;
            firstEventKeep = firstEvent;
            escapedKeep = escaped;
        }
    } else {
        radianceOut = radiance;
        directOut = firstVolNee;
        indirectOut = radiance - firstVolNee;
        firstHitTKeep = firstHitT;
        firstEventKeep = firstEvent;
        escapedKeep = escaped;
    }
    } /* cie_ch */

    if (cieOn) {
        radianceOut = make_float3(xyz[0], xyz[1], xyz[2]);
        directOut = make_float3(xyzFirst[0], xyzFirst[1], xyzFirst[2]);
        indirectOut = radianceOut - directOut;
    }
    if (prof && params.atmBounceCycles) {
        atomicAdd(params.atmBounceCycles + 0, cMesh);
        atomicAdd(params.atmBounceCycles + 1, cWc);
        atomicAdd(params.atmBounceCycles + 2, cGas);
        atomicAdd(params.atmBounceCycles + 3, cNee);
        atomicAdd(params.atmBounceCycles + 4, atm_cycles_since(tPath0));
        atomicAdd(params.atmBounceCycles + 5, nCamT);
        atomicAdd(params.atmBounceCycles + 6, nCamHg);
        atomicAdd(params.atmBounceCycles + 7, nCamHe);
        atomicAdd(params.atmBounceCycles + 8, cCam);
        atomicAdd(params.atmBounceCycles + 9, nPostT);
        atomicAdd(params.atmBounceCycles + 10, nPostHg);
        atomicAdd(params.atmBounceCycles + 11, nPostHe);
        atomicAdd(params.atmBounceCycles + 12, cPost);
        atomicAdd(params.atmBounceCycles + 13, nCdT);
        atomicAdd(params.atmBounceCycles + 14, nCdHg);
        atomicAdd(params.atmBounceCycles + 15, nCdHe);
        atomicAdd(params.atmBounceCycles + 16, cCd);
        atomicAdd(params.atmBounceCycles + 17, nCamDown);
        atomicAdd(params.atmBounceCycles + 18, nCamUp);
        atomicAdd(params.atmBounceCycles + 19, nSurfT);
        atomicAdd(params.atmBounceCycles + 20, nSurfHe);
        atomicAdd(params.atmBounceCycles + 21, cSurf);
        atomicAdd(params.atmBounceCycles + 22, nOthT);
        atomicAdd(params.atmBounceCycles + 23, nOthHe);
        atomicAdd(params.atmBounceCycles + 24, cOth);
    }
    writeAtmosphereOutputs(idx, radianceOut, directOut, indirectOut, firstHitTKeep, escapedKeep, firstEventKeep);
}
