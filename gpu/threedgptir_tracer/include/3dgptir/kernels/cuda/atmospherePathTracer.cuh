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

static __device__ AtmAccelHit atm_trace_accel(
    OptixTraversableHandle handle,
    const float org[3],
    const float dir[3],
    float tmin,
    float tmax,
    unsigned sbtOffset,
    unsigned rayFlags,
    Sampler& sampler) {
    AtmAccelHit hit;
    hit.t = ATM_INF;
    hit.kind = 0;
    hit.prim = -1;
    hit.event = ATM_EVENT_NONE;
    hit.temperature = 0.f;
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
    unsigned int p7 = 0, p8 = 0, p9 = 0, p10 = 0, p11 = 0, p12 = 0, p13 = 0, p14 = 0, p15 = 0;
    unsigned int p16 = 0, p17 = 0, p18 = 0, p19 = 0, p20 = 0, p21 = 0, p22 = 0, p23 = 0;
    unsigned int p24 = 0, p25 = 0, p26 = 0, p27 = 0, p28 = 0, p29 = 0, p30 = 0, p31 = 0;
    optixTrace(
        handle,
        make_float3(org[0], org[1], org[2]),
        make_float3(dir[0], dir[1], dir[2]),
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        rayFlags,
        sbtOffset,
        1,
        0,
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
        p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31);
    if (p1 == 1u || p1 == 2u) {
        hit.t = __uint_as_float(p0);
        hit.kind = static_cast<int>(p1);
        hit.prim = static_cast<int>(p2);
        hit.event = static_cast<int>(p3);
        hit.temperature = __uint_as_float(p4);
    }
    return hit;
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
    atm_world_to_ijk(&atm->voxels, hit->pos, ijk);
    hit->ka = atm_fetch_ka(&atm->voxels, ijk[0], ijk[1], ijk[2]);
    hit->ks = atm_fetch_ks(&atm->voxels, ijk[0], ijk[1], ijk[2]);
    hit->kext = hit->ka + hit->ks;
    hit->temperature = a.temperature > 0.f ? a.temperature : atm_fetch_temp(&atm->voxels, ijk[0], ijk[1], ijk[2]);
}

static __device__ struct AtmHit atm_merge_gas(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tmax,
    Sampler& sampler,
    struct AtmHit vox) {
    const int walk_ks = atm->spectral_mode != ATM_SPECTRAL_LW;
    struct AtmHit gas = atm_gas_free_path(&atm->gas, org, dir, tmax, sampler, walk_ks);
    if (gas.event != ATM_EVENT_NONE && gas.t <= vox.t && gas.t < tmax) {
        return gas;
    }
    if (vox.event != ATM_EVENT_NONE && vox.t < tmax) {
        return vox;
    }
    struct AtmHit none;
    atm_hit_clear(&none);
    none.t = tmax;
    none.event = ATM_EVENT_NONE;
    return none;
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

static __device__ int atm_ground_occluded(
    const float org[3],
    const float dir[3],
    float tmax) {
    /* htrdr atmosphere_ground_trace_ray: triangles only. Volume NEE must not
     * use GS shadows (sun from cloud is upward; GS false hits zero scatter). */
    if (params.meshHandle) {
        Sampler rng;
        rng.init(1ull, 1ull);
        const AtmAccelHit h = atm_trace_accel(
            params.meshHandle,
            org,
            dir,
            2.f,
            tmax,
            ATM_SBT_MESH,
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            rng);
        return (h.kind == 1 && h.t < tmax && h.t > 2.f) ? 1 : 0;
    }
    if (params.atmosphere.mesh.ntri > 0) {
        const float t_mesh = atm_mesh_closest(&params.atmosphere.mesh, org, dir, tmax);
        if (t_mesh < tmax && t_mesh > 2.f) {
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
    const float3 lambert = ix.material.albedo * (ndl * (1.f / 3.14159265f));
    float scatterPdf = 0.f;
    const float3 brdfTimesCos = eval_material_fast_brdf_light_sample(rayDir, ix, L, scatterPdf);
    (void)scatterPdf;
    const float3 fcos = (brdfTimesCos.x + brdfTimesCos.y + brdfTimesCos.z) > 1e-8f ? brdfTimesCos : lambert;
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
    int escaped) {
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
    params.rayHitsCount[idx.z][idx.y][idx.x][0] = 0.f;
}

static __device__ void traceAtmospherePath(const uint3& idx, Sampler& sampler) {
    const AtmParams* atm = &params.atmosphere;
    Ray ray(params.rayWorldOrigin(idx), params.rayWorldDirection(idx));

    float org[3], dir[3];
    atm_f3_to_arr(ray.origin, org);
    atm_f3_to_arr(ray.direction, dir);
    atm_normalize3(dir);

    float3 throughput = make_float3(1.f, 1.f, 1.f);
    float3 radiance = make_float3(0.f, 0.f, 0.f);
    float3 direct = make_float3(0.f, 0.f, 0.f);
    float3 indirect = make_float3(0.f, 0.f, 0.f);
    float firstHitT = 0.f;
    int recordedHit = 0;
    int escaped = 1;

    const int maxB = atm->max_bounces > 0 ? atm->max_bounces : (int)params.maxBounces;
    const float tmax0 = atm->path_tmax > 0.f ? atm->path_tmax : 1.0e5f;
    const int isLW = atm->spectral_mode == ATM_SPECTRAL_LW;

    for (int bounce = 0; bounce < maxB; ++bounce) {
        const float tmax = tmax0;
        const int above_terrain = (org[2] > 850.f);
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

        /* --ground gs: skip mesh above 850 so the walk reaches GS (104659).
         * --ground mesh: GS off — always trace mountain.obj from the camera. */
        if (!above_terrain || !atm->gs_enabled) {
            if (!params.meshHandle) {
                meshHit = atm_mesh_closest_hit(&atm->mesh, org, dir, tmax);
            } else {
                const AtmAccelHit a = atm_trace_accel(
                    params.meshHandle,
                    org,
                    dir,
                    1e-4f,
                    tmax,
                    ATM_SBT_MESH,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    sampler);
                atm_mesh_from_accel(&atm->mesh, a, dir, &meshHit);
            }
        }

        t_surf = fminf(fminf(t_gs, meshHit.t), tmax);
        if (above_terrain && dir[2] < -1e-5f) {
            const float t_band = (850.f - org[2]) / dir[2];
            if (t_band > 0.f) {
                t_surf = fminf(t_surf, t_band);
            }
        }
        if (params.volHandle) {
            const AtmAccelHit a = atm_trace_accel(
                params.volHandle,
                org,
                dir,
                1e-4f,
                t_surf,
                ATM_SBT_VOL,
                0u,
                sampler);
            atm_vol_from_accel(atm, org, dir, a, &vol);
            if (vol.event == ATM_EVENT_NONE) {
                vol = atm_next_volume_event(atm, org, dir, t_surf, sampler);
            } else {
                vol = atm_merge_gas(atm, org, dir, t_surf, sampler, vol);
            }
        } else {
            vol = atm_next_volume_event(atm, org, dir, t_surf, sampler);
        }

        const int vol_first = (vol.event != ATM_EVENT_NONE && vol.t < t_surf);
        const int gs_first = (!vol_first && t_gs < meshHit.t && t_gs < tmax);
        const int mesh_first = (!vol_first && !gs_first && meshHit.t < tmax);

        if (!recordedHit) {
            if (vol_first) {
                firstHitT = vol.t;
            } else if (gs_first) {
                firstHitT = t_gs;
            } else if (mesh_first) {
                firstHitT = meshHit.t;
            } else {
                firstHitT = tmax;
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

            /* Collision-point Bernoulli: r < ks_gas/ks → Rayleigh else HG. */
            float ka_g = 0.f, ks_g = 0.f;
            atm_gas_k_at_z(&atm->gas, vol.pos[2], &ka_g, &ks_g);
            int ijk[3];
            atm_world_to_ijk(&atm->voxels, vol.pos, ijk);
            const float ks_p = atm_fetch_ks(&atm->voxels, ijk[0], ijk[1], ijk[2]);
            const float ks_tot = ks_g + ks_p;
            if (ks_tot <= 1e-20f) {
                break;
            }
            const int voxelScatter = sampler.next_1d() >= (ks_g / ks_tot);
            atm_add_sun_nee_volume(atm, vol.pos, dir, voxelScatter, throughput, sampler, radiance, bounce == 0 ? direct : indirect);

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
        } else if (gs_first) {
            escaped = 0;
            const float Tr_abs = atm_absorption_transmittance(atm, org, dir, t_gs);
            if (Tr_abs <= 0.f) {
                break;
            }
            throughput *= Tr_abs;
            if (isLW) {
                const float B = atm_planck_monochromatic(atm->wavelength_m, 288.15f);
                const float rho = fmaxf(gsIx.material.albedo.x, fmaxf(gsIx.material.albedo.y, gsIx.material.albedo.z));
                radiance += throughput * ((1.f - atm_clampf(rho, 0.f, 1.f)) * B);
            }
            float gsPos[3];
            atm_f3_to_arr(gsIx.position, gsPos);
            atm_add_sun_nee_surface(atm, gsPos, gsRay.direction, gsIx, throughput, sampler, radiance, bounce == 0 ? direct : indirect);
            atm_add_sky_surface(atm, gsIx, throughput, radiance, bounce == 0 ? direct : indirect);

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
        } else if (mesh_first) {
            escaped = 0;
            const float Tr_abs = atm_absorption_transmittance(atm, org, dir, meshHit.t);
            if (Tr_abs <= 0.f) {
                break;
            }
            throughput *= Tr_abs;
            float p[3] = {
                org[0] + dir[0] * meshHit.t,
                org[1] + dir[1] * meshHit.t,
                org[2] + dir[2] * meshHit.t};
            if (isLW) {
                const float B = atm_planck_monochromatic(atm->wavelength_m, meshHit.temperature);
                const float rho = fmaxf(meshHit.albedo[0], fmaxf(meshHit.albedo[1], meshHit.albedo[2]));
                radiance += throughput * ((1.f - atm_clampf(rho, 0.f, 1.f)) * B);
            }
            const Material mat(
                make_float3(meshHit.albedo[0], meshHit.albedo[1], meshHit.albedo[2]),
                1.f,
                0.f);
            const Interaction ix(
                make_float3(p[0], p[1], p[2]),
                make_float3(meshHit.n[0], meshHit.n[1], meshHit.n[2]),
                mat,
                true);
            const float3 rayDir = make_float3(dir[0], dir[1], dir[2]);
            atm_add_sun_nee_surface(atm, p, rayDir, ix, throughput, sampler, radiance, bounce == 0 ? direct : indirect);
            atm_add_sky_surface(atm, ix, throughput, radiance, bounce == 0 ? direct : indirect);

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
        } else {
            /* Above the mountain: no volume hit this segment — step into the
             * terrain band and keep walking (do not treat as space escape). */
            if (above_terrain && dir[2] < -1e-5f) {
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
            /* Escaped atmosphere: SW looking into the sun (Tr already sampled as 1). */
            if (!isLW && atm_dir_in_sun_cone(&atm->sun, dir) && atm->sun.radiance > 0.f) {
                const float3 contrib = throughput * atm->sun.radiance;
                radiance += contrib;
                if (bounce == 0) {
                    direct += contrib;
                } else {
                    indirect += contrib;
                }
            }
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

    writeAtmosphereOutputs(idx, radiance, direct, indirect, firstHitT, escaped);
}
