// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <optix.h>

#include <3dgptir/kernels/cuda/gaussianParticles.cuh>
#include <3dgptir/kernels/cuda/lightSampler.cuh>
#include <3dgptir/kernels/cuda/material.cuh>
#include <3dgptir/kernels/cuda/sampler.cuh>
#include <3dgptir/payLoad.h>
#include <3dgptir/pipelineParameters.h>

static constexpr float kSelfOcclusionRayOriginOffset = 2e-2f;
static constexpr float kMaxSelfOcclusionOffset = 1e-1f - kSelfOcclusionRayOriginOffset;
static constexpr float kSelfOcclusionNormalDotThreshold = 0.1f;

struct RayHit {
    unsigned int particleId;
    float distance;

    static constexpr unsigned int InvalidParticleId = 0xFFFFFFFF;
    static constexpr float InfiniteDistance         = 1e20f;
};
using RayPayload = RayHit[PipelineParameters::MaxNumHitPerTrace];

static __device__ __forceinline__ float misWeight(float pdfA, float pdfB) {
    const float a2 = pdfA * pdfA;
    const float b2 = pdfB * pdfB;
    return a2 / fmaxf(a2 + b2, 1e-6f);
}

static __device__ __inline__ float2 intersectAABB(const OptixAabb& aabb, const Ray& ray) {
    const float3 t0   = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - ray.origin) / ray.direction;
    const float3 t1   = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - ray.origin) / ray.direction;
    const float3 tmax = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    const float3 tmin = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    return float2{fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z))), fminf(tmax.x, fminf(tmax.y, tmax.z))};
}

static __device__ __inline__ uint32_t optixPrimitiveIndex() {
    return PipelineParameters::InstancePrimitive ? optixGetInstanceIndex() : (PipelineParameters::CustomPrimitive ? optixGetPrimitiveIndex() : static_cast<uint32_t>(optixGetPrimitiveIndex() / params.gPrimNumTri));
}

static __device__ __inline__ void trace(
    RayPayload& rayPayload,
    const Ray& ray,
    const float tmin,
    const float tmax) {
    // 定义32个uint32寄存器，作为optixTrace跨追踪通信变量
    // 配对规则：r2k=粒子ID，r2k+1=相交距离float转uint，一组2个寄存器存1个交点，32个寄存器共支持16个交点
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;

    // 所有偶数寄存器初始化：存储粒子ID，赋值无效粒子标记
    r0 = r2 = r4 = r6 = r8 = r10 = r12 = r14 = r16 = r18 = r20 = r22 = r24 = r26 = r28 = r30 = RayHit::InvalidParticleId;
    // 所有奇数寄存器初始化：存储相交距离，先把无穷远浮点数无损转为uint32赋值
    r1 = r3 = r5 = r7 = r9 = r11 = r13 = r15 = r17 = r19 = r21 = r23 = r25 = r27 = r29 = r31 = __float_as_int(RayHit::InfiniteDistance);

#if defined(ATM_PAYLOAD_WORDS) && (ATM_PAYLOAD_WORDS < 32)
    (void)tmin;
    (void)tmax;
#else
    // Trace the ray against our scene hierarchy
    optixTrace(params.handle, ray.origin, ray.direction,
               tmin,                     // Min intersection distance
               tmax,                     // Max intersection distance
               0.0f,                     // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | (PipelineParameters::SurfelPrimitive ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES),
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
               r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31);
#endif

    rayPayload[0].particleId  = r0;
    rayPayload[0].distance    = __uint_as_float(r1);
    rayPayload[1].particleId  = r2;
    rayPayload[1].distance    = __uint_as_float(r3);
    rayPayload[2].particleId  = r4;
    rayPayload[2].distance    = __uint_as_float(r5);
    rayPayload[3].particleId  = r6;
    rayPayload[3].distance    = __uint_as_float(r7);
    rayPayload[4].particleId  = r8;
    rayPayload[4].distance    = __uint_as_float(r9);
    rayPayload[5].particleId  = r10;
    rayPayload[5].distance    = __uint_as_float(r11);
    rayPayload[6].particleId  = r12;
    rayPayload[6].distance    = __uint_as_float(r13);
    rayPayload[7].particleId  = r14;
    rayPayload[7].distance    = __uint_as_float(r15);
    rayPayload[8].particleId  = r16;
    rayPayload[8].distance    = __uint_as_float(r17);
    rayPayload[9].particleId  = r18;
    rayPayload[9].distance    = __uint_as_float(r19);
    rayPayload[10].particleId = r20;
    rayPayload[10].distance   = __uint_as_float(r21);
    rayPayload[11].particleId = r22;
    rayPayload[11].distance   = __uint_as_float(r23);
    rayPayload[12].particleId = r24;
    rayPayload[12].distance   = __uint_as_float(r25);
    rayPayload[13].particleId = r26;
    rayPayload[13].distance   = __uint_as_float(r27);
    rayPayload[14].particleId = r28;
    rayPayload[14].distance   = __uint_as_float(r29);
    rayPayload[15].particleId = r30;
    rayPayload[15].distance   = __uint_as_float(r31);
}

static __device__ __forceinline__ bool isSelfOcclusionHit(
    const RayHit& rayHit,
    const Ray& ray,
    const float* particleShadingNormal) {
    if ((rayHit.distance > kMaxSelfOcclusionOffset) || (particleShadingNormal == nullptr)) {
        return false;
    }

    const float3 shadingNormal = make_float3(
        particleShadingNormal[rayHit.particleId * 3 + 0],
        particleShadingNormal[rayHit.particleId * 3 + 1],
        particleShadingNormal[rayHit.particleId * 3 + 2]);
    return dot(shadingNormal, ray.direction) > kSelfOcclusionNormalDotThreshold;
}

template <int ParticleKernelDegree = 4, bool SurfelPrimitive = false>
static __device__ __forceinline__ float shadowHitAlpha(
    const Ray& ray,
    const int32_t particleIdx) {
    float3 particlePosition;
    float3 particleScale;
    float33 particleRotation;
    float particleDensity;
    fetchParticleDensity(
        particleIdx,
        params.particleDensity,
        particlePosition,
        particleScale,
        particleRotation,
        particleDensity);

    const float3 giscl   = make_float3(1.0f / particleScale.x, 1.0f / particleScale.y, 1.0f / particleScale.z);
    const float3 gposc   = ray.origin - particlePosition;
    const float3 gposcr  = gposc * particleRotation;
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = ray.direction * particleRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);
    const float3 gcrod   = SurfelPrimitive ? gro + grd * -gro.z / grd.z : cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);

    const float gres = particleResponse<ParticleKernelDegree>(grayDist);
    const float alpha = fminf(0.99f, gres * particleDensity);
    return (gres > params.hitMinGaussianResponse && alpha > params.alphaMinThreshold) ? alpha : 0.0f;
}

template <bool EnableSelfOcclusionRejection = true>
static __device__ __inline__ bool traceShadowMonteCarloOccluded(
    const Ray& ray,
    Sampler& sampler,
    const float tmin = 0.0f,
    const float tmax = RayHit::InfiniteDistance) {
    constexpr float epsT = 1e-9f;
    const float2 minMaxT = intersectAABB(params.aabb, ray);
    float startT = fmaxf(tmin, minMaxT.x - epsT);
    const float endT = fminf(tmax, minMaxT.y) + epsT;
    if (startT >= endT) {
        return false;
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
            if (EnableSelfOcclusionRejection && isSelfOcclusionHit(rayHit, ray, params.particleShadingNormal)) {
                continue;
            }

            const float alpha = shadowHitAlpha<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
                ray,
                rayHit.particleId);
            if (alpha > 0.0f && sampler.next_1d() < alpha) {
                return true;
            }
        }

        if (batchEndT <= startT) {
            break;
        }
        startT = batchEndT;
    }

    return false;
}

template <bool EnableSelfOcclusionRejection>
static __device__ __inline__ void rayIntersect(
    const Ray& ray,           // 输入：要追踪的光线
    rayPayload& payload,      // 输入/输出：携带路径状态（透射率、辐射度等）
    Sampler& sampler) {       // 输入：随机数采样器（用于后续俄罗斯轮盘赌等）
    constexpr float epsT = 1e-9;
    // 计算光线与场景全局包围盒的相交区间 (t_near, t_far)
    const float2 minMaxT = intersectAABB(params.aabb, ray);
    RayPayload hitPayload; // 长度为 16 的 RayHit 数组

#ifdef ENABLE_MIS
    const float scatterPdf = payload.scatterPdf;
    const float lightPdf   = payload.lightPdf;
#endif
    // 重置payload，从光线起点开始，t起始值略提前一点（负方向偏移epsT），避免光线起点就在包围盒边界上
    payload = rayPayload(ray, fmaxf(0.0f, minMaxT.x - epsT));
#ifdef ENABLE_MIS
    payload.scatterPdf = scatterPdf;
    payload.lightPdf   = lightPdf;
#endif
    float integratedDepth = 0.f;
    Material integratedMaterial;
    float3 integratedShadingnormal = make_float3(0.f);

    while ((payload.lastHitDistance <= minMaxT.y) && (payload.transmittance > params.minTransmittance)) {
        //发射一根光线
        trace(hitPayload, payload.ray, payload.lastHitDistance + epsT, minMaxT.y + epsT);
        if (hitPayload[0].particleId == RayHit::InvalidParticleId) {
            break;
        }

#pragma unroll
        for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; i++) {
            const RayHit rayHit = hitPayload[i];

            if ((rayHit.particleId != RayHit::InvalidParticleId) && (payload.transmittance > params.minTransmittance)) {
                if (EnableSelfOcclusionRejection && isSelfOcclusionHit(rayHit, payload.ray, params.particleShadingNormal)) {
                    payload.lastHitDistance = fmaxf(payload.lastHitDistance, rayHit.distance);
                    continue;
                }

                const bool acceptedHit = processHit<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
                    payload.ray.origin,
                    payload.ray.direction,
                    rayHit.particleId,
                    params.particleDensity,
                    params.particleMaterial,
                    params.particleRadiance,
                    params.hitMinGaussianResponse,
                    params.alphaMinThreshold,
                    params.sphDegree,
                    params.particleShadingNormal,
                    &payload.transmittance,
                    &payload.radiance,
                    &integratedDepth,
                    &payload.hitDistanceSecondMoment,
                    &integratedMaterial,
#ifdef ENABLE_NORMALS
                    &integratedShadingnormal,
                    &payload.normal
#else
                    nullptr,
                    nullptr
#endif
                );
                if (acceptedHit) {
                    payload.hit = 1;
                    const float rayOpacity = 1.0f - payload.transmittance;
                    //深度方差（景深模糊用）
                    payload.depthDistortion = fmaxf(rayOpacity * payload.hitDistanceSecondMoment - integratedDepth * integratedDepth, 0.0f);

                    // NOTE(qi): Race condition here, but as we are writing the same value, it seems it is safe.
                    params.particleVisibility[rayHit.particleId] = 1;
                }

                payload.lastHitDistance = fmaxf(payload.lastHitDistance, rayHit.distance);

#ifdef ENABLE_HIT_COUNTS
                payload.hitsCount += acceptedHit ? 1.0f : 0.f;
#endif
            }
        }
    }

    const float rayOpacity = 1.0f - payload.transmittance;
    payload.hitDistance = integratedDepth;
    payload.valid = false;
    if (payload.hit && rayOpacity > 0.5f) {
        payload.interaction = Interaction(
            payload.ray.origin,
            payload.ray.direction,
            integratedDepth,
            integratedShadingnormal,
            integratedMaterial,
            rayOpacity);
    } else {
        const float rand_u = sampler.next_1d();
        payload.valid = (rand_u < payload.transmittance);
    }
}

template <bool EnableSelfOcclusionRejection, typename PipelineParams>
static __device__ __inline__ void rayIntersectBwd(
    const Ray& ray,
    const float rayOpacity,
    const float rayMaxHitDistance,
    const MaterialGrad& materialGrad,
    const PipelineParams& pipelineParams) {
    const float invRayOpacity = 1.0f / fmaxf(rayOpacity, 1e-12f);
    const Material rayMaterialGrad(
        materialGrad.dAlbedo * invRayOpacity,
        materialGrad.dRoughness * invRayOpacity,
        materialGrad.dMetallic * invRayOpacity);

    constexpr float epsT = 1e-9;
    const float2 minMaxT = intersectAABB(pipelineParams.aabb, ray);
    float startT         = fmaxf(0.0f, minMaxT.x - epsT);
    const float endT     = fminf(rayMaxHitDistance, minMaxT.y) + epsT;

    float rayTransmittance = 1.f;
    RayPayload hitPayload;

    while ((startT < endT) && (rayTransmittance > pipelineParams.minTransmittance)) {
        trace(hitPayload, ray, startT + epsT, endT);
        if (hitPayload[0].particleId == RayHit::InvalidParticleId) {
            break;
        }

#pragma unroll
        for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; i++) {
            const RayHit rayHit = hitPayload[i];

            if ((rayHit.particleId != RayHit::InvalidParticleId) && (rayTransmittance > pipelineParams.minTransmittance)) {
                if (EnableSelfOcclusionRejection && isSelfOcclusionHit(rayHit, ray, pipelineParams.particleShadingNormal)) {
                    startT = fmaxf(startT, rayHit.distance);
                    continue;
                }

                float3 particlePosition;
                float3 particleScale;
                float33 particleRotation;
                float particleDensity;
                fetchParticleDensity(
                    rayHit.particleId,
                    pipelineParams.particleDensity,
                    particlePosition,
                    particleScale,
                    particleRotation,
                    particleDensity);

                const float3 giscl   = make_float3(1.0f / particleScale.x, 1.0f / particleScale.y, 1.0f / particleScale.z);
                const float3 gposc   = ray.origin - particlePosition;
                const float3 gposcr  = gposc * particleRotation;
                const float3 gro     = giscl * gposcr;
                const float3 rayDirR = ray.direction * particleRotation;
                const float3 grdu    = giscl * rayDirR;
                const float3 grd     = safe_normalize(grdu);
                const float3 gcrod   = PipelineParameters::SurfelPrimitive ? gro + grd * -gro.z / grd.z : cross(grd, gro);
                const float grayDist = dot(gcrod, gcrod);

                const float gres   = particleResponse<PipelineParameters::ParticleKernelDegree>(grayDist);
                const float galpha = fminf(0.99f, gres * particleDensity);
                if ((gres > pipelineParams.hitMinGaussianResponse) && (galpha > pipelineParams.alphaMinThreshold)) {
                    const float weight = galpha * rayTransmittance;
                    Material& particleMaterialGrad = pipelineParams.particleMaterialGrad[rayHit.particleId];
                    atomicAdd(&particleMaterialGrad.albedo.x, weight * rayMaterialGrad.albedo.x);
                    atomicAdd(&particleMaterialGrad.albedo.y, weight * rayMaterialGrad.albedo.y);
                    atomicAdd(&particleMaterialGrad.albedo.z, weight * rayMaterialGrad.albedo.z);
                    atomicAdd(&particleMaterialGrad.roughness, weight * rayMaterialGrad.roughness);
#ifdef ENABLE_METALLIC
                    atomicAdd(&particleMaterialGrad.metallic, weight * rayMaterialGrad.metallic);
#endif
                    rayTransmittance *= (1.0f - galpha);
                }
                startT = fmaxf(startT, rayHit.distance);
            }
        }
    }
}

template <typename PipelineParams>
static __device__ __inline__ void PendingRayDirectionGradBwd(
    pathPayload& path,
    const PipelineParams& pipelineParams) {
    PendingRayDirectionGrad& pending = path.pendingRayDirectionGrad;
    if (!pending.valid) {
        return;
    }

    const float dRoughness = dot(path.currentRayPayload.rayDirGrad, pending.dNextDirDRoughness);
    const Ray pendingRay = pending.ray;
    const float pendingOpacity = pending.opacity;
    const float pendingMaxHitDistance = pending.maxHitDistance;
    const unsigned int pendingNumBounces = pending.numBounces;
    path.pendingRayDirectionGrad.clear();
    if (dRoughness == 0.0f) {
        return;
    }

    MaterialGrad materialGrad;
    materialGrad.dRoughness = dRoughness;
    if (pendingNumBounces > 1u) {
        rayIntersectBwd<true>(
            pendingRay,
            pendingOpacity,
            pendingMaxHitDistance,
            materialGrad,
            pipelineParams);
    } else {
        rayIntersectBwd<false>(
            pendingRay,
            pendingOpacity,
            pendingMaxHitDistance,
            materialGrad,
            pipelineParams);
    }
}

static __device__ __inline__ void sampleBrdfNextDirection(
    pathPayload& path,
    Sampler& sampler) {
    const Ray currentRay = path.currentRayPayload.ray;
    const Interaction currentInteraction = path.currentRayPayload.interaction;

    float3 nextRayDirection = currentRay.direction;
    float scatterPdf = 0.0f;
    const float3 brdfThroughput = sample_material_fast_brdf_throughput(
        currentRay.direction,
        sampler,
        currentInteraction,
        nextRayDirection,
        scatterPdf);

    const float3 nextPathThroughput = path.pathThroughput * brdfThroughput;
    if (nextPathThroughput.x == 0.0f && nextPathThroughput.y == 0.0f && nextPathThroughput.z == 0.0f) {
        path.pathThroughput = make_float3(0.0f);
        path.active = 0u;
        return;
    }

    path.pathThroughput = nextPathThroughput;
    Ray nextRay(currentInteraction.position + safe_normalize(nextRayDirection) * kSelfOcclusionRayOriginOffset, nextRayDirection);
    path.currentRayPayload = rayPayload(nextRay, 0.0f);
#ifdef ENABLE_MIS
    path.currentRayPayload.scatterPdf = scatterPdf;
    path.currentRayPayload.lightPdf = lightSamplerPdf(currentInteraction.position, nextRay.direction);
#endif
}

static __device__ __inline__ void sampleNee(
    pathPayload& path,
    Sampler& sampler) {
    // 取出当前光线与场景的交点交互信息（位置、法向、材质、几何信息等）
    const Interaction currentInteraction = path.currentRayPayload.interaction;

    float lightPdf = 0.0f; // 光源采样概率密度
    float scatterPdf = 0.0f; // BRDF散射采样概率密度
    // 1. 对当前交点位置采样一条光源方向wi，返回光源采样结构体
    // LightSample包含：入射方向wi、光源辐照度Li、采样pdf、光源距离dist、光源类型
    const LightSample lightSample = sampleLight(currentInteraction.position, sampler);
    const float3 lightDirection = lightSample.wi;
    lightPdf = lightSample.pdf;

    // 分支1：光源采样失败（pdf为0 或光源辐射为纯黑，无照明），清空载荷直接返回
    if (lightPdf <= 0.0f || (lightSample.Li.x == 0.0f && lightSample.Li.y == 0.0f && lightSample.Li.z == 0.0f)) {
        path.emitterRayPayload = rayPayload();
        return;
    }
    // 分支2：光源方向在表面背面（点积<=0，背光），无贡献，返回
    if (dot(currentInteraction.shadingnormal, lightDirection) <= 0.0f) {
        path.emitterRayPayload = rayPayload();
        return;
    }

    // 2. 计算BRDF * cos(theta)，同时输出BRDF采样pdf(scatterPdf)
    // 参数：出射方向(当前射线反向)、交点信息、光源入射方向、输出散射pdf
    const float3 brdfTimesCos = eval_material_fast_brdf_light_sample(
        path.currentRayPayload.ray.direction,
        currentInteraction,
        lightDirection,
        scatterPdf);
    
    // 点光源特殊处理：取消BRDF散射pdf，仅使用光源采样pdf做权重分母
    // 点光源无立体角分布，不需要多重重要性采样(MIS)混合两个pdf
    if (lightSample.lightType == LightSamplerType_Point) {
        scatterPdf = 0.0f;
    }

    // 3. 构造阴影射线，增加微小偏移避免自遮挡
    // kSelfOcclusionRayOriginOffset：沿光源方向向前偏移一小段距离，防止交点自身遮挡阴影射线
    const Ray lightRay(
        currentInteraction.position + lightDirection * kSelfOcclusionRayOriginOffset,
        lightDirection);

    // 阴影射线最大t值：
    // 若光源距离有限，tmax = 光源距离 - 偏移量，防止射线穿过光源
    // 无限远光源(环境光)：设为无限距离
    const float shadowTmax = lightSample.dist < 1e19f
        ? fmaxf(0.0f, lightSample.dist - kSelfOcclusionRayOriginOffset)
        : RayHit::InfiniteDistance;
    
    // 4. 追踪阴影射线，判断是否被遮挡
    // traceShadowMonteCarloOccluded<true>：模板true代表开启蒙特卡洛阴影采样(软阴影/随机透明遮挡)
    // 返回true=中间有遮挡，光源不可见，清空载荷返回
    if (traceShadowMonteCarloOccluded<true>(lightRay, sampler, 0.0f, shadowTmax)) {
        path.emitterRayPayload = rayPayload();
        return;
    }
    path.emitterRayPayload = rayPayload(lightRay, 0.0f);

    path.emitterRayPayload.lightPdf = lightPdf;
    path.emitterRayPayload.scatterPdf = scatterPdf;
    path.emitterRayPayload.light = lightSample.Li;
    // 光路吞吐量 * BRDF*cos * 光源辐照度 / 光源采样pdf
    path.emitterRayPayload.contribution = path.pathThroughput * brdfTimesCos * path.emitterRayPayload.light / fmaxf(lightPdf, 1e-6f);
}

#ifdef ENABLE_MIS
template <typename PipelineParams>
static __device__ __inline__ void accumulateNeeGradBwd(
    pathPayload& path,
    const Ray& currentRay,
    const Interaction& currentInteraction,
    PipelineParams& pipelineParams) {
    if (path.numBounces != 1u) {
        return;
    }
    if (path.emitterRayPayload.lightPdf <= 0.0f) {
        return;
    }

    float scatterPdf = 0.0f;
    const float3 lightDirection = path.emitterRayPayload.ray.direction;
    const FastBrdfValueGrad brdfTimesCos = eval_material_fast_brdf_light_sample_with_grads(
        currentRay.direction,
        currentInteraction,
        lightDirection,
        scatterPdf);

    const float lightSideMis = misWeight(
        path.emitterRayPayload.lightPdf,
        path.emitterRayPayload.scatterPdf);
    const float neeScale = lightSideMis / fmaxf(path.emitterRayPayload.lightPdf, 1e-6f);

    // Keep NEE envmap gradients disabled by default: differentiating the sampled
    // environment texel here has high variance.
    //const float3 environmentGrad = path.accumulatedLightingGrad * path.pathThroughput * brdfTimesCos.value * neeScale;
    //const float3 background = getBackgroundColorBwd(lightDirection, environmentGrad, pipelineParams);
    const float3 background = getBackgroundColor(lightDirection);
    const float3 dLoss_dBrdf = path.accumulatedLightingGrad * path.pathThroughput * background * neeScale;

    path.currentRayPayload.interaction.materialGrad.dAlbedo += dLoss_dBrdf * brdfTimesCos.dBrdf_dAlbedo;
    
    // NEE samples bright environment directions directly; enabling this roughness gradient can easily overfit specular highlights.
    //path.currentRayPayload.interaction.materialGrad.dRoughness += dot(dLoss_dBrdf, brdfTimesCos.dBrdf_dRoughness);
// #ifdef ENABLE_METALLIC
//     path.currentRayPayload.interaction.materialGrad.dMetallic += dot(dLoss_dBrdf, brdfTimesCos.dBrdf_dMetallic);
// #endif
}
#endif


template <typename PipelineParams>
static __device__ __inline__ void sampleBrdfNextDirectionBwd(
    pathPayload& path,
    Sampler& sampler,
    PipelineParams& pipelineParams) {
    const Ray currentRay = path.currentRayPayload.ray;
    const Interaction currentInteraction = path.currentRayPayload.interaction;

    float3 nextRayDirection = currentRay.direction;
    float scatterPdf = 0.0f;
    const FastBrdfValueGrad brdfThroughput = sample_material_fast_brdf_throughput_with_grads(
        currentRay.direction,
        sampler,
        currentInteraction,
        nextRayDirection,
        scatterPdf);
    const float3 nextPathThroughput = path.pathThroughput * brdfThroughput.value;
    if (nextPathThroughput.x == 0.0f && nextPathThroughput.y == 0.0f && nextPathThroughput.z == 0.0f) {
        path.pathThroughput = make_float3(0.0f);
        path.active = 0u;
        return;
    }

#ifdef ENABLE_MIS
    const float lightPdf = lightSamplerPdf(currentInteraction.position, nextRayDirection);
    const float nextThroughputMax = fmaxf(nextPathThroughput.x, fmaxf(nextPathThroughput.y, nextPathThroughput.z));
    const bool firstBounceNeeActive = path.numBounces == 1u && nextThroughputMax >= 1e-4f;
    if (firstBounceNeeActive) {
        accumulateNeeGradBwd(path, currentRay, currentInteraction, pipelineParams);
    }
#endif

    float3 brdfDependentLighting = path.accumulatedLighting;
#ifdef ENABLE_MIS
    if (firstBounceNeeActive) {
        const float lightSideMis = misWeight(
            path.emitterRayPayload.lightPdf,
            path.emitterRayPayload.scatterPdf);
        brdfDependentLighting -= path.emitterRayPayload.contribution * lightSideMis;
    }
#endif

    const float3 dLoss_dBrdfNumerator = path.accumulatedLightingGrad * brdfDependentLighting;
    const float3 dLoss_dBrdf = make_float3(
        brdfThroughput.value.x > FastBrdfEps ? dLoss_dBrdfNumerator.x / brdfThroughput.value.x : 0.0f,
        brdfThroughput.value.y > FastBrdfEps ? dLoss_dBrdfNumerator.y / brdfThroughput.value.y : 0.0f,
        brdfThroughput.value.z > FastBrdfEps ? dLoss_dBrdfNumerator.z / brdfThroughput.value.z : 0.0f);
    path.currentRayPayload.interaction.materialGrad.dAlbedo += dLoss_dBrdf * brdfThroughput.dBrdf_dAlbedo;
    path.currentRayPayload.interaction.materialGrad.dRoughness += dot(dLoss_dBrdf, brdfThroughput.dBrdf_dRoughness);
#ifdef ENABLE_METALLIC
    path.currentRayPayload.interaction.materialGrad.dMetallic += dot(dLoss_dBrdf, brdfThroughput.dBrdf_dMetallic);
#endif

    path.pendingRayDirectionGrad.set(
        currentRay,
        1.0f - path.currentRayPayload.transmittance,
        path.currentRayPayload.lastHitDistance,
        brdfThroughput.dNextDir_dRoughness,
        path.numBounces);

    if (path.numBounces > 1u) {
        rayIntersectBwd<true>(
            currentRay,
            1.0f - path.currentRayPayload.transmittance,
            path.currentRayPayload.lastHitDistance,
            path.currentRayPayload.interaction.materialGrad,
            pipelineParams);
    } else {
        rayIntersectBwd<false>(
            currentRay,
            1.0f - path.currentRayPayload.transmittance,
            path.currentRayPayload.lastHitDistance,
            path.currentRayPayload.interaction.materialGrad,
            pipelineParams);
    }

    path.pathThroughput = nextPathThroughput;
    Ray nextRay(currentInteraction.position + safe_normalize(nextRayDirection) * kSelfOcclusionRayOriginOffset, nextRayDirection);
    path.currentRayPayload = rayPayload(nextRay, 0.0f);
#ifdef ENABLE_MIS
    path.currentRayPayload.scatterPdf = scatterPdf;
    path.currentRayPayload.lightPdf = lightPdf;
#endif
}

static __device__ __inline__ void writePrimaryRayOutputs(
    const uint3& idx,
    const rayPayload& payload) {
    // 输出RGB累积辐射亮度 ∑ T_iα_i L_i（未归一化，标准体积α混合颜色）
    params.rayRadiance[idx.z][idx.y][idx.x][0]    = payload.radiance.x;
    params.rayRadiance[idx.z][idx.y][idx.x][1]    = payload.radiance.y;
    params.rayRadiance[idx.z][idx.y][idx.x][2]    = payload.radiance.z;

    // rayDensity = rayOpacity = 1 - T_final = ∑ T_j α_j 总不透明度和
    params.rayDensity[idx.z][idx.y][idx.x][0]     = 1 - payload.transmittance;

    // hitDistance：归一化加权平均深度 (∑T_iα_i t_i) / ∑T_jα_j，来自 Interaction
    params.rayHitDistance[idx.z][idx.y][idx.x][0] = payload.hitDistance;
    // lastHitDistance：这条光线遍历到的最远粒子相交t（原始未加权最大距离）
    params.rayHitDistance[idx.z][idx.y][idx.x][1] = payload.lastHitDistance;

    // hitDistanceSecondMoment：累加项 ∑ T_iα_i · t_i²，用于计算深度方差
    params.rayHitDistanceSecondMoment[idx.z][idx.y][idx.x][0] = payload.hitDistanceSecondMoment;
    // depthDistortion：方差分子 S·∑Tαt² − (∑Tαt)²，景深模糊输入
    params.rayDepthDistortion[idx.z][idx.y][idx.x][0] = payload.depthDistortion;

    #ifdef ENABLE_NORMALS
    // normal：未归一化加权累加几何法线 ∑ T_iα_i n_geo_i
    params.rayNormal[idx.z][idx.y][idx.x][0] = payload.normal.x;
    params.rayNormal[idx.z][idx.y][idx.x][1] = payload.normal.y;
    params.rayNormal[idx.z][idx.y][idx.x][2] = payload.normal.z;
    // interaction.shadingnormal：加权平均着色法线 (∑Tα n_shade)/S，已归一化
    params.rayShadingNormal[idx.z][idx.y][idx.x][0] = payload.interaction.shadingnormal.x;
    params.rayShadingNormal[idx.z][idx.y][idx.x][1] = payload.interaction.shadingnormal.y;
    params.rayShadingNormal[idx.z][idx.y][idx.x][2] = payload.interaction.shadingnormal.z;
    #endif

    // 输出归一化加权平均材质（全部除以总权重S）
    // 0/1/2 albedo RGB，3 roughness粗糙度，4 metallic金属度
    params.rayMaterial[idx.z][idx.y][idx.x][0] = payload.interaction.material.albedo.x;
    params.rayMaterial[idx.z][idx.y][idx.x][1] = payload.interaction.material.albedo.y;
    params.rayMaterial[idx.z][idx.y][idx.x][2] = payload.interaction.material.albedo.z;
    params.rayMaterial[idx.z][idx.y][idx.x][3] = payload.interaction.material.roughness;
    params.rayMaterial[idx.z][idx.y][idx.x][4] = payload.interaction.material.metallic;

    #ifdef ENABLE_HIT_COUNTS
    // 这条光线有效相交粒子总个数，用于调试、密度可视化
    params.rayHitsCount[idx.z][idx.y][idx.x][0] = payload.hitsCount;
    #endif
}

static __device__ __inline__ void writePbrOutputs(
    const uint3& idx,
    const pathPayload& payload) {
    params.rayPbr[idx.z][idx.y][idx.x][0] = payload.accumulatedLighting.x;
    params.rayPbr[idx.z][idx.y][idx.x][1] = payload.accumulatedLighting.y;
    params.rayPbr[idx.z][idx.y][idx.x][2] = payload.accumulatedLighting.z;
    params.rayLight[idx.z][idx.y][idx.x][0] = payload.accumulatedLightNoBrdf.x;
    params.rayLight[idx.z][idx.y][idx.x][1] = payload.accumulatedLightNoBrdf.y;
    params.rayLight[idx.z][idx.y][idx.x][2] = payload.accumulatedLightNoBrdf.z;

    params.rayPbrComponents[idx.z][idx.y][idx.x][0][0] = payload.accumulatedDirectLighting.x;
    params.rayPbrComponents[idx.z][idx.y][idx.x][0][1] = payload.accumulatedDirectLighting.y;
    params.rayPbrComponents[idx.z][idx.y][idx.x][0][2] = payload.accumulatedDirectLighting.z;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][0] = payload.accumulatedIndirectLighting.x;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][1] = payload.accumulatedIndirectLighting.y;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][2] = payload.accumulatedIndirectLighting.z;
}
