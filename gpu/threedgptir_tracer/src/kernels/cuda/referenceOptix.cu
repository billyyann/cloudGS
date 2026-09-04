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

#define ATM_KERNEL_REV 34
#include <3dgptir/pipelineParameters.h>
#include <3dgptir/kernels/cuda/sampler.cuh>
// clang-format on

extern "C" {
__constant__ PipelineParameters params;
}

#include <3dgptir/kernels/cuda/environment.cuh>
#include <3dgptir/kernels/cuda/pathTracer.cuh>
#include <3dgptir/kernels/cuda/atmospherePathTracer.cuh>
/* NVRTC: ATM_KERNEL_REV 34 gas scatter keeps Rayleigh NEE (no ks_tot break). */

// Ablation early-return stages (only with ENABLE_RAYGEN_SETUP_PROFILING).
// Official release: do not define ENABLE_RAYGEN_SETUP_PROFILING.
//   1 = after primary rayIntersect
//   2 = after writePrimaryRayOutputs
//   3 = after sampleNee (skip bounce loop + writePbr)
#ifdef ENABLE_RAYGEN_SETUP_PROFILING
#ifndef RAYGEN_ABLATION_STAGE
#define RAYGEN_ABLATION_STAGE 1
#endif
#endif

/**
 * @brief OptiX 光线生成程序（Ray Generation Program）
 * 
 * 这是整个渲染管线的入口点，每个像素/线程执行一次。
 * 负责：
 * 1. 生成主光线
 * 2. 执行首次相交检测
 * 3. 输出传统渲染结果（3DGS α混合）
 * 4. 进行路径追踪（间接光照）
 * 5. 输出 PBR 渲染结果
 */
 extern "C" __global__ void __raygen__rg() {
    // ============================================================
    // 第 1 行：获取当前线程的启动索引
    // ============================================================
    // optixGetLaunchIndex() 返回当前线程在 3D 网格中的坐标
    // idx.x = 列（图像宽度方向）
    // idx.y = 行（图像高度方向）  
    // idx.z = 批次索引（通常为 0）
    // optixGetLaunchIndex() 是 OptiX 原生函数，用于获取当前着色器程序（如 Ray Generation Program）线程的三维启动索引 (x, y, z)。
#ifdef ENABLE_RAYGEN_SETUP_PROFILING
    const unsigned long long setupT0 = clock64();
#endif
    const uint3 idx = optixGetLaunchIndex();
    
    // ============================================================
    // 第 2-4 行：边界检查
    // ============================================================
    // 如果当前像素坐标超出有效帧范围，直接返回
    // frameBounds.x = 图像最大宽度，frameBounds.y = 图像最大高度
    // 这用于处理分块渲染或部分渲染的情况
    if ((idx.x > params.frameBounds.x) || (idx.y > params.frameBounds.y)) {
#ifdef ENABLE_RAYGEN_SETUP_PROFILING
        if (params.raygenSetupCycleCounter != nullptr) {
            atomicAdd(params.raygenSetupCycleCounter, clock64() - setupT0);
        }
#endif
        return;
    }

    // ============================================================
    // 第 6 行：创建光线对象
    // ============================================================
    // params.rayWorldOrigin(idx) ：
    //   根据像素坐标获取世界空间的光线起点（应用了相机变换）
    // params.rayWorldDirection(idx) ：
    //   获取世界空间的光线方向
    // Ray 类封装了光线的 origin, direction, tmin, tmax
    Ray ray(params.rayWorldOrigin(idx), params.rayWorldDirection(idx));
    
    // ============================================================
    // 第 7-8 行：初始化采样器
    // ============================================================
    // Sampler 用于生成随机数，用于：
    // - 蒙特卡洛积分（路径追踪）
    // - 重要性采样（BRDF 采样）
    // initFromLaunch 使用像素坐标和帧号初始化随机种子
    Sampler sampler;
    sampler.initFromLaunch(idx, params.frameNumber);
#ifdef ENABLE_RAYGEN_SETUP_PROFILING
    if (params.raygenSetupCycleCounter != nullptr) {
        atomicAdd(params.raygenSetupCycleCounter, clock64() - setupT0);
    }
#endif

    if (params.atmosphere.enabled) {
        if (ATM_KERNEL_REV < 0) {
            return;
        }
        /* rev19: mesh Triangle GAS + vol Custom GAS + mesh/vol IAS */
        traceAtmospherePath(idx, sampler);
        return;
    }

    // ============================================================
    // 第 10 行：初始化路径追踪状态
    // ============================================================
    // pathPayload 结构体包含：
    //   - pathThroughput: 路径吞吐量（累积权重），初始为 1.0
    //   - numBounces: 当前反弹次数，初始为 0
    //   - maxBounces: 最大反弹次数
    //   - active: 路径是否活跃
    //   - currentRayPayload: 当前光线的命中数据
    pathPayload path(1u, 0u, params.maxBounces);

    // ============================================================
    // 第 12-13 行：执行首次光线追踪

    rayIntersect<false>(ray, path.currentRayPayload, sampler);

#if defined(ENABLE_RAYGEN_SETUP_PROFILING) && (RAYGEN_ABLATION_STAGE <= 1)
    return;
#endif
    
    // ============================================================
    // 第 14-16 行：写入传统渲染输出
    // ============================================================

    writePrimaryRayOutputs(idx, path.currentRayPayload);

#if defined(ENABLE_RAYGEN_SETUP_PROFILING) && (RAYGEN_ABLATION_STAGE <= 2)
    return;
#endif

    // ============================================================
    // 第 18-20 行：早期返回（仅当未启用光源可视化）
    // ============================================================
#ifndef ENABLE_VISUALIZE_LIGHTS
    if (!path.currentRayPayload.interaction.valid) {
        return;
    }
#endif

    // ============================================================
    // 第 22-24 行：下一事件估计（NEE）
    // ============================================================
    // 如果宏 ENABLE_MIS 已定义：
#ifdef ENABLE_MIS
    //场景光源列表（点光 / 平行光 / 面光 / 环境光）→ 按光源 PDF 随机选一盏光源
    sampleNee(path, sampler);
#endif

#if defined(ENABLE_RAYGEN_SETUP_PROFILING) && (RAYGEN_ABLATION_STAGE <= 3)
    // maxBounces=0 ablation: stop before bounce loop / writePbr.
    return;
#endif

    // ============================================================
    // 第 26-44 行：路径追踪主循环（间接光照）
    // ============================================================
    // 循环条件：
    //   - depth < params.maxBounces: 未达到最大反弹次数
    //   - path.active: 路径仍然活跃
    for (unsigned int depth = 0; depth < params.maxBounces && path.active; ++depth) {
        
        // ---- 第 28 行：累积光照贡献 ----
        // accumulateLightContribution ：
        //   将当前命中点的光照贡献累加到路径中 - 直接光照（来自光源）- 环境光照（来自环境贴图）- 自发光
        accumulateLightContribution(path);

        // ---- 第 30-33 行：检查是否继续追踪 ----
        // 更新 path.active：
        path.active &= (depth + 1u < path.maxBounces) && 
                       path.currentRayPayload.interaction.valid;
        if (!path.active) {
            break;  // 路径终止，跳出循环
        }
        
        // ---- 第 34 行：更新反弹计数 ----
        path.numBounces = depth + 1u;

        // ---- 第 36-37 行：采样 BRDF 方向 ----
        // sampleBrdfNextDirection ：
        sampleBrdfNextDirection(path, sampler);
        
        // ---- 第 38-41 行：吞吐量检查（俄罗斯轮盘赌） ----
        // 计算路径吞吐量的最大值（RGB 中的最大值）
        const float throughputMax = fmaxf(path.pathThroughput.x, 
                                          fmaxf(path.pathThroughput.y, 
                                                path.pathThroughput.z));

        if (throughputMax < 1e-4f) {
            break;
        }
        
        // ---- 第 42 行：发射下一跳光线 ----
        // rayIntersect<true> ：
        //  EnableSelfOcclusionRejection（是否剔除自遮挡）
        rayIntersect<true>(path.currentRayPayload.ray, 
                           path.currentRayPayload, 
                           sampler);
    }

    // ============================================================
    // 第 46 行：写入 PBR 渲染输出
    // ============================================================
    // writePbrOutputs 写入：
    writePbrOutputs(idx, path);
}

extern "C" __global__ void __intersection__is() {
    float hitDistance;
    const bool intersect = PipelineParameters::InstancePrimitive ? intersectInstanceParticle(optixGetObjectRayOrigin(),
                                                                                             optixGetObjectRayDirection(),
                                                                                             optixGetInstanceIndex(),
                                                                                             optixGetRayTmin(),
                                                                                             optixGetRayTmax(),
                                                                                             params.hitMaxParticleSquaredDistance,
                                                                                             hitDistance)
                                                                 : intersectCustomParticle(optixGetWorldRayOrigin(),
                                                                                           optixGetWorldRayDirection(),
                                                                                           optixGetPrimitiveIndex(),
                                                                                           params.particleDensity,
                                                                                           optixGetRayTmin(),
                                                                                           optixGetRayTmax(),
                                                                                           params.hitMaxParticleSquaredDistance,
                                                                                           hitDistance);
    if (intersect) {
        optixReportIntersection(hitDistance, 0);
    }
}

#define compareAndSwapHitPayloadValue(hit, i_id, i_distance)                      \
    {                                                                             \
        const float distance = __uint_as_float(optixGetPayload_##i_distance());   \
        if (hit.distance < distance) {                                            \
            optixSetPayload_##i_distance(__float_as_uint(hit.distance));          \
            const uint32_t id = optixGetPayload_##i_id();                         \
            optixSetPayload_##i_id(hit.particleId);                               \
            hit.distance   = distance;                                            \
            hit.particleId = id;                                                  \
        }                                                                         \
    }

extern "C" __global__ void __anyhit__ah() {
    RayHit hit = RayHit{optixPrimitiveIndex(), optixGetRayTmax()};

    if (hit.distance < __uint_as_float(optixGetPayload_31())) {
        compareAndSwapHitPayloadValue(hit, 0, 1);
        compareAndSwapHitPayloadValue(hit, 2, 3);
        compareAndSwapHitPayloadValue(hit, 4, 5);
        compareAndSwapHitPayloadValue(hit, 6, 7);
        compareAndSwapHitPayloadValue(hit, 8, 9);
        compareAndSwapHitPayloadValue(hit, 10, 11);
        compareAndSwapHitPayloadValue(hit, 12, 13);
        compareAndSwapHitPayloadValue(hit, 14, 15);
        compareAndSwapHitPayloadValue(hit, 16, 17);
        compareAndSwapHitPayloadValue(hit, 18, 19);
        compareAndSwapHitPayloadValue(hit, 20, 21);
        compareAndSwapHitPayloadValue(hit, 22, 23);
        compareAndSwapHitPayloadValue(hit, 24, 25);
        compareAndSwapHitPayloadValue(hit, 26, 27);
        compareAndSwapHitPayloadValue(hit, 28, 29);
        compareAndSwapHitPayloadValue(hit, 30, 31);

        // ignore all inserted hits, expect if the last one
        if (__uint_as_float(optixGetPayload_31()) > optixGetRayTmax()) {
            optixIgnoreIntersection();
        }
    }
}

/* Mesh Triangle GAS closest-hit. Payload: t, kind=1, primId. */
extern "C" __global__ void __closesthit__mesh() {
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(1u);
    optixSetPayload_2(optixGetPrimitiveIndex());
}

/* Volume Custom GAS: one grid AABB; sequential Woodcock with a single Ts. */
extern "C" __global__ void __intersection__vol() {
    const float3 o3 = optixGetObjectRayOrigin();
    const float3 d3 = optixGetObjectRayDirection();
    const float org[3] = {o3.x, o3.y, o3.z};
    const float dir[3] = {d3.x, d3.y, d3.z};
    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();
    const unsigned seedLo = optixGetPayload_5();
    const unsigned seedHi = optixGetPayload_6();
    unsigned long long seed =
        (static_cast<unsigned long long>(seedHi) << 32ull) | static_cast<unsigned long long>(seedLo);
    Sampler rng;
    rng.init(seed ? seed : 1ull, 1ull);
    const int walk_ks = params.atmosphere.spectral_mode != ATM_SPECTRAL_LW;
    const float omega_arg = params.atmosphere.mie_omega > 0.f ? params.atmosphere.mie_omega : -1.f;
    /* Competing risks in IS: voxel Z-slab Woodcock + 1D gas over the full ray
     * interval. AABB Z must cover the gas column so sky rays are not clipped
     * at the cloud top (that miss used to skip Rayleigh entirely). */
    AtmHit vox;
    AtmHit gas;
    atm_woodcock_voxels_into(
        &params.atmosphere.voxels, org, dir, tmax, rng, omega_arg, walk_ks, &vox);
    atm_gas_free_path_into(
        &params.atmosphere.gas, org, dir, tmax, rng, walk_ks, &gas);
    AtmHit hit;
    atm_hit_clear(&hit);
    if (gas.event != ATM_EVENT_NONE && gas.t <= vox.t && gas.t >= tmin && gas.t <= tmax) {
        hit = gas;
    } else if (vox.event != ATM_EVENT_NONE && vox.t >= tmin && vox.t <= tmax) {
        hit = vox;
    } else {
        return;
    }
    optixReportIntersection(
        hit.t,
        0u,
        static_cast<unsigned>(hit.event),
        __float_as_uint(hit.temperature));
}

extern "C" __global__ void __closesthit__vol() {
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(2u);
    optixSetPayload_2(optixGetPrimitiveIndex());
    optixSetPayload_3(optixGetAttribute_0());
    optixSetPayload_4(optixGetAttribute_1());
}
