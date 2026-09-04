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

#include <3dgptir/environment.h>
#include <3dgptir/material.h>
#include <3dgptir/particleDensity.h>
#include <3dgptir/pipelineDefinitions.h>
#include <3dgptir/tensorAccessor.h>
#include <3dgptir/atmosphere.h>

enum AtmHitgroupSbt {
    ATM_SBT_GS   = 0,
    ATM_SBT_MESH = 1,
    ATM_SBT_VOL  = 2,
    ATM_SBT_HITGROUP_COUNT = 3
};
/**
 * @brief 渲染管线参数结构体
 * 
 * 该结构体封装了OptiX光线追踪渲染管线所需的所有参数，
 * 包括光线数据、粒子数据、材质属性、光照信息和输出缓冲。
 * 在GPU kernel中作为常量参数传递。
 */
 struct PipelineParameters {
    // ============================================================
    // 1. 光线变换参数
    // ============================================================
    
    /**
     * @brief 光线从相机空间到世界空间的变换矩阵（行主序）
     * 
     * 3x4矩阵，前三列是旋转/缩放，第四列是平移。
     * 用于将相机空间的光线原点和方向转换到世界空间。
     * 
     * 使用方式：
     *   worldOrigin = rayToWorld[0..2] * localOrigin + translation
     *   worldDirection = rayToWorld[0..2] * localDirection
     */
    float4 rayToWorld[3];

    /**
     * @brief 光线起点（相机空间）
     * 
     * 张量维度：[H, W, 1, 3] 或 [batch, H, W, 3]
     * 存储每条光线的起点位置（通常在相机空间）
     */
    PackedTensorAccessor32<float, 4> rayOrigin;

    /**
     * @brief 光线方向（相机空间）
     * 
     * 张量维度：[H, W, 1, 3]
     * 存储每条光线的方向向量（通常在相机空间）
     */
    PackedTensorAccessor32<float, 4> rayDirection;

    // ============================================================
    // 2. 粒子数据（高斯泼溅核心数据）
    // ============================================================
    
    /**
     * @brief 粒子密度数据指针
     * 
     * 包含每个高斯粒子的：
     * - position (float3): 中心位置
     * - scale (float3): 各轴缩放
     * - quaternion (float4): 旋转四元数
     * - density (float): 密度/透明度
     * 
     * 用于计算光线-粒子交点和透明度累积
     */
    const ParticleDensity* particleDensity;

    /**
     * @brief 粒子材质参数指针
     * 
     * 每个粒子的材质属性：
     * - albedo (float3): 反照率（颜色）
     * - roughness (float): 粗糙度
     * - metallic (float): 金属度
     */
    const Material* particleMaterial;

    /**
     * @brief 球谐函数系数（用于视角相关颜色）
     * 
     * 存储每个粒子的球谐系数，用于实现视角相关的颜色变化。
     * 通常使用0-3阶球谐（1+3+5+7=16个系数）
     */
    const float* particleRadiance;

    /**
     * @brief 每个粒子的着色法线
     * 
     * 用于光照计算的表面法线，可能与几何法线不同
     */
    const float* particleShadingNormal;

    /**
     * @brief 管线特定的粒子扩展数据（任意类型）
     * 
     * 用于不同管线（如自定义图元）传递额外数据
     * 使用void*允许灵活的类型转换
     */
    const void* particleExtendedData;

    /**
     * @brief 粒子可见性缓冲区
     * 
     * 记录每个粒子是否可见（用于调试或后处理）
     * int32_t数组，0=不可见，1=可见
     */
    int32_t* particleVisibility;

    /// Optional device counter for ENABLE_RAYGEN_SETUP_PROFILING (nullable).
    unsigned long long* raygenSetupCycleCounter;

    // ============================================================
    // 3. 输出缓冲区（渲染结果）
    // ============================================================
    
    /**
     * @brief 输出：累积辐射度（颜色）
     * 
     * 最终渲染图像的颜色值
     * 维度：[H, W, 1, 3] → RGB  NHWC 格式（Batch, Height, Width, Channel）?
     */
    PackedTensorAccessor32<float, 4> rayRadiance;

    /**
     * @brief 输出：累积密度（透明度）
     * 
     * 沿光线路径累积的不透明度
     * 用于合成或后处理
     */
    PackedTensorAccessor32<float, 4> rayDensity;

    /**
     * @brief 输出：命中距离
     * 
     * 光线到最近命中点的距离
     * 用于深度图或雾效
     */
    PackedTensorAccessor32<float, 4> rayHitDistance;

    /**
     * @brief 输出：命中距离的二阶矩（方差）
     * 
     * 用于计算深度分布的方差
     * 可用于去模糊或景深效果，存储的是 E[D²]（期望值），深度分布的不确定性
     */
    PackedTensorAccessor32<float, 4> rayHitDistanceSecondMoment;

    /**
     * @brief 输出：深度扭曲值
     * 
     * 用于神经渲染中的深度扭曲
     * 帮助网络学习几何一致性
     */
    PackedTensorAccessor32<float, 4> rayDepthDistortion;

    /**
     * @brief 输出：累积法线
     * 
     * 沿光线路径的几何法线加权平均
     * 用于法线贴图或边缘检测
     */
    PackedTensorAccessor32<float, 4> rayNormal;

    /**
     * @brief 输出：累积着色法线
     * 
     * 用于光照的平滑法线（可能经过插值）
     * 比几何法线更平滑
     */
    PackedTensorAccessor32<float, 4> rayShadingNormal;

    /**
     * @brief 输出：累积材质参数
     * 
     * 存储：albedo.xyz (颜色), roughness, metallic
     * 用于PBR光照计算
     */
    PackedTensorAccessor32<float, 4> rayMaterial;

    /**
     * @brief 输出：每条光线的命中次数
     * 
     * 仅在AnyHit管线中使用
     * 用于统计或调试
     */
    PackedTensorAccessor32<float, 4> rayHitsCount;

    /**
     * @brief 输出：累积PBR辐射度
     * 
     * 物理渲染的最终颜色
     * 包含直接光照和间接光照
     */
    PackedTensorAccessor32<float, 4> rayPbr;

    /**
     * @brief 输出：环境光照（无BRDF权重）
     * 
     * 环境光贡献，未乘以BRDF
     * 用于调试或单独合成
     */
    PackedTensorAccessor32<float, 4> rayLight;

    /**
     * @brief 输出：PBR分量分解
     * 
     * 维度：[H, W, 1, 5]
     * 存储直接光照和间接光照的分离值
     * 用于后处理或调试
     */
    PackedTensorAccessor32<float, 5> rayPbrComponents;

    // ============================================================
    // 4. 光照系统
    // ============================================================
    
    /**
     * @brief 环境光照参数
     * 
     * 包含环境贴图或环境光颜色/强度
     */
    Environment environment;

    /**
     * @brief 场景中的光源数量
     */
    unsigned int numLights;

    /**
     * @brief 光源数据 [N, 9]
     * 
     * 每个光源9个参数：
     * - type (0=点光源, 1=方向光, 2=聚光灯, 3=面光源)
     * - 8个参数（位置、方向、强度、角度等）
     */
    PackedTensorAccessor32<float, 2> lights;

    /**
     * @brief 光源条目数（用于采样）
     */
    unsigned int numLightEntries;

    /**
     * @brief 光源别名表 [5, N]
     * 
     * 用于重要性采样的别名方法
     * 加速光源选择
     */
    PackedTensorAccessor32<float, 2> lightAliasTable;

    /**
     * @brief 网格光源的顶点数
     */
    unsigned int numMeshLightVertices;

    /**
     * @brief 网格光源顶点数据 [V, 3]
     * 
     * 面光源的三角形顶点位置
     */
    PackedTensorAccessor32<float, 2> meshLightVertices;

    /**
     * @brief 网格光源的三角形数
     */
    unsigned int numMeshLightTriangles;

    /**
     * @brief 网格光源三角形索引 [T, 3]
     * 
     * 引用meshLightVertices中的顶点
     */
    PackedTensorAccessor32<int32_t, 2> meshLightTriangles;

    /**
     * @brief 网格光源数
     */
    unsigned int numMeshLights;

    /**
     * @brief 网格光源参数 [M, 8]
     * 
     * 每个网格光源的强度、颜色、变换等
     */
    PackedTensorAccessor32<float, 2> meshLights;

    /**
     * @brief 网格光源三角形别名表 [3, T]
     * 
     * 用于三角形面积重要性采样
     */
    PackedTensorAccessor32<float, 2> meshLightTriangleAliasTable;

    // ============================================================
    // 5. OptiX加速结构
    // ============================================================
    
    /**
     * @brief OptiX可遍历句柄
     * 
     * 指向GAS（底层加速结构）或IAS（实例加速结构）
     * 光线追踪的入口点
     */
    OptixTraversableHandle handle;

    /// Triangle GAS for AtmMesh (DISABLE_ANYHIT). 0 = none.
    OptixTraversableHandle meshHandle;
    /// Custom GAS: one AABB per non-empty voxel brick. 0 = none.
    OptixTraversableHandle volHandle;
    /// IAS of mesh+vol for one closest-event trace. 0 = sequential GAS traces.
    OptixTraversableHandle meshVolIas;
    /// Device int32[volPrimCount]: primitive → linear brick index.
    const int* volBrickId;
    int volPrimCount;

    /**
     * @brief 场景的AABB包围盒
     * 
     * 整个场景的边界框
     * 用于光线-场景相交测试的快速剔除
     */
    OptixAabb aabb;

    // ============================================================
    // 6. 渲染参数
    // ============================================================
    
    /**
     * @brief 最小透射率截断值
     * 
     * 当累积透射率低于此值时停止追踪
     * 优化性能，避免无效计算
     * 典型值：0.001 ~ 0.01
     */
    float minTransmittance;

    /**
     * @brief 命中的最小高斯响应阈值
     * 
     * 只考虑高斯密度 > hitMinGaussianResponse 的命中
     * 剔除低密度区域，加速渲染
     */
    float hitMinGaussianResponse;

    /**
     * @brief 粒子最大命中距离平方（固定为9.0）
     * 
     * 超出此距离的粒子被忽略
     * 9.0对应3个单位距离的截断
     * 用于快速剔除远处粒子
     */
    static constexpr float hitMaxParticleSquaredDistance = 9.f;

    /**
     * @brief Alpha混合的最小阈值
     * 
     * 透明度低于此值的粒子被忽略
     * 用于剔除近乎透明的粒子
     */
    float alphaMinThreshold;

    /**
     * @brief 球谐函数的阶数
     * 
     * 0阶：仅漫反射
     * 1阶：漫反射+简单视角变化
     * 2阶：更丰富的视角变化
     * 3阶：最高质量
     */
    unsigned int sphDegree;

    /**
     * @brief 最大反弹次数（路径追踪）
     * 
     * 限制光线追踪的递归深度
     * 影响质量和性能的平衡
     */
    unsigned int maxBounces;

    /**
     * @brief 渲染选项标志位
     * 
     * 控制各种渲染特性：
     * - 是否启用自适应核钳制
     * - 是否启用环境光
     * - 是否启用阴影等
     */
    unsigned int renderOpts;

    // ============================================================
    // Atmosphere (voxel cloud + 1D gas). enabled=0 keeps original PTIR.
    // Cloud k is voxels only; Gaussians remain surface primitives.
    // ============================================================
    AtmParams atmosphere;

    /* htrdr -R: repeating mesh tile AABB in world XY (and Z slab). Zero extent = no wrap. */
    float meshRepeatLow[3];
    float meshRepeatUpp[3];

    // ============================================================
    // 7. 帧信息
    // ============================================================
    
    /**
     * @brief 帧范围 [min_x, min_y, max_x, max_y]
     * 
     * 用于分块渲染或多帧累积
     * 只渲染图像的一部分
     */
    uint2 frameBounds;

    /**
     * @brief 当前帧号
     * 
     * 用于时间相关的效果（如TAA、降噪）
     */
    unsigned int frameNumber;

    /**
     * @brief 每个粒子的图元三角形数量
     * 
     * 根据gPrimType不同：
     * - 四面体：4
     * - 八面体：8
     * - 二十面体：20
     * - 球体：1（特殊）
     */
    int gPrimNumTri;

    /**
     * @brief 每条光线最大命中数（固定为16）
     * 
     * 限制单个光线可以命中的粒子数量
     * 防止无限循环或超出内存
     */
    static constexpr unsigned int MaxNumHitPerTrace = 16;

    // ============================================================
    // 8. 编译时常量（根据宏定义）
    // ============================================================
    
#ifdef PARTICLE_PRIMITIVE_TYPE
    /**
     * @brief 是否为自定义图元模式
     * 
     * 仅使用AABB，不存储顶点/三角形
     * 最快但质量最低
     */
    static constexpr bool CustomPrimitive = (PARTICLE_PRIMITIVE_TYPE == MOGPrimitiveTypes::MOGTracingCustom);

    /**
     * @brief 是否为实例化模式
     * 
     * 每个粒子作为独立实例
     * 支持变换矩阵
     */
    static constexpr bool InstancePrimitive = (PARTICLE_PRIMITIVE_TYPE == MOGPrimitiveTypes::MOGTracingInstances);

    /**
     * @brief 是否为曲面片模式
     * 
     * 使用2个三角形表示粒子
     * 包含额外粒子数据
     */
    static constexpr bool SurfelPrimitive = (PARTICLE_PRIMITIVE_TYPE == MOGPrimitiveTypes::MOGTracingTriSurfel);

    /**
     * @brief 是否启用钳制
     * 
     * 防止密度过大导致数值不稳定
     */
    static constexpr bool ClampedPrimitive = PARTICLE_PRIMITIVE_CLAMPED;
#endif

#ifdef PARTICLE_KERNEL_DEGREE
    /**
     * @brief 粒子核函数带宽（编译时常量）
     * 
     * 控制高斯函数的扩散范围
     * 值越大，粒子影响范围越广
     * 影响渲染质量和性能
     */
    static constexpr int ParticleKernelDegree = PARTICLE_KERNEL_DEGREE;
#endif

    // ============================================================
    // 9. 辅助函数（仅CUDA设备端）
    // ============================================================
    
#ifdef __CUDACC__
    /**
     * @brief 获取光线在世界空间的原点
     * 
     * @param idx 像素坐标 (x, y, z=0)
     * @return float3 世界空间的光线起点
     * 
     * 应用rayToWorld变换矩阵到相机空间的原点
     */
    inline __device__ float3 rayWorldOrigin(const uint3& idx) const {
        float3 origin = make_float3(rayOrigin[idx.z][idx.y][idx.x][0],
                                    rayOrigin[idx.z][idx.y][idx.x][1],
                                    rayOrigin[idx.z][idx.y][idx.x][2]);
        return make_float3(
            rayToWorld[0].x * origin.x + rayToWorld[0].y * origin.y + rayToWorld[0].z * origin.z + rayToWorld[0].w,
            rayToWorld[1].x * origin.x + rayToWorld[1].y * origin.y + rayToWorld[1].z * origin.z + rayToWorld[1].w,
            rayToWorld[2].x * origin.x + rayToWorld[2].y * origin.y + rayToWorld[2].z * origin.z + rayToWorld[2].w);
    }

    /**
     * @brief 获取光线在世界空间的方向
     * 
     * @param idx 像素坐标 (x, y, z=0)
     * @return float3 世界空间的光线方向（已归一化）
     * 
     * 应用rayToWorld的旋转部分到相机空间的方向
     * 注意：不包括平移（方向不受平移影响）
     */
    inline __device__ float3 rayWorldDirection(const uint3& idx) const {
        float3 direction = make_float3(rayDirection[idx.z][idx.y][idx.x][0],
                                       rayDirection[idx.z][idx.y][idx.x][1],
                                       rayDirection[idx.z][idx.y][idx.x][2]);
        return make_float3(
            rayToWorld[0].x * direction.x + rayToWorld[0].y * direction.y + rayToWorld[0].z * direction.z,
            rayToWorld[1].x * direction.x + rayToWorld[1].y * direction.y + rayToWorld[1].z * direction.z,
            rayToWorld[2].x * direction.x + rayToWorld[2].y * direction.y + rayToWorld[2].z * direction.z);
    }
#endif
};

/**
 * @brief 渲染管线反向传播参数结构体
 * 
 * 该结构体继承自 PipelineParameters，额外包含所有输出张量的梯度缓冲区。
 * 用于深度学习训练中的反向传播，计算损失函数对各渲染输出的梯度，
 * 进而反传回粒子参数（位置、缩放、旋转、密度、球谐系数等）。
 * 
 * 继承关系：PipelineBackwardParameters : PipelineParameters
 * 使用场景：3D高斯泼溅（3DGS）训练中的反向传播阶段
 */
 struct PipelineBackwardParameters : PipelineParameters {
    // ============================================================
    // 1. 图像空间输出梯度（Loss 对渲染图像的梯度）
    //    这些梯度由损失函数（如 L1 Loss、SSIM Loss）计算得到
    // ============================================================
    
    /**
     * @brief 输出：累积辐射度（颜色）的梯度
     * 
     * 维度：[1, H, W, 3]
     * 含义：∂Loss / ∂rayRadiance
     * 
     * 来源：损失函数对最终渲染图像 RGB 的偏导
     * 用途：反传给粒子颜色/球谐系数
     */
    PackedTensorAccessor32<float, 4> rayRadianceGrad;

    /**
     * @brief 输出：累积密度的梯度
     * 
     * 维度：[1, H, W, 1]
     * 含义：∂Loss / ∂rayDensity
     * 
     * 用途：用于密度正则化或透明度相关的损失
     */
    PackedTensorAccessor32<float, 4> rayDensityGrad;

    /**
     * @brief 输出：命中距离的梯度
     * 
     * 维度：[1, H, W, 1]
     * 含义：∂Loss / ∂rayHitDistance
     * 
     * 用途：深度图相关的损失（如深度监督、深度平滑）
     */
    PackedTensorAccessor32<float, 4> rayHitDistanceGrad;

    /**
     * @brief 输出：命中距离二阶矩（方差）的梯度
     * 
     * 维度：[1, H, W, 1]
     * 含义：∂Loss / ∂rayHitDistanceSecondMoment
     * 
     * 用途：不确定性损失、景深相关的梯度
     */
    PackedTensorAccessor32<float, 4> rayHitDistanceSecondMomentGrad;

    /**
     * @brief 输出：深度扭曲值的梯度
     * 
     * 维度：[1, H, W, 1]
     * 含义：∂Loss / ∂rayDepthDistortion
     * 
     * 用途：神经渲染中的深度一致性损失
     */
    PackedTensorAccessor32<float, 4> rayDepthDistortionGrad;

    /**
     * @brief 输出：累积法线的梯度
     * 
     * 维度：[1, H, W, 3]
     * 含义：∂Loss / ∂rayNormal
     * 
     * 用途：法线相关的损失（法线平滑、法线监督）
     */
    PackedTensorAccessor32<float, 4> rayNormalGrad;

    /**
     * @brief 输出：累积着色法线的梯度
     * 
     * 维度：[1, H, W, 3]
     * 含义：∂Loss / ∂rayShadingNormal
     * 
     * 用途：光照一致性损失、法线正则化
     */
    PackedTensorAccessor32<float, 4> rayShadingNormalGrad;

    /**
     * @brief 输出：累积材质参数的梯度
     * 
     * 维度：[1, H, W, 5]
     * 含义：∂Loss / ∂rayMaterial
     * 结构：[albedo.r, albedo.g, albedo.b, roughness, metallic]
     * 
     * 用途：材质参数的反向传播，用于训练 PBR 材质
     */
    PackedTensorAccessor32<float, 4> rayMaterialGrad;

    /**
     * @brief 输出：PBR 辐射度的梯度
     * 
     * 维度：[1, H, W, 3]
     * 含义：∂Loss / ∂rayPbr
     * 
     * 用途：物理渲染的梯度传播
     */
    PackedTensorAccessor32<float, 4> rayPbrGrad;

    /**
     * @brief 输出：环境光照的梯度（无 BRDF 权重）
     * 
     * 维度：[1, H, W, 3]
     * 含义：∂Loss / ∂rayLight
     * 
     * 用途：环境光相关的梯度（如环境光去噪）
     */
    PackedTensorAccessor32<float, 4> rayLightGrad;

    /**
     * @brief 输出：环境贴图的梯度（可选）
     * 
     * 维度：[H, W, 3] 或 [1, H, W, 3]
     * 含义：∂Loss / ∂environment
     * 
     * 用途：环境光照优化（可训练的环境贴图）
     * 注意：这是 3 维张量，不是 4 维（没有 Batch 维度）
     */
    PackedTensorAccessor32<float, 3> environmentGrad;

    // ============================================================
    // 2. 粒子参数梯度（需要反传给优化器，用于更新粒子）
    //    这些是训练的目标：通过梯度下降优化粒子属性
    // ============================================================
    
    /**
     * @brief 粒子密度数据的梯度
     * 
     * 结构体包含：
     * - position (float3): 粒子位置梯度
     * - scale (float3): 缩放梯度
     * - quaternion (float4): 旋转四元数梯度
     * - density (float): 密度梯度
     * 
     * 维度：[N]（每个粒子一份）
     * 
     * 这是最重要的梯度之一，直接控制粒子的几何形状和透明度
     */
    ParticleDensity* particleDensityGrad;

    /**
     * @brief 粒子材质参数的梯度
     * 
     * 结构体包含：
     * - albedo (float3): 反照率梯度
     * - roughness (float): 粗糙度梯度
     * - metallic (float): 金属度梯度
     * 
     * 维度：[N]（每个粒子一份）
     * 
     * 用于优化粒子的 PBR 材质属性
     */
    Material* particleMaterialGrad;

    /**
     * @brief 球谐函数系数的梯度
     * 
     * 存储每个粒子的球谐系数梯度
     * 球谐系数决定了粒子的视角相关颜色
     * 
     * 维度：[N, C]（C 为球谐系数数量，如 16 或 25）
     * 
     * 这是 3DGS 训练中最关键的梯度之一
     */
    float* particleRadianceGrad;

    /**
     * @brief 粒子着色法线的梯度
     * 
     * 存储每个粒子的着色法线梯度
     * 
     * 维度：[N, 3]（每个粒子一个 float3）
     * 
     * 用于优化光照计算中的法线方向
     */
    float* particleShadingNormalGrad;
};