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

#include <3dgptir/pipelineDefinitions.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <string>

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData {
    // No data needed
};
typedef SbtRecord<RayGenData> RayGenSbtRecord;

struct MissData {
    // No data needed
};
typedef SbtRecord<MissData> MissSbtRecord;

struct HitGroupData {
    // No data needed
};

typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

enum CreatePipelineFlags {
    PipelineFlag_None       = 0,
    PipelineFlag_HasCH      = 1 << 0,
    PipelineFlag_HasIS      = 1 << 1,
    PipelineFlag_HasAH      = 1 << 2,
    PipelineFlag_HasMS      = 1 << 3,
    PipelineFlag_HasRG      = 1 << 4,
    PipelineFlag_SpherePrim = 1 << 5,
    PipelineFlag_AtmAccel   = 1 << 6  // mesh CH + volume IS; GAS or single-level IAS
};

class OptixTracer {
    //IAS = Instance Acceleration Structure  (顶层加速结构)
    //GAS = Geometry Acceleration Structure   (几何加速结构)
protected:
    // 定义状态结构体，用于管理整个OptiX渲染引擎的核心数据
    struct State {
        // -------- OptiX 核心上下文和加速结构（BLAS/GAS） --------
        OptixDeviceContext context;          // OptiX设备上下文句柄，管理GPU状态、内存分配和所有操作
        OptixTraversableHandle gasHandle;    // GAS（底层加速结构）的句柄，用于光线与高斯图元求交
        CUdeviceptr gasBuffer;               // GPU显存指针，存储GAS数据的缓冲区
        size_t gasBufferSz;                  // gasBuffer缓冲区的大小（字节数）
        CUdeviceptr gasBufferTmp;            // GPU显存指针，构建GAS时所需的临时工作空间缓冲区
        size_t gasBufferTmpSz;               // gasBufferTmp临时缓冲区的大小（字节数）
        OptixAabb gasAABB;                   // GAS的世界空间轴对齐包围盒（AABB），用于视锥体剔除

        OptixTraversableHandle meshGasHandle;
        CUdeviceptr meshGasBuffer;
        size_t meshGasBufferSz;
        int meshGasNtri;
        int meshGasNvert;
        CUdeviceptr meshGasVertsPtr;
        CUdeviceptr meshGasIdxPtr;

        OptixTraversableHandle volGasHandle;
        CUdeviceptr volGasBuffer;
        size_t volGasBufferSz;
        CUdeviceptr volAabbBuffer;
        size_t volAabbBufferSz;
        CUdeviceptr volBrickIdBuffer;
        size_t volBrickIdBufferSz;
        int volPrimCount;
        CUdeviceptr volKmaxPtr;
        int volAabbWrap;

        OptixTraversableHandle meshVolIasHandle;
        CUdeviceptr meshVolIasBuffer;
        size_t meshVolIasBufferSz;
        CUdeviceptr meshVolIasInstances;
        size_t meshVolIasInstancesSz;

        CUdeviceptr optixAabbPtr;            // GPU显存指针，指向AABB数据的设备端地址
        CUdeviceptr paramsDevice;            // GPU显存指针，指向运行时参数结构体（如相机矩阵、图像尺寸等）
        size_t paramsDeviceSz;               // paramsDevice参数缓冲区的大小（字节数）

        // -------- 高斯粒子渲染参数（3D Gaussian Splatting） --------
        float particleKernelDegree;          // 粒子核函数的带宽/平滑度参数，控制高斯衰减范围
        float particleKernelMinResponse;     // 粒子核函数的最小响应阈值，低于此值忽略该粒子（加速渲染）
        bool particleKernelDensityClamping;  // 是否启用密度钳制，防止密度过大导致数值不稳定
        int particleRadianceSphDegree;       // 球谐函数（SH）的阶数，用于表示方向相关的颜色/辐射度

        // -------- 高斯图元几何数据（自定义3D图元表示高斯椭球） --------
        uint32_t gNum;                       // 当前场景中的高斯粒子总数
        uint32_t gPrimType;                  // 图元类型：0表示八面体（octahedron），1表示四面体（tetrahedron）
        uint32_t gPrimNumVert;               // 每个高斯图元包含的顶点数量（取决于图元类型）
        uint32_t gPrimNumTri;                // 每个高斯图元包含的三角形数量（取决于图元类型）
        
        CUdeviceptr gPrimVrt;                // GPU显存指针，存储所有高斯图元的顶点坐标数据（float3数组）
        size_t gPrimVrtSz;                   // gPrimVrt顶点缓冲区的大小（字节数）
        CUdeviceptr gPrimTri;                // GPU显存指针，存储三角形索引数据（每个三角形的三个顶点索引）
        size_t gPrimTriSz;                   // gPrimTri三角形索引缓冲区的大小（字节数）
        CUdeviceptr gPrimAABB;               // GPU显存指针，存储每个高斯粒子的独立AABB包围盒（用于快速剔除）
        size_t gPrimAABBSz;                  // gPrimAABB包围盒缓冲区的大小（字节数）
        CUdeviceptr iasBuffer;               // GPU显存指针，存储IAS（顶层加速结构）的缓冲区
        size_t iasBufferSz;                  // iasBuffer顶层加速结构缓冲区的大小（字节数）
        CUdeviceptr gPipelineParticleData;   // GPU显存指针，存储管线特定的粒子附加数据（颜色、协方差、不透明度等）
        size_t gPipelineParticleDataSz;      // gPipelineParticleData粒子数据缓冲区的大小（字节数）

        // -------- 前向渲染管线（Forward Rendering，用于最终画面生成） --------
        OptixPipeline pipelineTracingFwd;    // 前向光线追踪管线句柄，从相机发射光线计算最终像素颜色
        OptixShaderBindingTable sbtTracingFwd; // 前向管线的着色器绑定表（SBT），包含光线生成、命中、未命中程序的入口
        OptixModule moduleTracingFwd;        // 前向管线的编译后OptiX模块（PTX代码），包含所有着色器程序
        // -------- 后向渲染管线（Backward Rendering，用于可微渲染和训练） --------
        OptixPipeline pipelineTracingBwd;    // 后向光线追踪管线句柄，用于反向传播计算梯度（训练阶段）
        OptixShaderBindingTable sbtTracingBwd; // 后向管线的着色器绑定表（SBT）
        OptixModule moduleTracingBwd;        // 后向管线的编译后OptiX模块（PTX代码）

        // Host-side CUDA Event timing of the last forward optixLaunch (__raygen__rg).
        // Populated only when PTIR_PROFILE_RAYGEN=1.
        float lastOptixLaunchMs;
        // ENABLE_RAYGEN_SETUP_PROFILING: device cycle counter + converted mean setup ms.
        bool enableRaygenSetupProfiling;
        CUdeviceptr raygenSetupCycleCounter;
        unsigned long long lastRaygenSetupCycles;
        float lastRaygenSetupMs;
        int smClockRateKHz;
    }* _state;  // 声明指向State结构体的指针 _state，通常在堆上动态分配（cudaMallocManaged或new）

    std::vector<std::string> generateDefines(
        float particleKernelDegree,              // 高斯核函数的带宽/平滑度参数
        bool particleKernelDensityClamping,      // 是否启用密度钳制（防止数值溢出）
        int particleRadianceSphDegree,           // 球谐函数（SH）阶数（0-3，控制颜色精度）
        bool enableNormals,                      // 是否启用法线计算（用于光照）
        bool enableHitCounts,                    // 是否统计命中次数（用于调试/性能分析）
        bool enableMIS,                          // 是否启用多重重要性采样（Multiple Importance Sampling）
        bool enableMetallic,                     // 是否启用金属材质效果（PBR渲染）
        bool enableVisualizeLights,              // 是否开启光源可视化调试模式
        bool enableRaygenSetupProfiling,         // ENABLE_RAYGEN_SETUP_PROFILING（正式发版关闭）
        int raygenAblationStage);                // RAYGEN_ABLATION_STAGE 1/2/3
    //创建 OptiX 管线
    void createPipeline(
            const OptixDeviceContext context,        // OptiX 设备上下文句柄（CPU端的"遥控器"）
            const std::string& path,                 // 主 CUDA 源码文件路径（.cu 文件）
            const std::string& dependencies_path,    // 依赖文件路径（头文件、辅助模块）
            const std::string& cuda_path,            // CUDA Toolkit 安装路径（用于 nvcc 编译器）
            const std::vector<std::string>& defines, // 预处理器宏定义（由 generateDefines 生成）
            const std::string& kernel_name,          // 内核入口函数名（如 "__raygen__render"）
            uint32_t flags,                          // OptiX 编译标志（优化级别、调试模式等）
            OptixModule* module,                     // 输出：编译后的 OptiX 模块句柄（CPU端）
            OptixPipeline* pipeline,                 // 输出：创建的管线句柄（CPU端）
            OptixShaderBindingTable& sbt,            // 输出：着色器绑定表（包含GPU端的函数地址）
            uint32_t numPayloadValues = 32,          // 光线载荷（Payload）的大小（默认32个uint32_t）
            const std::vector<std::string>& extra_includes = {}); // 额外的包含路径
    //通用 GPU 缓冲区重分配
    void reallocateBuffer(CUdeviceptr* bufferPtr, size_t& size, size_t newSize, cudaStream_t cudaStream);
    //重分配图元几何缓冲区
    void reallocatePrimGeomBuffer(cudaStream_t stream);
    //重分配运行时参数缓冲区
    void reallocateParamsDevice(size_t sz, cudaStream_t stream);
    //创建高斯粒子实例加速结构
    OptixTraversableHandle createParticleInstanceAS(cudaStream_t cudaStream);
    void buildAtmosphereAccel(
        torch::Tensor meshVerts,
        torch::Tensor meshIdx,
        torch::Tensor voxelKmax,
        int nx,
        int ny,
        int nz,
        int brick,
        int nbx,
        int nby,
        int nbz,
        const float origin[3],
        float dx,
        float dy,
        float dz,
        cudaStream_t cudaStream);

public:
    OptixTracer(
        const std::string& path,
        const std::string& cuda_path,
        const std::string& pipeline,
        const std::string& backwardPipeline,
        const std::string& primitive,
        float particleKernelDegree,
        float particleKernelMinResponse,
        bool particleKernelDensityClamping,
        int particleRadianceSphDegree,
        bool enableNormals,
        bool enableHitCounts,
        bool enableMIS,
        bool enableMetallic,
        bool enableVisualizeLights,
        bool enableRaygenSetupProfiling = false,
        int raygenAblationStage = 1);

    virtual ~OptixTracer();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> virtual trace(
        uint32_t frameNumber,
        torch::Tensor rayToWorld,
        torch::Tensor rayOri,
        torch::Tensor rayDir,
        torch::Tensor particleDensity,
        torch::Tensor particleMaterial,
        torch::Tensor particleRadiance,
        torch::Tensor particleShadingNormal,
        torch::Tensor environment,
        torch::Tensor environmentAliasTable,
        torch::Tensor lights,
        torch::Tensor lightAliasTable,
        torch::Tensor meshLightVertices,
        torch::Tensor meshLightTriangles,
        torch::Tensor meshLights,
        torch::Tensor meshLightTriangleAliasTable,
        uint32_t shIndirect,
        int sphDegree,
        float minTransmittance,
        uint32_t maxBounces);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> virtual traceBwd(
        uint32_t frameNumber,
        torch::Tensor rayToWorld,
        torch::Tensor rayOri,
        torch::Tensor rayDir,
        torch::Tensor rayRad,
        torch::Tensor rayDns,
        torch::Tensor rayHit,
        torch::Tensor rayHitSecondMoment,
        torch::Tensor rayDepthDistortion,
        torch::Tensor rayNrm,
        torch::Tensor rayShadingNrm,
        torch::Tensor rayMaterial,
        torch::Tensor rayPbr,
        torch::Tensor rayLight,
        torch::Tensor particleDensity,
        torch::Tensor particleMaterial,
        torch::Tensor particleRadiance,
        torch::Tensor particleShadingNormal,
        torch::Tensor environment,
        torch::Tensor environmentAliasTable,
        torch::Tensor lights,
        torch::Tensor lightAliasTable,
        torch::Tensor meshLightVertices,
        torch::Tensor meshLightTriangles,
        torch::Tensor meshLights,
        torch::Tensor meshLightTriangleAliasTable,
        torch::Tensor rayRadGrd,
        torch::Tensor rayDnsGrd,
        torch::Tensor rayHitGrd,
        torch::Tensor rayHitSecondMomentGrd,
        torch::Tensor rayDepthDistortionGrd,
        torch::Tensor rayNrmGrd,
        torch::Tensor rayShadingNrmGrd,
        torch::Tensor rayMaterialGrd,
        torch::Tensor rayPbrGrd,
        torch::Tensor rayLightGrd,
        uint32_t shIndirect,
        int sphDegree,
        float minTransmittance,
        uint32_t maxBounces);

    virtual void buildBVH(torch::Tensor mogPos,
                          torch::Tensor mogRot,
                          torch::Tensor mogScl,
                          torch::Tensor mogDns,
                          unsigned int rebuild,
                          bool allow_update);

    /// Voxel cloud + 1D gas + optional GS/mesh surfaces. Same 13-tuple as trace().
    /// Cloud extinction is voxels only; Gaussians are surfaces.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    traceAtmosphere(
        uint32_t frameNumber,
        torch::Tensor rayToWorld,
        torch::Tensor rayOri,
        torch::Tensor rayDir,
        torch::Tensor particleDensity,
        torch::Tensor particleMaterial,
        torch::Tensor particleRadiance,
        torch::Tensor particleShadingNormal,
        torch::Tensor voxelKa,
        torch::Tensor voxelKs,
        torch::Tensor voxelTemp,
        torch::Tensor voxelKmax,
        torch::Tensor voxelOrigin,
        torch::Tensor voxelSpacing,
        int brick,
        torch::Tensor gasZ,
        torch::Tensor gasKa,
        torch::Tensor gasKs,
        torch::Tensor gasTemp,
        torch::Tensor meshVerts,
        torch::Tensor meshIdx,
        torch::Tensor meshAlbedo,
        torch::Tensor meshTemp,
        torch::Tensor sunDir,
        float sunRadiance,
        float sunCosHalf,
        int spectralMode,
        float wavelength_m,
        float mieOmega,
        float mieG,
        float pathTmax,
        uint32_t maxBounces,
        torch::Tensor cieCdf = torch::Tensor(),
        torch::Tensor cieRcpIntegral = torch::Tensor(),
        float cieLambda0Nm = 380.f,
        float cieDlambdaNm = 1.f,
        torch::Tensor swLambdaLo = torch::Tensor(),
        torch::Tensor swLambdaHi = torch::Tensor(),
        torch::Tensor swGasKa = torch::Tensor(),
        torch::Tensor swGasKs = torch::Tensor(),
        torch::Tensor brdfLambdaNm = torch::Tensor(),
        torch::Tensor brdfR = torch::Tensor(),
        torch::Tensor swNquad = torch::Tensor(),
        torch::Tensor swQuadCdf = torch::Tensor(),
        torch::Tensor mieCsBand = torch::Tensor(),
        torch::Tensor mieCaBand = torch::Tensor(),
        torch::Tensor mieGnm = torch::Tensor(),
        float mieCsRef = 1.f,
        float mieCaRef = 1.f,
        torch::Tensor voxelZIface = torch::Tensor());

    /// Last forward optixLaunch wall time in ms (__raygen__rg + nested OptiX programs).

    /// Last forward optixLaunch wall time in ms (__raygen__rg + nested OptiX programs).
    /// Valid after trace() when PTIR_PROFILE_RAYGEN=1; otherwise 0.
    float getLastOptixLaunchMs() const;

    /// Mean per-pixel raygen setup (lines ~47-77) in ms from device clock64.
    /// Valid after trace() when ENABLE_RAYGEN_SETUP_PROFILING is compiled in.
    float getLastRaygenSetupMs() const;

    /// Sum of setup cycles across all launched pixels (last forward).
    unsigned long long getLastRaygenSetupCycles() const;
};
