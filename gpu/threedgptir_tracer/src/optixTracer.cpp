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

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgptir/cuoptixMacros.h>
#include <3dgptir/optixTracer.h>
#include <3dgptir/particlePrimitives.h>
#include <3dgptir/pipelineParameters.h>
#include <3dgptir/atmosphere.h>
#include <3dgptir/tensorBuffering.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>
#include <ATen/cuda/CUDAUtils.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <nvrtc.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <vector>

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
namespace {
//OptiX 上下文日志回调函数
void contextLogCB(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

//读取源码文件到字符串
bool readSourceFile(std::string& str, const std::string& filename) {
    // Try to open file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good()) {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

void getCuStringFromFile(std::string& cu, const char* filename) {
    // Try to get source code from file
    if (readSourceFile(cu, filename)) {
        return;
    }

    // Wasn't able to find or open the requested file
    throw std::runtime_error("Couldn't open source file " + std::string(filename));
}
//在运行时将 CUDA C++ 源码（.cu）编译成 PTX 中间代码，为后续 OptiX 模块创建做准备
//PTX（Parallel Thread Execution） 是 NVIDIA GPU 的中间语言
void getPtxFromCuString(std::string& ptx,
                        const char* include_dir,
                        const char* optix_include_dir,
                        const char* cuda_include_dir,
                        const char* cu_source,
                        const char* name,
                        const std::vector<std::string>& defines,
                        const char** log_string,
                        const std::vector<std::string>& extra_includes = {}) {
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cu_source, name, 0, NULL, NULL));

    // Gather NVRTC options
    std::vector<const char*> options;

    std::string sample_dir;
    sample_dir = std::string("-I") + include_dir;
    options.push_back(sample_dir.c_str());

    // Collect include dirs
    std::vector<std::string> include_dirs;

    include_dirs.push_back(std::string("-I") + optix_include_dir);
    include_dirs.push_back(std::string("-I") + cuda_include_dir);

    for (const std::string& extra_path : extra_includes) {
        include_dirs.push_back(std::string("-I") + extra_path);
    }

    for (const std::string& dir : include_dirs) {
        options.push_back(dir.c_str());
    }

    for (const std::string& def : defines) {
        options.push_back(def.c_str());
    }

    // Collect NVRTC options
    const char* compiler_options[] = {CUDA_NVRTC_OPTIONS};
    std::copy(std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options.size(), options.data());

    // Retrieve log output
    std::string g_nvrtcLog;
    size_t log_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
    g_nvrtcLog.resize(log_size);
    if (log_size > 1) {
        NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &g_nvrtcLog[0]));
        if (log_string)
            *log_string = g_nvrtcLog.c_str();
    }
    if (compileRes != NVRTC_SUCCESS)
        throw std::runtime_error("NVRTC Compilation failed.\n" + g_nvrtcLog);

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));
    std::fprintf(stderr, "[atm] nvrtc %s ptx=%zu src=%zu\n", name, ptx.size(), std::strlen(cu_source));

    // Cleanup
    NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
}

//OptiX 着色器代码的"加载+编译"一体化函数
const char* getInputData(const char* filename,
                         const char* include_dir,
                         const char* optix_include_dir,
                         const char* cuda_include_dir,
                         const char* name,
                         size_t& dataSize,
                         const std::vector<std::string>& defines,
                         const char** log,
                         const std::vector<std::string>& extra_includes = {}) {
    if (log)
        *log = NULL;

    std::string *ptx, cu;
    ptx = new std::string();

    getCuStringFromFile(cu, filename);
    getPtxFromCuString(*ptx, include_dir, optix_include_dir, cuda_include_dir, cu.c_str(), name, defines, log,
                       extra_includes);

    dataSize = ptx->size();
    return ptx->c_str();
}

//输入：字符串描述 → 输出：对应的图元类型枚举值
// 内联函数：将字符串转换为图元类型枚举
// 参数：primitiveTypeStr - 图元类型的字符串表示（如 "octahedron"）
// 返回：对应的 MOGPrimitiveTypes 枚举值
inline MOGPrimitiveTypes primitiveTypeFromStr(const std::string& primitiveTypeStr) {
    // 判断是否为二十面体（20个三角形，最精细的多面体近似）
    if (primitiveTypeStr == "icosahedron") {
        return MOGPrimitiveTypes::MOGTracingIcosaHedron;

    // 判断是否为八面体（8个三角形，中等精度）
    } else if (primitiveTypeStr == "octahedron") {
        return MOGPrimitiveTypes::MOGTracingOctraHedron;

    // 判断是否为四面体（4个三角形，最简单，性能最快）
    } else if (primitiveTypeStr == "tetrahedron") {
        return MOGPrimitiveTypes::MOGTracingTetraHedron;

    // 判断是否为菱形/钻石形（自定义形状）
    } else if (primitiveTypeStr == "diamond") {
        return MOGPrimitiveTypes::MOGTracingDiamond;

    // 判断是否为球体（高精度球面近似）
    } else if (primitiveTypeStr == "sphere") {
        return MOGPrimitiveTypes::MOGTracingSphere;

    // 判断是否为三六面体（特定几何结构）
    } else if (primitiveTypeStr == "trihexa") {
        return MOGPrimitiveTypes::MOGTracingTriHexa;

    // 判断是否为三角面片（表面元素，用于曲面表示）
    } else if (primitiveTypeStr == "trisurfel") {
        return MOGPrimitiveTypes::MOGTracingTriSurfel;
    
    // 判断是否为自定义图元（用户自己定义的形状）
    } else if (primitiveTypeStr == "custom") {
        return MOGPrimitiveTypes::MOGTracingCustom;
    
    // 默认情况：未匹配任何已知类型，返回实例模式（回退选项）
    } else {
        return MOGPrimitiveTypes::MOGTracingInstances;
    }
}

} // namespace

//------------------------------------------------------------------------------
// OptixTracer
//------------------------------------------------------------------------------

// 环境光照别名表类的构造函数
// 参数：table - 包含别名表数据的 PyTorch 张量
EnvAliasTable::EnvAliasTable(const torch::Tensor& table)
    : width(0),           // 环境贴图宽度（像素列数）
      height(0),          // 环境贴图高度（像素行数）
      numCells(0),        // 总像素数（width * height）
      prob(nullptr),      // 概率数组指针（GPU/CPU）
      alias(nullptr),     // 别名索引数组指针
      pdf(nullptr) {      // 概率密度函数数组指针

    // 验证输入张量的形状和大小
    // 期望：3 维张量，形状为 [3, height, width]
    //       第一维（索引0）存储概率
    //       第二维（索引1）存储别名索引
    //       第三维（索引2）存储PDF值
    if (table.dim() == 3 &&                    // 必须是 3 维张量
        table.size(0) == 3 &&                  // 第一维大小必须为 3（prob, alias, pdf）
        table.size(1) > 0 &&                   // 高度 > 0
        table.size(2) > 0 &&                   // 宽度 > 0
        table.numel() > 0) {                   // 总元素数 > 0

        // 提取环境贴图尺寸
        height = static_cast<int>(table.size(1));  // 高度（行数）
        width  = static_cast<int>(table.size(2));  // 宽度（列数）

        // 计算总像素数
        const int cells = width * height;
        numCells = cells;  // 保存到成员变量

        // 获取数据指针（指向张量第一个元素）
        // 数据布局： [prob数据 | alias数据 | pdf数据]
        //             ↑          ↑           ↑
        //           prob      alias       pdf
        prob     = table.data_ptr<float>();      // 指向第一个数组的起始位置
        alias    = prob + cells;                  // 跳过 prob 数组，指向 alias 数组
        pdf      = prob + 2 * cells;              // 跳过 prob 和 alias，指向 pdf 数组
    }
}

//管理 2D/Cube 贴图、数据存储
Environment::Environment(const torch::Tensor& environment)
    : data(nullptr),
      width(0),
      height(0),
      type(EnvironmentType_2D),
      offset{0.0f, 0.0f},
      aliasTable() {
    if (environment.dim() >= 3 && environment.size(0) > 0 && environment.size(1) > 0 && environment.size(2) == 4 && environment.numel() > 0) {
        data   = reinterpret_cast<const float4*>(environment.data_ptr<float>());
        height = static_cast<int>(environment.size(0));
        width  = static_cast<int>(environment.size(1));
        if (environment.size(0) == 6 * environment.size(1)) {
            type = EnvironmentType_Cube;
        }
    }
}

Environment::Environment(const torch::Tensor& environment, const torch::Tensor& aliasTable)
    : Environment(environment) {
    this->aliasTable = EnvAliasTable(aliasTable);
}

std::vector<std::string> OptixTracer::generateDefines(
    float particleKernelDegree,
    bool particleKernelDensityClamping,
    int particleRadianceSphDegree,
    bool enableNormals,
    bool enableHitCounts,
    bool enableMIS,
    bool enableMetallic,
    bool enableVisualizeLights,
    bool enableRaygenSetupProfiling,
    int raygenAblationStage) {
    std::vector<std::string> defines;
    if (_state) {
        defines.emplace_back("-DPARTICLE_KERNEL_DEGREE=" + std::to_string(static_cast<int32_t>(particleKernelDegree)));
        if (enableNormals) {
            defines.emplace_back("-DENABLE_NORMALS");
        }
        if (enableHitCounts) {
            defines.emplace_back("-DENABLE_HIT_COUNTS");
        }
        if (enableMIS) {
            defines.emplace_back("-DENABLE_MIS");
        }
        if (enableMetallic) {
            defines.emplace_back("-DENABLE_METALLIC");
        }
        if (enableVisualizeLights) {
            defines.emplace_back("-DENABLE_VISUALIZE_LIGHTS");
        }
        if (enableRaygenSetupProfiling) {
            defines.emplace_back("-DENABLE_RAYGEN_SETUP_PROFILING");
            const int stage = (raygenAblationStage < 1) ? 1 : ((raygenAblationStage > 3) ? 3 : raygenAblationStage);
            defines.emplace_back("-DRAYGEN_ABLATION_STAGE=" + std::to_string(stage));
        }
        defines.emplace_back("-DSPH_MAX_NUM_COEFFS=" + std::to_string((_state->particleRadianceSphDegree + 1) * (_state->particleRadianceSphDegree + 1)));
        defines.emplace_back("-DPARTICLE_PRIMITIVE_TYPE=" + std::to_string(_state->gPrimType));
        defines.emplace_back("-DPARTICLE_PRIMITIVE_CLAMPED=" + std::to_string(particleKernelDensityClamping ? 1 : 0));
    }
    return defines;
}

//初始化 OptiX 渲染器：创建上下文、解析参数、生成宏定义、建立前向和反向两条光线追踪管线
OptixTracer::OptixTracer(
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
    bool enableRaygenSetupProfiling,
    int raygenAblationStage) {

    _state = new State();
    memset(_state, 0, sizeof(State));

    char log[2048]; // For error reporting from OptiX creation functions

    // create OptiX context
    _state->context = nullptr;
    {
        // Initialize the OptiX API, loading all API entry points
        OPTIX_CHECK(optixInit());

        // Specify context options
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &contextLogCB;
        options.logCallbackLevel          = 3;

        // Associate a CUDA context (and therefore a specific GPU) with this
        // device context
        CUcontext cuCtx = 0; // zero means take the current context
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &_state->context));
    }

    _state->particleRadianceSphDegree     = particleRadianceSphDegree;
    _state->particleKernelDegree          = particleKernelDegree;
    _state->particleKernelMinResponse     = particleKernelMinResponse;
    _state->particleKernelDensityClamping = particleKernelDensityClamping;
    _state->gNum                          = 0;
    _state->gPrimType                     = primitiveTypeFromStr(primitive);
    _state->gPrimNumVert                  = 0;
    _state->gPrimNumTri                   = 0;
    _state->gPrimNumVert                  = 0;
    _state->gPrimNumTri                   = 0;
    _state->enableRaygenSetupProfiling    = enableRaygenSetupProfiling;
    _state->raygenSetupCycleCounter       = 0;
    _state->lastRaygenSetupCycles         = 0;
    _state->lastRaygenSetupMs             = 0.f;
    {
        int clockRateKHz = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, 0));
        _state->smClockRateKHz = clockRateKHz;
    }
    if (_state->enableRaygenSetupProfiling) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_state->raygenSetupCycleCounter),
                              sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(_state->raygenSetupCycleCounter), 0,
                              sizeof(unsigned long long)));
    }

    std::vector<std::string> defines = generateDefines(particleKernelDegree, particleKernelDensityClamping,
                                                       particleRadianceSphDegree, enableNormals, enableHitCounts, enableMIS, enableMetallic, enableVisualizeLights,
                                                       enableRaygenSetupProfiling, raygenAblationStage);

    const uint32_t sharedFlags =
        (_state->gPrimType == MOGTracingSphere ? PipelineFlag_SpherePrim : ((_state->gPrimType == MOGTracingCustom) || (_state->gPrimType == MOGTracingInstances) ? PipelineFlag_HasIS : 0));

    //
    // Create pipelines
    //
    _state->moduleTracingFwd   = nullptr;
    _state->pipelineTracingFwd = nullptr;
    _state->sbtTracingFwd      = {};
    //调用 createPipeline 函数，编译着色器源码、创建 OptiX 模块、链接程序组、生成着色器绑定表，最终建立一条完整的光线追踪管线。
    createPipeline(_state->context, path, path, cuda_path, defines, pipeline + "Optix",
                   sharedFlags | PipelineFlag_HasRG | PipelineFlag_HasAH | PipelineFlag_AtmAccel,
                   &_state->moduleTracingFwd, &_state->pipelineTracingFwd, _state->sbtTracingFwd, 32);

    _state->moduleTracingBwd   = nullptr;
    _state->pipelineTracingBwd = nullptr;
    _state->sbtTracingBwd      = {};
    createPipeline(_state->context, path, path, cuda_path, defines, backwardPipeline + "Optix", sharedFlags | PipelineFlag_HasRG | PipelineFlag_HasAH,
                   &_state->moduleTracingBwd, &_state->pipelineTracingBwd, _state->sbtTracingBwd, 32);
}


float OptixTracer::getLastOptixLaunchMs() const {
    return _state ? _state->lastOptixLaunchMs : 0.f;
}

float OptixTracer::getLastRaygenSetupMs() const {
    return _state ? _state->lastRaygenSetupMs : 0.f;
}

unsigned long long OptixTracer::getLastRaygenSetupCycles() const {
    return _state ? _state->lastRaygenSetupCycles : 0ull;
}

OptixTracer::~OptixTracer(void) {
    OPTIX_CHECK(optixPipelineDestroy(_state->pipelineTracingFwd));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingFwd.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingFwd.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingFwd.hitgroupRecordBase)));
    OPTIX_CHECK(optixModuleDestroy(_state->moduleTracingFwd));

    OPTIX_CHECK(optixPipelineDestroy(_state->pipelineTracingBwd));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingBwd.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingBwd.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingBwd.hitgroupRecordBase)));
    OPTIX_CHECK(optixModuleDestroy(_state->moduleTracingBwd));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->gPrimVrt)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->gPrimTri)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->gPrimAABB)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->iasBuffer)));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->optixAabbPtr)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->paramsDevice)));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->gasBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->meshGasBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->volGasBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->volAabbBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->volBrickIdBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->meshVolIasBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->meshVolIasInstances)));
    if (_state->raygenSetupCycleCounter) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->raygenSetupCycleCounter)));
        _state->raygenSetupCycleCounter = 0;
    }
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->gasBufferTmp)));
    OPTIX_CHECK(optixDeviceContextDestroy(_state->context));

    delete _state;
}

// ============================================================================
// 创建 OptiX 光线追踪管线
// ============================================================================
// 功能：将 CUDA 着色器源码编译、链接成 OptiX 管线，并准备着色器绑定表
// 参数：
//   context           - OptiX 设备上下文
//   path              - 项目根路径（用于查找源码和头文件）
//   dependencies_path - 依赖库路径（OptiX 头文件位置）
//   cuda_path         - CUDA Toolkit 安装路径
//   defines           - 预处理器宏定义列表（如 "-DENABLE_NORMALS"）
//   kernel_name       - 着色器内核名称（用于构造 .cu 文件名）
//   flags             - 管线功能标志（HasRG, HasAH, SpherePrim 等）
//   module            - 输出：OptiX 模块句柄
//   pipeline          - 输出：OptiX 管线句柄
//   sbt               - 输出：着色器绑定表（包含 GPU 地址）
//   numPayloadValues  - Payload 大小（默认 32 个 uint32_t）
//   extra_includes    - 额外的头文件搜索路径
// ============================================================================
void OptixTracer::createPipeline(
    const OptixDeviceContext context,          // OptiX 上下文（CPU 句柄）
    const std::string& path,                   // 项目根路径
    const std::string& dependencies_path,      // 依赖库路径
    const std::string& cuda_path,              // CUDA 路径
    const std::vector<std::string>& defines,   // 编译宏定义
    const std::string& kernel_name,            // 内核名称
    uint32_t flags,                            // 管线标志
    OptixModule* module,                       // [输出] 模块句柄
    OptixPipeline* pipeline,                   // [输出] 管线句柄
    OptixShaderBindingTable& sbt,              // [输出] 着色器绑定表
    uint32_t numPayloadValues,                 // Payload 大小
    const std::vector<std::string>& extra_includes  // 额外头文件路径
) {
    char log[2048];  // 用于存储 OptiX API 的日志/错误信息

    // =========================================================================
    // 第一阶段：编译着色器 → 创建 OptiX 模块
    // =========================================================================
    
    OptixPipelineCompileOptions pipeline_compile_options = {};  // 管线编译选项（全零初始化）
    OptixModule builtinIsModule = nullptr;  // 内置相交模块（仅球体图元使用）
    
    {
        // ---------- 1.1 设置模块编译选项 ----------
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;  // 最大寄存器数（使用默认值）
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;  // 最高优化级别（O3）
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;  // 最小调试信息（减少体积）

        // ---------- 1.2 设置管线编译选项 ----------
        pipeline_compile_options.usesMotionBlur = false;  // 不使用运动模糊（简化）
        pipeline_compile_options.traversableGraphFlags =
            (flags & PipelineFlag_AtmAccel)
                ? (OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS |
                   OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING)
                : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = numPayloadValues;  // 光线载荷大小（32 个 uint32_t）
        pipeline_compile_options.numAttributeValues = (flags & PipelineFlag_AtmAccel) ? 2 : 0;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // 不启用异常处理（性能优先）
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";  // 启动参数结构体名称（在着色器中对应 extern "C" __constant__ Params params）

        // ---------- 1.3 设置图元类型 ----------
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;  // 默认支持三角形
        if ((flags & PipelineFlag_HasIS) || (flags & PipelineFlag_AtmAccel)) {
            pipeline_compile_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        } else if (flags & PipelineFlag_SpherePrim) {
            pipeline_compile_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
        }

        // ---------- 1.4 构建源码文件路径 ----------
        size_t inputSize = 0;
        // 着色器源码文件：path/src/kernels/cuda/<kernel_name>.cu
        std::string shaderFile = path + "/src/kernels/cuda/" + kernel_name + ".cu";
        // 项目头文件目录：path/include
        std::string includeDir = path + "/include";
        // OptiX 头文件目录：dependencies_path/dependencies/optix-dev/include
        std::string optix_include_dir = dependencies_path + "/dependencies/optix-dev/include";
        // CUDA 头文件目录：cuda_path/include
        std::string cuda_include_dir = cuda_path + "/include";

        // ---------- 1.5 加载并编译源码 ----------
        // getInputData 内部调用 getCuStringFromFile 读取 .cu 文件，
        // 然后调用 getPtxFromCuString 使用 NVRTC 编译成 PTX
        const char* input = getInputData(
            shaderFile.c_str(),          // 源码文件路径
            includeDir.c_str(),          // 项目头文件目录
            optix_include_dir.c_str(),   // OptiX 头文件目录
            cuda_include_dir.c_str(),    // CUDA 头文件目录
            kernel_name.c_str(),         // 内核名称（用于错误信息）
            inputSize,                   // [输出] PTX 数据大小
            defines,                     // 编译宏定义
            (const char**)&log,          // [输出] 编译日志
            extra_includes               // 额外头文件路径
        );
        size_t sizeof_log = sizeof(log);

        // ---------- 1.6 从 PTX 创建 OptiX 模块 ----------
        // 这是 OptiX 的核心步骤：将 PTX 代码加载到 GPU，生成可执行的机器码
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            context,                      // OptiX 上下文
            &module_compile_options,      // 模块编译选项
            &pipeline_compile_options,    // 管线编译选项
            input,                        // PTX 代码指针
            inputSize,                    // PTX 代码大小
            log,                          // 日志缓冲区
            &sizeof_log,                  // [输入/输出] 日志缓冲区大小
            module                        // [输出] 模块句柄
        ));

        // ---------- 1.7 球体图元：获取内置相交模块 ----------
        // 如果是球体图元且没有自定义相交，使用 OptiX 内置的球体求交
        if (!(flags & PipelineFlag_HasIS) && (flags & PipelineFlag_SpherePrim)) {
            OptixBuiltinISOptions isOptions = {
                OPTIX_PRIMITIVE_TYPE_SPHERE,  // 图元类型：球体
                0,                             // 保留字段
                OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,  // 构建标志：优先快速追踪
                0                              // 保留字段
            };
            OPTIX_CHECK_LOG(optixBuiltinISModuleGet(
                context,                      // OptiX 上下文
                &module_compile_options,      // 模块编译选项
                &pipeline_compile_options,    // 管线编译选项
                &isOptions,                   // 球体选项
                &builtinIsModule              // [输出] 内置相交模块句柄
            ));
        }
    }

    // =========================================================================
    // 第二阶段：创建程序组（Program Groups）
    // =========================================================================
    // 程序组是 OptiX 中可执行的最小单元，每个程序组对应一个着色器函数
    // =========================================================================

    OptixProgramGroup raygen_prog_group = nullptr;    // 光线生成程序组
    OptixProgramGroup miss_prog_group = nullptr;      // 未命中程序组
    OptixProgramGroup hitgroup_prog_group = nullptr;  // GS anyhit
    OptixProgramGroup hitgroup_mesh_prog_group = nullptr;
    OptixProgramGroup hitgroup_vol_prog_group = nullptr;

    {
        OptixProgramGroupOptions program_group_options = {};  // 程序组选项（全零）

        // ---------- 2.1 创建光线生成程序组 ----------
        // 光线生成着色器是管线的入口点，每个像素/线程调用一次
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        if (flags & PipelineFlag_HasRG) {
            raygen_prog_group_desc.raygen.module = *module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";  // 入口函数名
        }
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1,                              // 程序组数量（1 个）
            &program_group_options,         // 程序组选项
            log,                            // 日志缓冲区
            &sizeof_log,                    // 日志缓冲区大小
            &raygen_prog_group              // [输出] 程序组句柄
        ));

        // ---------- 2.2 创建未命中程序组 ----------
        // 未命中着色器在光线没有与任何几何体相交时执行
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        if (flags & PipelineFlag_HasMS) {
            miss_prog_group_desc.miss.module = *module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        }
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,
            &program_group_options,
            log,
            &sizeof_log,
            &miss_prog_group
        ));

        // ---------- 2.3 创建命中组程序组 ----------
        // 命中组包含多个着色器：最近命中、任意命中、相交
        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        // 最近命中着色器（Closest Hit）：找到最近的交点后执行
        if (flags & PipelineFlag_HasCH) {
            hitgroup_prog_group_desc.hitgroup.moduleCH = *module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        }

        // 相交着色器（Intersection）：自定义图元的求交计算
        if (flags & PipelineFlag_HasIS) {
            // 使用用户自定义的相交着色器
            hitgroup_prog_group_desc.hitgroup.moduleIS = *module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
        } else if (flags & PipelineFlag_SpherePrim) {
            // 使用 OptiX 内置的球体相交（更快）
            hitgroup_prog_group_desc.hitgroup.moduleIS = builtinIsModule;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;  // 内置模块无需函数名
        }

        // 任意命中着色器（Any Hit）：每次找到交点时执行（用于透明度等）
        if (flags & PipelineFlag_HasAH) {
            hitgroup_prog_group_desc.hitgroup.moduleAH = *module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        }

        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hitgroup_prog_group_desc,
            1,
            &program_group_options,
            log,
            &sizeof_log,
            &hitgroup_prog_group
        ));

        if (flags & PipelineFlag_AtmAccel) {
            OptixProgramGroupDesc mesh_hg = {};
            mesh_hg.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            mesh_hg.hitgroup.moduleCH = *module;
            mesh_hg.hitgroup.entryFunctionNameCH = "__closesthit__mesh";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context, &mesh_hg, 1, &program_group_options, log, &sizeof_log, &hitgroup_mesh_prog_group));

            OptixProgramGroupDesc vol_hg = {};
            vol_hg.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            vol_hg.hitgroup.moduleCH = *module;
            vol_hg.hitgroup.entryFunctionNameCH = "__closesthit__vol";
            vol_hg.hitgroup.moduleIS = *module;
            vol_hg.hitgroup.entryFunctionNameIS = "__intersection__vol";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context, &vol_hg, 1, &program_group_options, log, &sizeof_log, &hitgroup_vol_prog_group));
        }
    }

    // =========================================================================
    // 第三阶段：链接管线
    // =========================================================================
    // 将所有程序组链接成一个完整的管线，并设置栈大小
    // =========================================================================

    {
        const uint32_t max_trace_depth = 1;  // 最大光线追踪深度（1 = 只追踪主光线，不处理次级光线）

        // 收集所有程序组
        std::vector<OptixProgramGroup> program_groups;
        program_groups.push_back(raygen_prog_group);
        program_groups.push_back(miss_prog_group);
        program_groups.push_back(hitgroup_prog_group);
        if (flags & PipelineFlag_AtmAccel) {
            program_groups.push_back(hitgroup_mesh_prog_group);
            program_groups.push_back(hitgroup_vol_prog_group);
        }

        // ---------- 3.1 设置管线链接选项 ----------
        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

        // ---------- 3.2 创建管线 ----------
        // 这是链接步骤：将程序组组合成可执行的管线
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(
            context,
            &pipeline_compile_options,     // 管线编译选项
            &pipeline_link_options,        // 管线链接选项
            program_groups.data(),         // 程序组数组
            static_cast<unsigned int>(program_groups.size()),  // 程序组数量
            log,                           // 日志缓冲区
            &sizeof_log,                   // 日志缓冲区大小
            pipeline                       // [输出] 管线句柄
        ));

        // ---------- 3.3 计算栈大小 ----------
        // OptiX 需要知道每个程序组使用的栈大小，以便分配 GPU 内存
        OptixStackSizes stack_sizes = {};
        for (auto& prog_group : program_groups) {
            // 累加每个程序组的栈使用量
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(
            &stack_sizes,
            max_trace_depth,
            0,
            0,
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state,
            &continuation_stack_size
        ));
        if (flags & PipelineFlag_AtmAccel) {
            /* 8M-triangle mesh GAS + custom volume IS needs a deeper continuation stack. */
            if (direct_callable_stack_size_from_traversal < 2048u) {
                direct_callable_stack_size_from_traversal = 2048u;
            }
            if (direct_callable_stack_size_from_state < 2048u) {
                direct_callable_stack_size_from_state = 2048u;
            }
            if (continuation_stack_size < 16384u) {
                continuation_stack_size = 16384u;
            }
            std::fprintf(stderr, "[atm] pipeline stack trav=%u state=%u cont=%u\n",
                         direct_callable_stack_size_from_traversal,
                         direct_callable_stack_size_from_state,
                         continuation_stack_size);
        }

        // ---------- 3.4 设置管线栈大小 ----------
        OPTIX_CHECK(optixPipelineSetStackSize(
            *pipeline,
            direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state,
            continuation_stack_size,
            (flags & PipelineFlag_AtmAccel) ? 2 : 1
        ));
    }

    // =========================================================================
    // 第四阶段：构建着色器绑定表（SBT）
    // =========================================================================
    // SBT 是 GPU 内存中的一个表，包含所有程序组的入口地址和数据
    // OptiX 在运行时通过 SBT 找到对应的着色器函数
    // =========================================================================

    {
        // ---------- 4.1 创建光线生成记录 ----------
        CUdeviceptr raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        // 在 GPU 上分配内存
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));

        RayGenSbtRecord rg_sbt;
        // 将程序组信息打包到 SBT 记录中
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        // 拷贝到 GPU
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice
        ));

        // ---------- 4.2 创建未命中记录 ----------
        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));

        MissSbtRecord ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(miss_record),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice
        ));

        // ---------- 4.3 创建命中组记录 ----------
        const unsigned hgCount = (flags & PipelineFlag_AtmAccel) ? ATM_SBT_HITGROUP_COUNT : 1u;
        CUdeviceptr hitgroup_record;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord) * hgCount;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));

        std::vector<HitGroupSbtRecord> hg_sbt(hgCount);
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt[0]));
        if (hgCount > 1) {
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_mesh_prog_group, &hg_sbt[ATM_SBT_MESH]));
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_vol_prog_group, &hg_sbt[ATM_SBT_VOL]));
        }
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(hitgroup_record),
            hg_sbt.data(),
            hitgroup_record_size,
            cudaMemcpyHostToDevice
        ));

        // ---------- 4.4 填充 SBT 结构 ----------
        sbt.raygenRecord = raygen_record;
        sbt.missRecordBase = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = hgCount;
        std::fprintf(stderr, "[atm] SBT hitgroups=%u recSize=%zu\n", hgCount, sizeof(HitGroupSbtRecord));
    }
    // 函数结束：管线已创建，SBT 已就绪，可以调用 optixLaunch 执行渲染
}

OptixTraversableHandle OptixTracer::createParticleInstanceAS(cudaStream_t cudaStream) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    if (!_state->optixAabbPtr) {
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&_state->optixAabbPtr), sizeof(OptixAabb), cudaStream));
    }
    OptixAabb hostOptixAabb{-1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(_state->optixAabbPtr), &hostOptixAabb, sizeof(OptixAabb), cudaMemcpyHostToDevice, cudaStream));

    OptixBuildInput prim_input                    = {};
    uint32_t prim_input_flags                     = 0;
    prim_input_flags                              = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    prim_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    prim_input.customPrimitiveArray.numPrimitives = 1;
    prim_input.customPrimitiveArray.aabbBuffers   = &_state->optixAabbPtr;
    prim_input.customPrimitiveArray.strideInBytes = 0;
    prim_input.customPrimitiveArray.flags         = &prim_input_flags;
    prim_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(_state->context, &accel_options, &prim_input,
                                             1, // Number of build inputs
                                             &gas_buffer_sizes));
    if (_state->gasBufferTmpSz < gas_buffer_sizes.tempSizeInBytes) {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->gasBufferTmp), cudaStream));
        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void**>(&_state->gasBufferTmp), gas_buffer_sizes.tempSizeInBytes, cudaStream));
        _state->gasBufferTmpSz = gas_buffer_sizes.tempSizeInBytes;
    }
    if ((_state->iasBufferSz < gas_buffer_sizes.outputSizeInBytes)) {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->iasBuffer), cudaStream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&_state->iasBuffer),
                                   gas_buffer_sizes.outputSizeInBytes, cudaStream));
        _state->iasBufferSz = gas_buffer_sizes.outputSizeInBytes;
    }

    OptixTraversableHandle iasHandle;
    OPTIX_CHECK(optixAccelBuild(_state->context,
                                cudaStream, // CUDA stream
                                &accel_options, &prim_input,
                                1, // num build inputs
                                _state->gasBufferTmp, gas_buffer_sizes.tempSizeInBytes, _state->iasBuffer,
                                gas_buffer_sizes.outputSizeInBytes, &iasHandle,
                                nullptr, // emitted property list
                                0        // num emitted properties
                                ));
    return iasHandle;
}

void OptixTracer::reallocateBuffer(CUdeviceptr* bufferPtr, size_t& size, size_t newSize, cudaStream_t cudaStream) {
    if (newSize > size) {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(*bufferPtr), cudaStream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(bufferPtr), newSize, cudaStream));
        size = newSize;
    }
}

void OptixTracer::reallocatePrimGeomBuffer(cudaStream_t cudaStream) {
    if (_state) {
        reallocateBuffer(&_state->gPrimVrt, _state->gPrimVrtSz, sizeof(float3) * _state->gPrimNumVert * _state->gNum, cudaStream);
        reallocateBuffer(&_state->gPrimTri, _state->gPrimTriSz, sizeof(int3) * _state->gPrimNumTri * _state->gNum, cudaStream);
    }
}

void OptixTracer::reallocateParamsDevice(size_t sz, cudaStream_t cudaStream) {
    if (_state) {
        reallocateBuffer(&_state->paramsDevice, _state->paramsDeviceSz, sz, cudaStream);
    }
}

// ============================================================================
// 构建/更新 BVH 加速结构
// ============================================================================
// 功能：将高斯粒子转换为几何图元，构建 OptiX GAS（底层加速结构）
// 参数：
//   mogPos       - [N, 3] 高斯粒子的位置 (x, y, z)
//   mogRot       - [N, 4] 高斯粒子的旋转 (四元数 w, x, y, z)
//   mogScl       - [N, 3] 高斯粒子的缩放 (sx, sy, sz)
//   mogDns       - [N, 1] 高斯粒子的密度/不透明度
//   rebuild      - 0=更新(update), 1=完全重建(build)
//   allow_update - 是否允许后续增量更新
// ============================================================================
void OptixTracer::buildBVH(
    torch::Tensor mogPos,      // 高斯位置 [N, 3]
    torch::Tensor mogRot,      // 高斯旋转 [N, 4]
    torch::Tensor mogScl,      // 高斯缩放 [N, 3]
    torch::Tensor mogDns,      // 高斯密度 [N, 1]
    unsigned int rebuild,      // 0=更新, 1=重建
    bool allow_update)         // 是否允许后续更新
{
    // ============================================================
    // 1. 获取高斯粒子数量
    // ============================================================
    const uint32_t gNum = mogPos.size(0);  // 当前高斯粒子总数

    // ============================================================
    // 2. 设置渲染选项
    // ============================================================
    // 是否启用自适应核钳制（防止密度过大导致数值不稳定）
    const uint32_t primitiveOpts = _state->particleKernelDensityClamping 
        ? MOGRenderOpts::MOGRenderAdaptiveKernelClamping 
        : MOGRenderOpts::MOGRenderNone;

    // PyTorch Tensor 选项（仅供类型推导，未实际使用）
    const torch::TensorOptions opts = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    // ============================================================
    // 3. 检查粒子数量变化
    // ============================================================
    // 如果粒子数量变了，必须重建（不能只更新）
    if (!rebuild && (_state->gNum != gNum)) {
        std::cerr << "ERROR:: cannot refit GAS with a different number of gaussian" << std::endl;
        rebuild = 1;  // 强制重建
    }
    _state->gNum = gNum;  // 更新当前粒子数

    // ============================================================
    // 4. 获取 CUDA 流（用于异步操作）
    // ============================================================
    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    // ============================================================
    // 5. 根据图元类型生成几何数据
    // ============================================================
    // 将每个高斯粒子转换为对应的 3D 图元（三角形网格/自定义图元）
    // ============================================================

    // ---------- 5.1 自定义图元（只使用 AABB） ----------
    if (_state->gPrimType == MOGTracingCustom) {
        // 分配 GPU 内存存储每个高斯的 AABB
        if (_state->gPrimAABBSz < sizeof(OptixAabb) * gNum) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->gPrimAABB), cudaStream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&_state->gPrimAABB), 
                sizeof(OptixAabb) * gNum, 
                cudaStream));
            _state->gPrimAABBSz = sizeof(OptixAabb) * gNum;
        }

        // 自定义图元没有顶点和三角形（只需 AABB）
        _state->gPrimNumVert = 0;
        _state->gPrimNumTri  = 0;

        // 计算每个高斯粒子的包围盒（AABB）
        computeGaussianEnclosingAABB(
            gNum,                                    // 粒子数量
            getPtr<float3>(mogPos),                  // 位置
            getPtr<float4>(mogRot),                  // 旋转
            getPtr<float3>(mogScl),                  // 缩放
            getPtr<float>(mogDns),                   // 密度
            _state->particleKernelMinResponse,       // 最小响应阈值
            primitiveOpts,                           // 渲染选项
            _state->particleKernelDegree,            // 核函数带宽
            reinterpret_cast<OptixAabb*>(_state->gPrimAABB),  // [输出] AABB 数组
            cudaStream
        );
    }

    // ---------- 5.2 实例化图元（顶层加速结构） ----------
    else if (_state->gPrimType == MOGTracingInstances) {
        // 分配 GPU 内存存储每个高斯的实例数据
        if (_state->gPrimAABBSz < sizeof(OptixInstance) * gNum) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->gPrimAABB), cudaStream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&_state->gPrimAABB), 
                sizeof(OptixInstance) * gNum, 
                cudaStream));
            _state->gPrimAABBSz = sizeof(OptixInstance) * gNum;
        }

        // 创建顶层加速结构（IAS）句柄
        OptixTraversableHandle ias = createParticleInstanceAS(cudaStream);

        // 实例化图元没有顶点和三角形（使用 IAS）
        _state->gPrimNumVert = 0;
        _state->gPrimNumTri  = 0;

        // 计算每个高斯粒子的实例数据
        computeGaussianEnclosingInstances(
            gNum,
            getPtr<float3>(mogPos),
            getPtr<float4>(mogRot),
            getPtr<float3>(mogScl),
            getPtr<float>(mogDns),
            _state->particleKernelMinResponse,
            primitiveOpts,
            _state->particleKernelDegree,
            ias,                                     // IAS 句柄
            reinterpret_cast<OptixInstance*>(_state->gPrimAABB),  // [输出] 实例数组
            cudaStream
        );
    }

    // ---------- 5.3 标准图元（三角形网格） ----------
    else {
        // ----- 5.3.1 二十面体（20 个三角形，最高精度） -----
        if (_state->gPrimType == MOGTracingIcosaHedron) {
            _state->gPrimNumVert = 12;   // 每个粒子 12 个顶点
            _state->gPrimNumTri  = 20;   // 每个粒子 20 个三角形
            reallocatePrimGeomBuffer(cudaStream);  // 重新分配顶点/索引缓冲区

            computeGaussianEnclosingIcosaHedron(
                gNum,
                getPtr<float3>(mogPos),
                getPtr<float4>(mogRot),
                getPtr<float3>(mogScl),
                getPtr<float>(mogDns),
                _state->particleKernelMinResponse,
                primitiveOpts,
                _state->particleKernelDegree,
                reinterpret_cast<float3*>(_state->gPrimVrt),  // [输出] 顶点数组
                reinterpret_cast<int3*>(_state->gPrimTri),    // [输出] 三角形索引
                cudaStream
            );
        }

        // ----- 5.3.2 八面体（8 个三角形，中等精度） -----
        else if (_state->gPrimType == MOGTracingOctraHedron) {
            _state->gPrimNumVert = 6;
            _state->gPrimNumTri  = 8;
            reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingOctaHedron(
                gNum,
                getPtr<float3>(mogPos),
                getPtr<float4>(mogRot),
                getPtr<float3>(mogScl),
                getPtr<float>(mogDns),
                _state->particleKernelMinResponse,
                primitiveOpts,
                _state->particleKernelDegree,
                reinterpret_cast<float3*>(_state->gPrimVrt),
                reinterpret_cast<int3*>(_state->gPrimTri),
                cudaStream
            );
            CUDA_CHECK_LAST();
        }

        // ----- 5.3.3 三六面体（6 个三角形） -----
        else if (_state->gPrimType == MOGTracingTriHexa) {
            _state->gPrimNumVert = 6;
            _state->gPrimNumTri  = 6;
            reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingTriHexa(
                gNum,
                getPtr<float3>(mogPos),
                getPtr<float4>(mogRot),
                getPtr<float3>(mogScl),
                getPtr<float>(mogDns),
                _state->particleKernelMinResponse,
                primitiveOpts,
                _state->particleKernelDegree,
                reinterpret_cast<float3*>(_state->gPrimVrt),
                reinterpret_cast<int3*>(_state->gPrimTri),
                cudaStream
            );
            CUDA_CHECK_LAST();
        }

        // ----- 5.3.4 三角面片（2 个三角形，带额外粒子数据） -----
        else if (_state->gPrimType == MOGTracingTriSurfel) {
            _state->gPrimNumVert = 4;
            _state->gPrimNumTri  = 2;
            reallocatePrimGeomBuffer(cudaStream);
            // 为每个粒子分配额外的管线数据
            reallocateBuffer(
                &_state->gPipelineParticleData, 
                _state->gPipelineParticleDataSz, 
                sizeof(float4) * gNum, 
                cudaStream
            );

            computeGaussianEnclosingTriSurfel(
                gNum,
                getPtr<float3>(mogPos),
                getPtr<float4>(mogRot),
                getPtr<float3>(mogScl),
                getPtr<float>(mogDns),
                _state->particleKernelMinResponse,
                primitiveOpts,
                _state->particleKernelDegree,
                reinterpret_cast<float3*>(_state->gPrimVrt),
                reinterpret_cast<int3*>(_state->gPrimTri),
                reinterpret_cast<float4*>(_state->gPipelineParticleData),  // [输出] 额外数据
                cudaStream
            );
            CUDA_CHECK_LAST();
        }

        // ----- 5.3.5 四面体（4 个三角形，最快） -----
        else if (_state->gPrimType == MOGTracingTetraHedron) {
            _state->gPrimNumVert = 4;
            _state->gPrimNumTri  = 4;
            reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingTetraHedron(
                gNum,
                getPtr<float3>(mogPos),
                getPtr<float4>(mogRot),
                getPtr<float3>(mogScl),
                getPtr<float>(mogDns),
                _state->particleKernelMinResponse,
                primitiveOpts,
                _state->particleKernelDegree,
                reinterpret_cast<float3*>(_state->gPrimVrt),
                reinterpret_cast<int3*>(_state->gPrimTri),
                cudaStream
            );
        }

        // ----- 5.3.6 球体（OptiX 内置图元，最高性能） -----
        else if (_state->gPrimType == MOGTracingSphere) {
            _state->gPrimNumVert = 0;      // 球体不需要顶点
            _state->gPrimNumTri  = 1;      // 每个粒子 1 个图元
            // 分配球心缓冲区
            reallocateBuffer(
                &_state->gPrimVrt, 
                _state->gPrimVrtSz, 
                sizeof(float3) * gNum, 
                cudaStream
            );
            // 分配半径缓冲区
            reallocateBuffer(
                &_state->gPrimTri, 
                _state->gPrimTriSz, 
                sizeof(float) * gNum, 
                cudaStream
            );

            computeGaussianEnclosingSphere(
                gNum,
                getPtr<float3>(mogPos),
                getPtr<float4>(mogRot),
                getPtr<float3>(mogScl),
                getPtr<float>(mogDns),
                _state->particleKernelMinResponse,
                primitiveOpts,
                _state->particleKernelDegree,
                reinterpret_cast<float3*>(_state->gPrimVrt),  // [输出] 球心位置
                reinterpret_cast<float*>(_state->gPrimTri),    // [输出] 半径
                cudaStream
            );
        }

        // ----- 5.3.7 菱形/钻石形（6 个三角形） -----
        else {
            _state->gPrimNumVert = 5;
            _state->gPrimNumTri  = 6;
            reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingDiamond(
                gNum,
                getPtr<float3>(mogPos),
                getPtr<float4>(mogRot),
                getPtr<float3>(mogScl),
                getPtr<float>(mogDns),
                _state->particleKernelMinResponse,
                primitiveOpts,
                _state->particleKernelDegree,
                reinterpret_cast<float3*>(_state->gPrimVrt),
                reinterpret_cast<int3*>(_state->gPrimTri),
                cudaStream
            );
        }
    }

    // ============================================================
    // 6. 构建 OptiX 底层加速结构（GAS）
    // ============================================================
    {
        // ---------- 6.1 设置加速结构构建选项 ----------
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;  // 优先追踪速度
        if (allow_update) {
            accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;  // 允许后续更新
        }
        // rebuild=0 → 更新（快速），rebuild=1 → 完全重建（慢但可以改变结构）
        accel_options.operation = rebuild 
            ? OPTIX_BUILD_OPERATION_BUILD   // 完全重建
            : OPTIX_BUILD_OPERATION_UPDATE; // 增量更新

        // ---------- 6.2 设置构建输入 ----------
        OptixBuildInput prim_input = {};
        uint32_t prim_input_flags = 0;

        // ---- 自定义图元（只有 AABB） ----
        if (_state->gPrimType == MOGTracingCustom) {
            prim_input_flags = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            prim_input.customPrimitiveArray.numPrimitives = gNum;
            prim_input.customPrimitiveArray.aabbBuffers = &_state->gPrimAABB;
            prim_input.customPrimitiveArray.strideInBytes = 0;
            prim_input.customPrimitiveArray.flags = &prim_input_flags;
            prim_input.customPrimitiveArray.numSbtRecords = 1;
        }

        // ---- 实例化图元（IAS） ----
        else if (_state->gPrimType == MOGTracingInstances) {
            prim_input_flags = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            prim_input.instanceArray.numInstances = gNum;
            prim_input.instanceArray.instances = _state->gPrimAABB;
        }

        // ---- 球体图元（OptiX 内置） ----
        else if (_state->gPrimType == MOGTracingSphere) {
            prim_input_flags = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
            prim_input.sphereArray.vertexBuffers = &_state->gPrimVrt;      // 球心
            prim_input.sphereArray.vertexStrideInBytes = 0;
            prim_input.sphereArray.numVertices = gNum;
            prim_input.sphereArray.radiusBuffers = &_state->gPrimTri;      // 半径
            prim_input.sphereArray.radiusStrideInBytes = 0;
            prim_input.sphereArray.singleRadius = 0;
            prim_input.sphereArray.flags = &prim_input_flags;
            prim_input.sphereArray.numSbtRecords = 1;
        }

        // ---- 三角形网格（默认） ----
        else {
            prim_input_flags = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            prim_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            prim_input.triangleArray.numVertices = _state->gPrimNumVert * gNum;
            prim_input.triangleArray.vertexBuffers = &_state->gPrimVrt;
            prim_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            prim_input.triangleArray.numIndexTriplets = _state->gPrimNumTri * gNum;
            prim_input.triangleArray.indexBuffer = _state->gPrimTri;
            prim_input.triangleArray.flags = &prim_input_flags;
            prim_input.triangleArray.numSbtRecords = 1;
        }

        // ---------- 6.3 计算所需内存大小 ----------
        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            _state->context,     // OptiX 上下文
            &accel_options,      // 构建选项
            &prim_input,         // 构建输入
            1,                   // 输入数量
            &gas_buffer_sizes    // [输出] 内存大小
        ));

        // ---------- 6.4 分配临时缓冲区 ----------
        if (_state->gasBufferTmpSz < gas_buffer_sizes.tempSizeInBytes) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->gasBufferTmp), cudaStream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&_state->gasBufferTmp), 
                gas_buffer_sizes.tempSizeInBytes, 
                cudaStream));
            _state->gasBufferTmpSz = gas_buffer_sizes.tempSizeInBytes;
        }

        // ---------- 6.5 分配 GAS 输出缓冲区 ----------
        if (rebuild && (_state->gasBufferSz < gas_buffer_sizes.outputSizeInBytes)) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->gasBuffer), cudaStream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&_state->gasBuffer),
                gas_buffer_sizes.outputSizeInBytes, 
                cudaStream));
            _state->gasBufferSz = gas_buffer_sizes.outputSizeInBytes;
        }

        // ---------- 6.6 分配 AABB 输出缓冲区 ----------
        if (!_state->optixAabbPtr) {
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&_state->optixAabbPtr), 
                sizeof(OptixAabb), 
                cudaStream));
        }

        // ---------- 6.7 设置 AABB 发射描述 ----------
        OptixAccelEmitDesc emit_desc = {};
        emit_desc.type = OPTIX_PROPERTY_TYPE_AABBS;  // 发射 AABB 属性
        emit_desc.result = _state->optixAabbPtr;     // 输出到该地址

        // ---------- 6.8 构建 GAS ----------
        OPTIX_CHECK(optixAccelBuild(
            _state->context,                         // OptiX 上下文
            cudaStream,                              // CUDA 流
            &accel_options,                          // 构建选项
            &prim_input,                             // 构建输入
            1,                                       // 输入数量
            _state->gasBufferTmp,                    // 临时缓冲区
            gas_buffer_sizes.tempSizeInBytes,        // 临时缓冲区大小
            _state->gasBuffer,                       // 输出缓冲区
            gas_buffer_sizes.outputSizeInBytes,      // 输出缓冲区大小
            &_state->gasHandle,                      // [输出] GAS 句柄
            &emit_desc,                              // 发射属性列表
            1                                        // 发射属性数量
        ));

        // ---------- 6.9 拷贝 AABB 到 CPU ----------
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(&_state->gasAABB),           // CPU 端
            reinterpret_cast<void*>(_state->optixAabbPtr),       // GPU 端
            sizeof(OptixAabb), 
            cudaMemcpyDeviceToHost, 
            cudaStream
        ));
    }

    // 检查 CUDA 错误
    CUDA_CHECK_LAST();
}

/**
 * @brief OptiX 光线追踪前向传播入口函数
 * 
 * 该函数是 3D 高斯泼溅（3DGS）渲染的核心入口，负责：
 * 1. 分配 GPU 输出张量（颜色、深度、法线、材质等）
 * 2. 组装 PipelineParameters 参数结构体
 * 3. 启动 OptiX 光线追踪内核
 * 4. 返回所有渲染结果
 * 
 * 返回值包含 13 个张量，涵盖传统渲染和 PBR 渲染的所有输出。
 * 
 * @param frameNumber           当前帧编号（用于时序累积）
 * @param rayToWorld            光线从相机空间到世界空间的变换矩阵 [3, 4]
 * @param rayOri                光线起点（相机空间）[B, H, W, 3]
 * @param rayDir                光线方向（相机空间）[B, H, W, 3]
 * @param particleDensity       粒子密度数据 [N] (位置/缩放/四元数/密度)
 * @param particleMaterial      粒子材质参数 [N] (albedo/roughness/metallic)
 * @param particleRadiance      粒子球谐系数 [N, C] (视角相关颜色)
 * @param particleShadingNormal 粒子着色法线 [N, 3]
 * @param environment           环境光照数据
 * @param environmentAliasTable 环境光重要性采样表
 * @param lights                光源列表 [N, 9]
 * @param lightAliasTable       光源重要性采样表
 * @param meshLightVertices     网格光源顶点 [V, 3]
 * @param meshLightTriangles    网格光源三角形索引 [T, 3]
 * @param meshLights            网格光源参数 [M, 8]
 * @param meshLightTriangleAliasTable 网格光源三角形采样表
 * @param shIndirect            是否启用间接光照（球谐间接光）
 * @param sphDegree             球谐函数阶数
 * @param minTransmittance      最小透射率截断值
 * @param maxBounces            最大光线反弹次数
 * 
 * @return std::tuple<13个张量> 包含：
 *   rayRadiance, rayDensity, rayHitDistance, rayHitDistanceSecondMoment,
 *   rayDepthDistortion, rayNormal, rayShadingNormal, rayMaterial,
 *   rayHitsCount, particleVisibility, rayPbr, rayLight, rayPbrComponents
 */
 std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
 torch::Tensor>
OptixTracer::trace(uint32_t frameNumber,
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
         uint32_t maxBounces) {

// ============================================================
// 1. 创建输出张量（GPU 内存分配）
// ============================================================

const torch::TensorOptions opts = torch::TensorOptions()
.dtype(torch::kFloat32)
.device(torch::kCUDA);

// ---- 传统 3DGS 输出 ----
torch::Tensor rayRad = torch::empty(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);  // [B, H, W, 3] RGB颜色

torch::Tensor rayDns = torch::empty(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);  // [B, H, W, 1] 累积密度

torch::Tensor rayHit = torch::empty(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 2}, opts);  // [B, H, W, 2] 命中距离(一阶矩+二阶矩? 注:这里是2通道)

torch::Tensor rayHitSecondMoment = torch::empty(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);  // [B, H, W, 1] 二阶矩

torch::Tensor rayDepthDistortion = torch::empty(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);  // [B, H, W, 1] 深度扭曲

torch::Tensor rayNrm = torch::empty(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);  // [B, H, W, 3] 几何法线

torch::Tensor rayShadingNrm = torch::empty(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);  // [B, H, W, 3] 着色法线

torch::Tensor rayMaterial = torch::empty(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 5}, opts);  // [B, H, W, 5] 材质参数

// ---- PBR 输出（初始化为零） ----
torch::Tensor rayHitsCount = torch::zeros(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);  // [B, H, W, 1] 命中计数

torch::Tensor rayPbr = torch::zeros(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);  // [B, H, W, 3] PBR颜色

torch::Tensor rayLight = torch::zeros(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);  // [B, H, W, 3] 环境光

torch::Tensor rayPbrComponents = torch::zeros(
{rayOri.size(0), rayOri.size(1), rayOri.size(2), 2, 3}, opts); // [B, H, W, 2, 3] PBR分量分解

// ---- 粒子可见性 ----
torch::Tensor particleVisibility = torch::zeros(
{particleDensity.size(0), 1}, opts);  // [N, 1] 每个粒子是否可见

// ============================================================
// 2. 组装 CPU 端参数结构体
// ============================================================

PipelineParameters paramsHost;
std::memset(&paramsHost.atmosphere, 0, sizeof(paramsHost.atmosphere));
paramsHost.atmosphere.enabled = 0;

// ---- 2.1 OptiX 加速结构句柄 ----
paramsHost.handle = _state->gasHandle;  // GAS 或 IAS 句柄
paramsHost.meshHandle = 0;
paramsHost.volHandle = 0;
paramsHost.meshVolIas = 0;
paramsHost.volBrickId = nullptr;
paramsHost.volPrimCount = 0;
paramsHost.meshRepeatLow[0] = paramsHost.meshRepeatLow[1] = paramsHost.meshRepeatLow[2] = 0.f;
paramsHost.meshRepeatUpp[0] = paramsHost.meshRepeatUpp[1] = paramsHost.meshRepeatUpp[2] = 0.f;
paramsHost.aabb   = _state->gasAABB;    // 场景包围盒

// ---- 2.2 帧信息 ----
paramsHost.frameBounds.x = rayOri.size(2) - 1;  // 图像宽度
paramsHost.frameBounds.y = rayOri.size(1) - 1;  // 图像高度
paramsHost.frameNumber   = frameNumber;
paramsHost.gPrimNumTri   = _state->gPrimNumTri;  // 每个粒子的三角形数

// ---- 2.3 渲染参数 ----
paramsHost.minTransmittance       = minTransmittance;
paramsHost.hitMinGaussianResponse = _state->particleKernelMinResponse;
paramsHost.alphaMinThreshold      = 1.0f / 255.0f;  // 透明度阈值（1/255）
paramsHost.sphDegree              = sphDegree;

// 间接光照：最多 2 次反弹（性能考虑）
paramsHost.maxBounces = shIndirect ? (maxBounces < 2u ? maxBounces : 2u) : maxBounces;
paramsHost.renderOpts = shIndirect;  // 渲染选项标志

// ---- 2.4 光线变换矩阵 ----
std::memcpy(&paramsHost.rayToWorld[0].x, 
      rayToWorld.cpu().data_ptr<float>(), 
      3 * sizeof(float4));

// ---- 2.5 光线输入张量 ----
paramsHost.rayOrigin    = packed_accessor32<float, 4>(rayOri);
paramsHost.rayDirection = packed_accessor32<float, 4>(rayDir);

// ---- 2.6 粒子数据指针 ----
paramsHost.particleDensity       = getPtr<const ParticleDensity>(particleDensity);
paramsHost.particleMaterial      = getPtr<const Material>(particleMaterial);
paramsHost.particleRadiance      = getPtr<const float>(particleRadiance);
paramsHost.particleShadingNormal = getPtr<const float>(particleShadingNormal);
paramsHost.particleExtendedData  = reinterpret_cast<const void*>(_state->gPipelineParticleData);
paramsHost.particleVisibility    = getPtr<int32_t>(particleVisibility);
paramsHost.raygenSetupCycleCounter =
    _state->enableRaygenSetupProfiling
        ? reinterpret_cast<unsigned long long*>(_state->raygenSetupCycleCounter)
        : nullptr;

// ---- 2.7 输出张量（写入 GPU） ----
paramsHost.rayRadiance    = packed_accessor32<float, 4>(rayRad);
paramsHost.rayDensity     = packed_accessor32<float, 4>(rayDns);
paramsHost.rayHitDistance = packed_accessor32<float, 4>(rayHit);
paramsHost.rayHitDistanceSecondMoment = packed_accessor32<float, 4>(rayHitSecondMoment);
paramsHost.rayDepthDistortion = packed_accessor32<float, 4>(rayDepthDistortion);
paramsHost.rayNormal      = packed_accessor32<float, 4>(rayNrm);
paramsHost.rayShadingNormal = packed_accessor32<float, 4>(rayShadingNrm);
paramsHost.rayMaterial    = packed_accessor32<float, 4>(rayMaterial);
paramsHost.rayHitsCount   = packed_accessor32<float, 4>(rayHitsCount);
paramsHost.rayPbr         = packed_accessor32<float, 4>(rayPbr);
paramsHost.rayLight       = packed_accessor32<float, 4>(rayLight);
paramsHost.rayPbrComponents = packed_accessor32<float, 5>(rayPbrComponents);  // 注意：这里是5维

// ---- 2.8 光照系统 ----
paramsHost.environment = Environment(environment, environmentAliasTable);

paramsHost.numLights = lights.size(0);
paramsHost.lights    = packed_accessor32<float, 2>(lights);

paramsHost.numLightEntries = lightAliasTable.size(1);
paramsHost.lightAliasTable = packed_accessor32<float, 2>(lightAliasTable);

paramsHost.numMeshLightVertices = meshLightVertices.size(0);
paramsHost.meshLightVertices = packed_accessor32<float, 2>(meshLightVertices);

paramsHost.numMeshLightTriangles = meshLightTriangles.size(0);
paramsHost.meshLightTriangles = packed_accessor32<int32_t, 2>(meshLightTriangles);

paramsHost.numMeshLights = meshLights.size(0);
paramsHost.meshLights = packed_accessor32<float, 2>(meshLights);

paramsHost.meshLightTriangleAliasTable = packed_accessor32<float, 2>(meshLightTriangleAliasTable);

// ============================================================
// 3. 传输参数到 GPU
// ============================================================

cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

// 确保 GPU 端有足够内存存放参数
reallocateParamsDevice(sizeof(paramsHost), cudaStream);

// 将参数从 CPU 拷贝到 GPU
CUDA_CHECK(cudaMemcpyAsync(
reinterpret_cast<void*>(_state->paramsDevice), 
&paramsHost, 
sizeof(paramsHost), 
cudaMemcpyHostToDevice, 
cudaStream));

// ============================================================
// 4. 启动 OptiX 光线追踪内核
// ============================================================

_state->lastOptixLaunchMs = 0.f;
_state->lastRaygenSetupMs = 0.f;
_state->lastRaygenSetupCycles = 0ull;
if (_state->enableRaygenSetupProfiling && _state->raygenSetupCycleCounter) {
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(_state->raygenSetupCycleCounter), 0,
                               sizeof(unsigned long long), cudaStream));
}
const char* profileRaygenEnv = std::getenv("PTIR_PROFILE_RAYGEN");
const bool profileRaygen = profileRaygenEnv && profileRaygenEnv[0] == '1';
cudaEvent_t raygenEventStart = nullptr;
cudaEvent_t raygenEventStop  = nullptr;
if (profileRaygen) {
    CUDA_CHECK(cudaEventCreate(&raygenEventStart));
    CUDA_CHECK(cudaEventCreate(&raygenEventStop));
    CUDA_CHECK(cudaEventRecord(raygenEventStart, cudaStream));
}

OPTIX_CHECK(optixLaunch(
                _state->pipelineTracingFwd,  //pipelineTracingFwd: 编译好的 OptiX 管线
                  cudaStream, //cudaStream: CUDA 流（异步执行）
                  _state->paramsDevice,  //paramsDevice: 参数结构体（GPU 端）
                  sizeof(PipelineParameters),  //sizeof(PipelineParameters): 参数大小
                  &_state->sbtTracingFwd,  //sbtTracingFwd: Shader Binding Table（着色器绑定表）
                  rayRad.size(2),   // rayRad.size(2): 启动宽度（图像宽度）
                  rayRad.size(1),   // rayRad.size(1): 启动高度（图像高度）
                  rayRad.size(0))); // rayRad.size(0): 启动深度（Batch 大小）

CUDA_CHECK_LAST();

if (profileRaygen) {
    CUDA_CHECK(cudaEventRecord(raygenEventStop, cudaStream));
    CUDA_CHECK(cudaEventSynchronize(raygenEventStop));
    CUDA_CHECK(cudaEventElapsedTime(&_state->lastOptixLaunchMs, raygenEventStart, raygenEventStop));
    CUDA_CHECK(cudaEventDestroy(raygenEventStart));
    CUDA_CHECK(cudaEventDestroy(raygenEventStop));
}

if (_state->enableRaygenSetupProfiling && _state->raygenSetupCycleCounter) {
    CUDA_CHECK(cudaStreamSynchronize(cudaStream));
    unsigned long long setupCycles = 0ull;
    CUDA_CHECK(cudaMemcpy(&setupCycles,
                          reinterpret_cast<void*>(_state->raygenSetupCycleCounter),
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    _state->lastRaygenSetupCycles = setupCycles;
    const double numPixels =
        static_cast<double>(rayRad.size(0)) * static_cast<double>(rayRad.size(1)) *
        static_cast<double>(rayRad.size(2));
    const double smHz = static_cast<double>(_state->smClockRateKHz) * 1000.0;
    if (numPixels > 0.0 && smHz > 0.0) {
        // Mean per-pixel setup time in ms.
        _state->lastRaygenSetupMs =
            static_cast<float>((static_cast<double>(setupCycles) / numPixels) / smHz * 1000.0);
    }
}

// ============================================================
// 5. 返回所有结果张量
// ============================================================

return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, 
            torch::Tensor, torch::Tensor, torch::Tensor, 
            torch::Tensor, torch::Tensor, torch::Tensor, 
            torch::Tensor, torch::Tensor, torch::Tensor, 
            torch::Tensor>(
rayRad,              // 0: 颜色（传统渲染）
rayDns,              // 1: 累积密度
rayHit,              // 2: 命中距离
rayHitSecondMoment,  // 3: 二阶矩
rayDepthDistortion,  // 4: 深度扭曲
rayNrm,              // 5: 几何法线
rayShadingNrm,       // 6: 着色法线
rayMaterial,         // 7: 材质参数
rayHitsCount,        // 8: 命中计数
particleVisibility,  // 9: 粒子可见性
rayPbr,              // 10: PBR颜色
rayLight,            // 11: 环境光
rayPbrComponents     // 12: PBR分量分解
);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
OptixTracer::traceBwd(uint32_t frameNumber,
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
                      uint32_t maxBounces) {

    const torch::TensorOptions opts    = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor particleDensityGrad  = torch::zeros({particleDensity.size(0), particleDensity.size(1)}, opts);
    torch::Tensor particleMaterialGrad  = torch::zeros({particleMaterial.size(0), particleMaterial.size(1)}, opts);
    torch::Tensor particleRadianceGrad = torch::zeros({particleRadiance.size(0), particleRadiance.size(1)}, opts);
    torch::Tensor particleShadingNormalGrad = torch::zeros({particleShadingNormal.size(0), particleShadingNormal.size(1)}, opts);
    torch::Tensor particleVisibility = torch::zeros({particleDensity.size(0), 1}, opts);
    torch::Tensor environmentGrad = torch::zeros_like(environment);

    PipelineBackwardParameters paramsHost;
    std::memset(&paramsHost.atmosphere, 0, sizeof(paramsHost.atmosphere));
    paramsHost.atmosphere.enabled = 0;
    paramsHost.handle = _state->gasHandle;
    paramsHost.meshHandle = 0;
    paramsHost.volHandle = 0;
    paramsHost.meshVolIas = 0;
    paramsHost.volBrickId = nullptr;
    paramsHost.volPrimCount = 0;
    paramsHost.meshRepeatLow[0] = paramsHost.meshRepeatLow[1] = paramsHost.meshRepeatLow[2] = 0.f;
    paramsHost.meshRepeatUpp[0] = paramsHost.meshRepeatUpp[1] = paramsHost.meshRepeatUpp[2] = 0.f;
    paramsHost.aabb   = _state->gasAABB;

    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber   = frameNumber;
    paramsHost.gPrimNumTri   = _state->gPrimNumTri;

    paramsHost.minTransmittance       = minTransmittance;
    paramsHost.hitMinGaussianResponse = _state->particleKernelMinResponse;
    paramsHost.alphaMinThreshold      = 1.0f / 255.0f;
    paramsHost.sphDegree              = sphDegree;
    paramsHost.maxBounces             = shIndirect ? (maxBounces < 2u ? maxBounces : 2u) : maxBounces;
    paramsHost.renderOpts             = shIndirect;

    std::memcpy(&paramsHost.rayToWorld[0].x, rayToWorld.cpu().data_ptr<float>(), 3 * sizeof(float4));
    paramsHost.rayOrigin    = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDirection = packed_accessor32<float, 4>(rayDir);

    paramsHost.particleDensity      = getPtr<const ParticleDensity>(particleDensity);
    paramsHost.particleMaterial     = getPtr<const Material>(particleMaterial);
    paramsHost.particleRadiance     = getPtr<const float>(particleRadiance);
    paramsHost.particleShadingNormal = getPtr<const float>(particleShadingNormal);
    paramsHost.particleExtendedData = reinterpret_cast<const void*>(_state->gPipelineParticleData);
    paramsHost.particleVisibility   = getPtr<int32_t>(particleVisibility);
    paramsHost.raygenSetupCycleCounter = nullptr;

    paramsHost.rayRadiance    = packed_accessor32<float, 4>(rayRad);
    paramsHost.rayDensity     = packed_accessor32<float, 4>(rayDns);
    paramsHost.rayHitDistance = packed_accessor32<float, 4>(rayHit);
    paramsHost.rayHitDistanceSecondMoment = packed_accessor32<float, 4>(rayHitSecondMoment);
    paramsHost.rayDepthDistortion = packed_accessor32<float, 4>(rayDepthDistortion);
    paramsHost.rayNormal      = packed_accessor32<float, 4>(rayNrm);
    paramsHost.rayShadingNormal = packed_accessor32<float, 4>(rayShadingNrm);
    paramsHost.rayMaterial    = packed_accessor32<float, 4>(rayMaterial);
    paramsHost.rayPbr         = packed_accessor32<float, 4>(rayPbr);
    paramsHost.rayLight       = packed_accessor32<float, 4>(rayLight);

    paramsHost.particleDensityGrad  = getPtr<ParticleDensity>(particleDensityGrad);
    paramsHost.particleMaterialGrad = getPtr<Material>(particleMaterialGrad);
    paramsHost.particleRadianceGrad = getPtr<float>(particleRadianceGrad);
    paramsHost.particleShadingNormalGrad = getPtr<float>(particleShadingNormalGrad);

    paramsHost.rayRadianceGrad    = packed_accessor32<float, 4>(rayRadGrd);
    paramsHost.rayDensityGrad     = packed_accessor32<float, 4>(rayDnsGrd);
    paramsHost.rayHitDistanceGrad = packed_accessor32<float, 4>(rayHitGrd);
    paramsHost.rayHitDistanceSecondMomentGrad = packed_accessor32<float, 4>(rayHitSecondMomentGrd);
    paramsHost.rayDepthDistortionGrad = packed_accessor32<float, 4>(rayDepthDistortionGrd);
    paramsHost.rayNormalGrad      = packed_accessor32<float, 4>(rayNrmGrd);
    paramsHost.rayShadingNormalGrad = packed_accessor32<float, 4>(rayShadingNrmGrd);
    paramsHost.rayMaterialGrad    = packed_accessor32<float, 4>(rayMaterialGrd);
    paramsHost.rayPbrGrad         = packed_accessor32<float, 4>(rayPbrGrd);
    paramsHost.rayLightGrad       = packed_accessor32<float, 4>(rayLightGrd);
    paramsHost.environmentGrad    = packed_accessor32<float, 3>(environmentGrad);

    paramsHost.environment = Environment(environment, environmentAliasTable);
    paramsHost.numLights = lights.size(0);
    paramsHost.lights    = packed_accessor32<float, 2>(lights);
    paramsHost.numLightEntries = lightAliasTable.size(1);
    paramsHost.lightAliasTable = packed_accessor32<float, 2>(lightAliasTable);
    paramsHost.numMeshLightVertices = meshLightVertices.size(0);
    paramsHost.meshLightVertices = packed_accessor32<float, 2>(meshLightVertices);
    paramsHost.numMeshLightTriangles = meshLightTriangles.size(0);
    paramsHost.meshLightTriangles = packed_accessor32<int32_t, 2>(meshLightTriangles);
    paramsHost.numMeshLights = meshLights.size(0);
    paramsHost.meshLights = packed_accessor32<float, 2>(meshLights);
    paramsHost.meshLightTriangleAliasTable = packed_accessor32<float, 2>(meshLightTriangleAliasTable);

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    reallocateParamsDevice(sizeof(paramsHost), cudaStream);
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(_state->paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice, cudaStream));

    OPTIX_CHECK(optixLaunch(_state->pipelineTracingBwd, cudaStream, _state->paramsDevice,
                            sizeof(PipelineBackwardParameters), &_state->sbtTracingBwd,
                            rayRad.size(2), rayRad.size(1), rayRad.size(0)));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
        particleDensityGrad, particleMaterialGrad, particleRadianceGrad, particleShadingNormalGrad, environmentGrad);
}

namespace {

const float* atmFloatPtr(const torch::Tensor& t) {
    if (!t.defined() || t.numel() == 0) {
        return nullptr;
    }
    return t.data_ptr<float>();
}

const int* atmIntPtr(const torch::Tensor& t) {
    if (!t.defined() || t.numel() == 0) {
        return nullptr;
    }
    return t.data_ptr<int>();
}

void atmCopy3(const torch::Tensor& t, float out[3], float x, float y, float z) {
    out[0] = x;
    out[1] = y;
    out[2] = z;
    if (!t.defined() || t.numel() < 3) {
        return;
    }
    const torch::Tensor h = t.detach().cpu().contiguous();
    const float* p = h.data_ptr<float>();
    out[0] = p[0];
    out[1] = p[1];
    out[2] = p[2];
}

} // namespace

void OptixTracer::buildAtmosphereAccel(
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
    cudaStream_t stream) {
    State* st = _state;

    auto reallocDevice = [&](CUdeviceptr* ptr, size_t* sz, size_t need) {
        if (*sz >= need && *ptr) {
            return;
        }
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(*ptr), stream));
        *ptr = 0;
        *sz = 0;
        if (need == 0) {
            return;
        }
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(ptr), need, stream));
        *sz = need;
    };

    auto buildGASFromInput = [&](const OptixBuildInput& input,
                                 OptixTraversableHandle* outHandle,
                                 CUdeviceptr* outBuffer,
                                 size_t* outBufferSz) {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(st->context, &accel_options, &input, 1, &sizes));
        if (st->gasBufferTmpSz < sizes.tempSizeInBytes) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(st->gasBufferTmp), stream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&st->gasBufferTmp), sizes.tempSizeInBytes, stream));
            st->gasBufferTmpSz = sizes.tempSizeInBytes;
        }
        reallocDevice(outBuffer, outBufferSz, sizes.outputSizeInBytes);
        OPTIX_CHECK(optixAccelBuild(
            st->context,
            stream,
            &accel_options,
            &input,
            1,
            st->gasBufferTmp,
            sizes.tempSizeInBytes,
            *outBuffer,
            *outBufferSz,
            outHandle,
            nullptr,
            0));
    };

    if (!meshVerts.defined() || !meshIdx.defined() || meshIdx.numel() < 3) {
        st->meshGasHandle = 0;
        st->meshGasNtri = 0;
        st->meshGasNvert = 0;
        st->meshGasVertsPtr = 0;
        st->meshGasIdxPtr = 0;
    } else {
        meshVerts = meshVerts.contiguous();
        meshIdx = meshIdx.contiguous();
        const int nvert = static_cast<int>(meshVerts.size(0));
        const int ntri = static_cast<int>(meshIdx.size(0));
        const CUdeviceptr vptr = reinterpret_cast<CUdeviceptr>(meshVerts.data_ptr());
        const CUdeviceptr iptr = reinterpret_cast<CUdeviceptr>(meshIdx.data_ptr());
        if (!(st->meshGasHandle && st->meshGasNtri == ntri && st->meshGasNvert == nvert &&
              st->meshGasVertsPtr == vptr && st->meshGasIdxPtr == iptr)) {
            CUdeviceptr vertexBuffer = vptr;
            uint32_t geomFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            OptixBuildInput input = {};
            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            input.triangleArray.numVertices = static_cast<unsigned int>(nvert);
            input.triangleArray.vertexBuffers = &vertexBuffer;
            input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            input.triangleArray.numIndexTriplets = static_cast<unsigned int>(ntri);
            input.triangleArray.indexBuffer = iptr;
            input.triangleArray.flags = &geomFlags;
            input.triangleArray.numSbtRecords = 1;
            buildGASFromInput(input, &st->meshGasHandle, &st->meshGasBuffer, &st->meshGasBufferSz);
            st->meshGasNtri = ntri;
            st->meshGasNvert = nvert;
            st->meshGasVertsPtr = vptr;
            st->meshGasIdxPtr = iptr;
            std::fprintf(stderr, "[atm] mesh GAS nvert=%d ntri=%d\n", nvert, ntri);
        }
    }

    AtmVoxelGrid grid = {};
    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.brick = brick > 0 ? brick : 8;
    grid.nbx = nbx;
    grid.nby = nby;
    grid.nbz = nbz;
    grid.origin[0] = origin[0];
    grid.origin[1] = origin[1];
    grid.origin[2] = origin[2];
    grid.dx = dx;
    grid.dy = dy;
    grid.dz = dz;

    if (!voxelKmax.defined() || voxelKmax.numel() == 0 || grid.nbx <= 0) {
        st->volGasHandle = 0;
        st->volPrimCount = 0;
        st->volKmaxPtr = 0;
    } else {
        const CUdeviceptr kptr = reinterpret_cast<CUdeviceptr>(voxelKmax.data_ptr());
        if (!(st->volGasHandle && st->volKmaxPtr == kptr && st->volPrimCount == 1 && st->volAabbWrap == 2)) {
            const torch::Tensor kHost = voxelKmax.detach().cpu().contiguous();
            const float* kmax = kHost.data_ptr<float>();
            int any = 0;
            const int nbrick = grid.nbx * grid.nby * grid.nbz;
            for (int i = 0; i < nbrick; ++i) {
                if (kmax[i] > 1e-12f) {
                    any = 1;
                    break;
                }
            }
            if (!any) {
                st->volGasHandle = 0;
                st->volPrimCount = 0;
                st->volKmaxPtr = kptr;
                st->volAabbWrap = 2;
                std::fprintf(stderr, "[atm] vol GAS empty (all bricks kmax=0)\n");
            } else {
                /* Wide XY + tall Z so camera-outside and sky rays still enter IS.
                 * IS Woodcock clips the cloud Z-slab and wraps XY; gas walks tmax. */
                const float span = 2.0e5f;
                const float zTop = fmaxf(grid.origin[2] + (float)grid.nz * grid.dz, 1.0e5f);
                OptixAabb box;
                box.minX = grid.origin[0] - span;
                box.minY = grid.origin[1] - span;
                box.minZ = fminf(grid.origin[2], 0.f);
                box.maxX = grid.origin[0] + (float)grid.nx * grid.dx + span;
                box.maxY = grid.origin[1] + (float)grid.ny * grid.dy + span;
                box.maxZ = zTop;
                const int brick0 = 0;
                reallocDevice(&st->volAabbBuffer, &st->volAabbBufferSz, sizeof(OptixAabb));
                reallocDevice(&st->volBrickIdBuffer, &st->volBrickIdBufferSz, sizeof(int));
                CUDA_CHECK(cudaMemcpyAsync(
                    reinterpret_cast<void*>(st->volAabbBuffer), &box, sizeof(box), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    reinterpret_cast<void*>(st->volBrickIdBuffer), &brick0, sizeof(brick0), cudaMemcpyHostToDevice, stream));

                uint32_t geomFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
                OptixBuildInput input = {};
                input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
                input.customPrimitiveArray.aabbBuffers = &st->volAabbBuffer;
                input.customPrimitiveArray.numPrimitives = 1;
                input.customPrimitiveArray.strideInBytes = 0;
                input.customPrimitiveArray.flags = &geomFlags;
                input.customPrimitiveArray.numSbtRecords = 1;
                buildGASFromInput(input, &st->volGasHandle, &st->volGasBuffer, &st->volGasBufferSz);
                st->volPrimCount = 1;
                st->volKmaxPtr = kptr;
                st->volAabbWrap = 2;
                std::fprintf(stderr, "[atm] vol GAS 1 AABB (Z-atm + wrap XY) / %d bricks\n", nbrick);
            }
        }
    }

    /* Volume and mesh stay separate GAS traces (htrdr Embree ∥ SVX). */
    st->meshVolIasHandle = 0;
    std::fprintf(stderr, "[atm] IAS disabled (separate mesh GAS + vol GAS traces)\n");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
OptixTracer::traceAtmosphere(
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
    torch::Tensor cieCdf,
    torch::Tensor cieRcpIntegral,
    float cieLambda0Nm,
    float cieDlambdaNm,
    torch::Tensor swLambdaLo,
    torch::Tensor swLambdaHi,
    torch::Tensor swGasKa,
    torch::Tensor swGasKs,
    torch::Tensor brdfLambdaNm,
    torch::Tensor brdfR,
    torch::Tensor swNquad,
    torch::Tensor swQuadCdf,
    torch::Tensor mieCsBand,
    torch::Tensor mieCaBand,
    torch::Tensor mieGnm,
    float mieCsRef,
    float mieCaRef,
    torch::Tensor voxelZIface) {

    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor rayRad = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayDns = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor rayHit = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 2}, opts);
    torch::Tensor rayHitSecondMoment = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor rayDepthDistortion = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor rayNrm = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayShadingNrm = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayMaterial = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 5}, opts);
    torch::Tensor rayHitsCount = torch::zeros({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor rayPbr = torch::zeros({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayLight = torch::zeros({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayPbrComponents = torch::zeros({rayOri.size(0), rayOri.size(1), rayOri.size(2), 2, 3}, opts);

    const int64_t nPart = particleDensity.defined() ? particleDensity.size(0) : 0;
    torch::Tensor particleVisibility = torch::zeros({nPart, 1}, opts);

    torch::Tensor environment = torch::zeros({1, 1, 4}, opts);
    torch::Tensor environmentAliasTable = torch::empty({3, 0, 0}, opts);
    torch::Tensor lights = torch::empty({0, 9}, opts);
    torch::Tensor lightAliasTable = torch::empty({5, 0}, opts);
    torch::Tensor meshLightVertices = torch::empty({0, 3}, opts);
    torch::Tensor meshLightTriangles = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor meshLights = torch::empty({0, 8}, opts);
    torch::Tensor meshLightTriangleAliasTable = torch::empty({3, 0}, opts);

    PipelineParameters paramsHost;
    std::memset(&paramsHost, 0, sizeof(paramsHost));

    paramsHost.handle = _state->gasHandle;
    paramsHost.aabb = _state->gasAABB;
    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber = frameNumber;
    paramsHost.gPrimNumTri = _state->gPrimNumTri;
    paramsHost.minTransmittance = 1e-3f;
    paramsHost.hitMinGaussianResponse = _state->particleKernelMinResponse;
    paramsHost.alphaMinThreshold = 1.0f / 255.0f;
    paramsHost.sphDegree = 0;
    paramsHost.maxBounces = maxBounces;
    paramsHost.renderOpts = 0;

    std::memcpy(&paramsHost.rayToWorld[0].x, rayToWorld.cpu().data_ptr<float>(), 3 * sizeof(float4));

    paramsHost.rayOrigin = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDirection = packed_accessor32<float, 4>(rayDir);

    if (nPart > 0) {
        paramsHost.particleDensity = getPtr<const ParticleDensity>(particleDensity);
        paramsHost.particleMaterial = getPtr<const Material>(particleMaterial);
        paramsHost.particleRadiance = getPtr<const float>(particleRadiance);
        paramsHost.particleShadingNormal = getPtr<const float>(particleShadingNormal);
        paramsHost.particleVisibility = getPtr<int32_t>(particleVisibility);
    }
    paramsHost.particleExtendedData = reinterpret_cast<const void*>(_state->gPipelineParticleData);

    paramsHost.rayRadiance = packed_accessor32<float, 4>(rayRad);
    paramsHost.rayDensity = packed_accessor32<float, 4>(rayDns);
    paramsHost.rayHitDistance = packed_accessor32<float, 4>(rayHit);
    paramsHost.rayHitDistanceSecondMoment = packed_accessor32<float, 4>(rayHitSecondMoment);
    paramsHost.rayDepthDistortion = packed_accessor32<float, 4>(rayDepthDistortion);
    paramsHost.rayNormal = packed_accessor32<float, 4>(rayNrm);
    paramsHost.rayShadingNormal = packed_accessor32<float, 4>(rayShadingNrm);
    paramsHost.rayMaterial = packed_accessor32<float, 4>(rayMaterial);
    paramsHost.rayHitsCount = packed_accessor32<float, 4>(rayHitsCount);
    paramsHost.rayPbr = packed_accessor32<float, 4>(rayPbr);
    paramsHost.rayLight = packed_accessor32<float, 4>(rayLight);
    paramsHost.rayPbrComponents = packed_accessor32<float, 5>(rayPbrComponents);

    paramsHost.environment = Environment(environment, environmentAliasTable);
    paramsHost.numLights = 0;
    paramsHost.lights = packed_accessor32<float, 2>(lights);
    paramsHost.numLightEntries = 0;
    paramsHost.lightAliasTable = packed_accessor32<float, 2>(lightAliasTable);
    paramsHost.numMeshLightVertices = 0;
    paramsHost.meshLightVertices = packed_accessor32<float, 2>(meshLightVertices);
    paramsHost.numMeshLightTriangles = 0;
    paramsHost.meshLightTriangles = packed_accessor32<int32_t, 2>(meshLightTriangles);
    paramsHost.numMeshLights = 0;
    paramsHost.meshLights = packed_accessor32<float, 2>(meshLights);
    paramsHost.meshLightTriangleAliasTable = packed_accessor32<float, 2>(meshLightTriangleAliasTable);

    AtmParams& atm = paramsHost.atmosphere;
    atm.enabled = 1;
    atm.gs_enabled = (nPart > 0 && _state->gasHandle != 0) ? 1 : 0;
    atm.spectral_mode = spectralMode;
    atm.wavelength_m = wavelength_m > 0.f ? wavelength_m : 5.0e-7f;
    atm.mie_omega = mieOmega;
    atm.mie_g = mieG;
    atm.path_tmax = pathTmax > 0.f ? pathTmax : 1.0e5f;
    atm.max_bounces = static_cast<int>(maxBounces > 0 ? maxBounces : 8u);

    if (voxelKa.defined() && voxelKa.dim() == 3 && voxelKa.numel() > 0) {
        atm.voxels.nz = static_cast<int>(voxelKa.size(0));
        atm.voxels.ny = static_cast<int>(voxelKa.size(1));
        atm.voxels.nx = static_cast<int>(voxelKa.size(2));
        atm.voxels.brick = brick > 0 ? brick : 8;
        atm.voxels.nbx = (atm.voxels.nx + atm.voxels.brick - 1) / atm.voxels.brick;
        atm.voxels.nby = (atm.voxels.ny + atm.voxels.brick - 1) / atm.voxels.brick;
        atm.voxels.nbz = (atm.voxels.nz + atm.voxels.brick - 1) / atm.voxels.brick;
        atmCopy3(voxelOrigin, atm.voxels.origin, 0.f, 0.f, 0.f);
        float sp[3];
        atmCopy3(voxelSpacing, sp, 1.f, 1.f, 1.f);
        atm.voxels.dx = sp[0];
        atm.voxels.dy = sp[1];
        atm.voxels.dz = sp[2];
        atm.voxels.ka = atmFloatPtr(voxelKa);
        atm.voxels.ks = atmFloatPtr(voxelKs);
        atm.voxels.temperature = atmFloatPtr(voxelTemp);
        atm.voxels.kext_max = atmFloatPtr(voxelKmax);
    }

    if (gasZ.defined() && gasZ.numel() >= 2 && gasKa.defined() && gasKa.numel() > 0) {
        atm.gas.nlayers = static_cast<int>(gasKa.numel());
        atm.gas.z_iface = atmFloatPtr(gasZ);
        atm.gas.ka = atmFloatPtr(gasKa);
        atm.gas.ks = atmFloatPtr(gasKs);
        atm.gas.temperature = atmFloatPtr(gasTemp);
    }

    if (meshVerts.defined() && meshIdx.defined() && meshIdx.numel() >= 3) {
        atm.mesh.nvert = static_cast<int>(meshVerts.size(0));
        atm.mesh.ntri = static_cast<int>(meshIdx.size(0));
        atm.mesh.vertices = atmFloatPtr(meshVerts);
        atm.mesh.indices = atmIntPtr(meshIdx);
        atm.mesh.albedo = atmFloatPtr(meshAlbedo);
        atm.mesh.temperature = atmFloatPtr(meshTemp);
    }

    atmCopy3(sunDir, atm.sun.dir, 0.f, 0.f, 1.f);
    const float sn = std::sqrt(atm.sun.dir[0] * atm.sun.dir[0] + atm.sun.dir[1] * atm.sun.dir[1] + atm.sun.dir[2] * atm.sun.dir[2]);
    if (sn > 1e-12f) {
        atm.sun.dir[0] /= sn;
        atm.sun.dir[1] /= sn;
        atm.sun.dir[2] /= sn;
    }
    atm.sun.radiance = sunRadiance;
    atm.sun.cos_half_angle = sunCosHalf;

    atm.cie_enabled = 0;
    atm.cie_nbins = 0;
    atm.cie_lambda0_nm = cieLambda0Nm;
    atm.cie_dlambda_nm = cieDlambdaNm;
    atm.cie_rcp_integral[0] = atm.cie_rcp_integral[1] = atm.cie_rcp_integral[2] = 0.f;
    atm.cie_cdf = nullptr;
    atm.sw_nbands = 0;
    atm.sw_nquad_max = 0;
    atm.sw_lambda_lo_nm = nullptr;
    atm.sw_lambda_hi_nm = nullptr;
    atm.sw_gas_ka = nullptr;
    atm.sw_gas_ks = nullptr;
    if (cieCdf.defined() && cieCdf.numel() >= 3 && cieRcpIntegral.defined() && cieRcpIntegral.numel() >= 3) {
        const int ncdf = static_cast<int>(cieCdf.numel());
        atm.cie_nbins = ncdf / 3;
        atm.cie_cdf = atmFloatPtr(cieCdf);
        const auto rcpCpu = cieRcpIntegral.contiguous().cpu();
        const float* rcp = rcpCpu.data_ptr<float>();
        atm.cie_rcp_integral[0] = rcp[0];
        atm.cie_rcp_integral[1] = rcp[1];
        atm.cie_rcp_integral[2] = rcp[2];
        atm.cie_enabled = (atm.cie_nbins > 0 && atm.cie_cdf != nullptr) ? 1 : 0;
    }
    if (swLambdaLo.defined() && swGasKa.defined() && swGasKa.numel() > 0) {
        if (swGasKa.dim() == 3) {
            atm.sw_nbands = static_cast<int>(swGasKa.size(0));
            atm.sw_nquad_max = static_cast<int>(swGasKa.size(1));
        } else {
            atm.sw_nbands = static_cast<int>(swGasKa.size(0));
            atm.sw_nquad_max = 1;
        }
        atm.sw_lambda_lo_nm = atmFloatPtr(swLambdaLo);
        atm.sw_lambda_hi_nm = atmFloatPtr(swLambdaHi);
        atm.sw_gas_ka = atmFloatPtr(swGasKa);
        atm.sw_gas_ks = atmFloatPtr(swGasKs);
    }
    atm.sw_nquad = nullptr;
    atm.sw_quad_cdf = nullptr;
    atm.mie_cs_band = nullptr;
    atm.mie_ca_band = nullptr;
    atm.mie_g_nm = nullptr;
    atm.mie_g_n = 0;
    atm.mie_g_lambda0_nm = 380.f;
    atm.mie_g_dlambda_nm = 1.f;
    atm.mie_cs_ref = mieCsRef > 0.f ? mieCsRef : 1.f;
    atm.mie_ca_ref = mieCaRef > 0.f ? mieCaRef : 1.f;
    atm.path_ks_scale = 1.f;
    atm.path_ka_scale = 1.f;
    atm.voxel_z_iface = nullptr;
    if (voxelZIface.defined() && voxelZIface.numel() >= static_cast<int64_t>(atm.voxels.nz) + 1) {
        atm.voxel_z_iface = atmFloatPtr(voxelZIface);
    }
    if (swNquad.defined() && swNquad.numel() > 0) {
        atm.sw_nquad = atmIntPtr(swNquad);
    }
    if (swQuadCdf.defined() && swQuadCdf.numel() > 0) {
        atm.sw_quad_cdf = atmFloatPtr(swQuadCdf);
    }
    if (mieCsBand.defined() && mieCsBand.numel() > 0) {
        atm.mie_cs_band = atmFloatPtr(mieCsBand);
    }
    if (mieCaBand.defined() && mieCaBand.numel() > 0) {
        atm.mie_ca_band = atmFloatPtr(mieCaBand);
    }
    if (mieGnm.defined() && mieGnm.numel() > 1) {
        atm.mie_g_nm = atmFloatPtr(mieGnm);
        atm.mie_g_n = static_cast<int>(mieGnm.numel());
        atm.mie_g_lambda0_nm = 380.f;
        atm.mie_g_dlambda_nm = 400.f / float(atm.mie_g_n - 1);
    }
    atm.brdf_n = 0;
    atm.brdf_lambda_nm = nullptr;
    atm.brdf_R = nullptr;
    atm.path_albedo = -1.f;
    if (brdfLambdaNm.defined() && brdfR.defined() && brdfLambdaNm.numel() > 0 && brdfR.numel() == brdfLambdaNm.numel()) {
        atm.brdf_n = static_cast<int>(brdfLambdaNm.numel());
        atm.brdf_lambda_nm = atmFloatPtr(brdfLambdaNm);
        atm.brdf_R = atmFloatPtr(brdfR);
    }

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();
    buildAtmosphereAccel(
        meshVerts,
        meshIdx,
        voxelKmax,
        atm.voxels.nx,
        atm.voxels.ny,
        atm.voxels.nz,
        atm.voxels.brick,
        atm.voxels.nbx,
        atm.voxels.nby,
        atm.voxels.nbz,
        atm.voxels.origin,
        atm.voxels.dx,
        atm.voxels.dy,
        atm.voxels.dz,
        cudaStream);
    paramsHost.meshHandle = _state->meshGasHandle;
    paramsHost.volHandle = _state->volGasHandle;
    paramsHost.meshVolIas = 0;
    paramsHost.volBrickId = _state->volBrickIdBuffer
                                ? reinterpret_cast<const int*>(_state->volBrickIdBuffer)
                                : nullptr;
    paramsHost.volPrimCount = _state->volPrimCount;
    if (meshVerts.defined() && meshVerts.numel() >= 3) {
        const torch::Tensor mnH = std::get<0>(meshVerts.min(0)).detach().cpu().contiguous();
        const torch::Tensor mxH = std::get<0>(meshVerts.max(0)).detach().cpu().contiguous();
        const float* a = mnH.data_ptr<float>();
        const float* b = mxH.data_ptr<float>();
        paramsHost.meshRepeatLow[0] = a[0];
        paramsHost.meshRepeatLow[1] = a[1];
        paramsHost.meshRepeatLow[2] = a[2];
        paramsHost.meshRepeatUpp[0] = b[0];
        paramsHost.meshRepeatUpp[1] = b[1];
        paramsHost.meshRepeatUpp[2] = b[2];
        if (paramsHost.meshRepeatUpp[2] - paramsHost.meshRepeatLow[2] < 1.f) {
            paramsHost.meshRepeatLow[2] -= 1.f;
            paramsHost.meshRepeatUpp[2] += 1.f;
        }
        std::fprintf(stderr, "[atm] mesh -R AABB [%.1f,%.1f,%.1f] / [%.1f,%.1f,%.1f]\n",
                     (double)a[0], (double)a[1], (double)a[2],
                     (double)b[0], (double)b[1], (double)b[2]);
    }

    std::fprintf(stderr,
                 "[atm] sizeof PipelineParameters=%zu AtmParams=%zu sunL=%g mesh_ntri=%d voxels=%dx%dx%d gs=%d "
                 "meshGAS=%d volGAS=%d ias=%d volPrims=%d gas_nlayers=%d gas_ks=%d path_tmax=%g z_iface=%d\n",
                 sizeof(PipelineParameters), sizeof(AtmParams), (double)atm.sun.radiance,
                 atm.mesh.ntri, atm.voxels.nx, atm.voxels.ny, atm.voxels.nz, atm.gs_enabled,
                 paramsHost.meshHandle != 0, paramsHost.volHandle != 0, paramsHost.meshVolIas != 0,
                 paramsHost.volPrimCount, atm.gas.nlayers, atm.gas.ks != nullptr, (double)atm.path_tmax,
                 atm.voxel_z_iface != nullptr);

    reallocateParamsDevice(sizeof(paramsHost), cudaStream);
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(_state->paramsDevice),
        &paramsHost,
        sizeof(paramsHost),
        cudaMemcpyHostToDevice,
        cudaStream));

    OPTIX_CHECK(optixLaunch(
        _state->pipelineTracingFwd,
        cudaStream,
        _state->paramsDevice,
        sizeof(PipelineParameters),
        &_state->sbtTracingFwd,
        rayRad.size(2),
        rayRad.size(1),
        rayRad.size(0)));
    CUDA_CHECK_LAST();

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor>(
        rayRad, rayDns, rayHit, rayHitSecondMoment, rayDepthDistortion,
        rayNrm, rayShadingNrm, rayMaterial, rayHitsCount, particleVisibility,
        rayPbr, rayLight, rayPbrComponents);
}
