# GPU 实现 vs `methods.md`（htrdr CPU）

对照对象：

- 文档：`docs/methods.md` = htrdr 0.12 + Starter Pack 0.8.0，主场景 `mountain.sh`
- GPU：`gpu/threedgptir_tracer/`（OptiX），入口 `tools/render_mountain_gs_gpu.py`

GPU **不是** Meso-Star 官方 htrdr-gpu，而是本`仓库为对齐 mountain.sh` 短波 CIE 图写的路径追踪器。下面按 `methods.md` 的条目列差异；未写的条目视为主干意图一致。

---

## 0. 范围


|        | `methods.md` / htrdr               | GPU                                                           |
| ------ | ---------------------------------- | ------------------------------------------------------------- |
| 求解器    | 大气 SW + LW、透视/正交/薄透镜/通量图、多地面几何     | 短波 CIE 图像为主；正交相机可选；**无薄透镜、无通量图**                              |
| 地面     | OBJ + Embree，任意 MruMtl             | `--ground mesh`：OptiX 三角（山）；`--ground gs`：**3DGS 额外图元**（文档没有） |
| 精度     | `double`                           | 全程 `float`                                                    |
| 默认 spp | `mountain.sh` 256（每 CIE 通道）        | 脚本默认 **16**；对齐实验需显式 `--spp 256`                               |
| 路径深度   | 体积无 bounce 上限，直到 miss / 吸收 / 表面 RR | `max_bounces` 默认 **256**（本场景与 1024 窗均值相同）；表面另有 `bounce>=8` 的权俄罗斯轮盘 |


对齐 `mountain.sh` 的推荐调用：`--camera mountain --ground mesh --spp 256`（GS 关掉）。

---



## 1. 图元



### 1.1 地面

文档：`mountain.obj` 838 万三角，`air:sand`，`-R` XY 周期，Embree + slab（最多 32 格）。

GPU `--ground mesh`：同一 OBJ（经 `ensure_mountain_mesh`），OptiX GAS + `atm_trace_mesh_slab`（同样 32 步）。周期 AABB 在 `meshRepeatLow/Upp`。

差异：

- GPU 还有 `--ground gs`：把山变成 3DGS，再用两三角形补洞。文档图元表里没有 Gaussian。
- GS 模式：`org.z > 850` 时 **不打 GS**；向下走时把体积 `t_surf` 裁到 z=850，好让路径进入 GS 带。这是工程裁剪，htrdr 没有「850 m 地形带」。
- 表面自交：GPU NEE / bounce 沿法线偏置 **2 m** 再沿出射方向 0.05 m。htrdr 用 `s3d_hit_prev` 滤同一图元。
- 非 CIE 或 `path_albedo<0` 时，GPU 可走 PTIR **Disney/GGX**（`eval_material_fast_brdf_`*，F0=0.04）。文档地面是 Lambert / 镜面 MruMtl。CIE + `sand.mrumtl` 时 GPU 强制标量 R(\lambda)，与文档一致。

海面镜面、城市 SLUM、沙漠几何：GPU 这条管线 **未接**。

### 1.2 云

文档：HTCP **不规则 Z**（160 层），查询时用 `svx2htcp_z`；光学系数当场 \rho_{\mathrm{da}}\cdot\mathrm{RCT}\cdot\overline{C}。

GPU：`load_htcp_voxels` 把不规则 Z **重采样成均匀** `dz` 的规则网格（默认 `nreg=nz`，范围取 HTCP `[lower_z,upper_z]`）。`AtmVoxelGrid` ABI **不能**存不规则层厚。

- 原生层厚变化被摊进均匀体素；文档明确 GPU「只能均匀 dz」。
- 体素 `ka/ks` 先按 **550 nm** Mie 截面烘焙，路径上再乘 `path_ks_scale = Cs_band/Cs_ref`（整场一个因子）。htrdr 每谱段用该段 \overline{C}_{a,s} 直接算 q_l C。期望接近，实现不是同一套查询。
- 无 HTCP dump 时 GPU 可改用 **程序化 fbm 积云**（`fbm_cumulus`），与文档 LES 场无关。



### 1.3 气体

文档：1D 层 + 云内按 LES x_{\mathrm{H_2O}} 插值 k。

GPU：`AtmGasLayers` 只有名义 `ka/ks(z)`（CIE 路径上换成当前 `(iband,iquad)` 的 HTGOP 层表）。**云盒子里不读 RCT/RVT，不做 x_{\mathrm{H_2O}} 插值。**

### 1.4 加速结构

文档：云 **八叉树**（光学厚度 \tilde\tau=1 合并），气体 **Z 二叉树**，每 (谱段, quadrature) 一棵；`-O` cache。

GPU：

- 云：规则 **8³ 砖**，砖 majorant = `max(ks)`（SW），无 \tilde\tau 合并、无 per-quad 多树。
- 气体：逐层常数，Woodcock 扫层，无 SVX 树。
- CIE 时 **不用** `volHandle` OptiX 体积 IS（注释：IS 读的是 host 550 nm 气体，不是 per-path 谱段）。改走 `atm_next_volume_event_into`。



### 1.5 太阳

半角 4.6524\times10^{-3}、球冠采样、pdf =1/\Omega、CIE 下 L_\odot=B_\lambda(5778\mathrm{K})：与文档一致。

差异：`AtmosphereBuffers.from_numpy` 仍传入占位 `sun_radiance=24.0`；CIE 路径上 `atm_cie_bind_path` **覆盖**为 Planck。非 CIE 单色会用这个 24，文档则始终是 Planck(\lambda)。

---



## 2. 无偏 MC



### 2.2 自由程

文档：在 **合并场** K_s=k_{s,\mathrm{gas}}+k_{s,\mathrm{particle}} 上 **一次** Woodcock（htsky 树 leaf 的 k_{\max}）。

GPU：气体、粒子 **两路独立** Woodcock，再取先发生的事件（竞争风险）。泊松过程上期望等价于合场首次碰撞，但：

- majorant 不同（层 k_s vs 砖 `max(ks_particle)`，不是合场 k_{\max}）；
- 云内气体仍是 1D 名义值（见上）。

SW 接受概率 k_s/\hat k、零碰撞重抽 \tau：与文档相同。

### 2.3 透过率

文档：**分解追踪** — analog T_{\min} + 残差 delta-tracking；无碰则 \mathrm{e}^{-T_{\min}}，有碰则 **0**。用于 `Kext`（太阳）和 `Ka`（吸收权）。

GPU：`atm_optical_depth` **DDA analog** \tau=\int kds，`Tr=\mathrm{e}^{-\tau}`（\tau>80 截成 0）。无残差 Bernoulli。

- 体素/层为分片常数时，GPU 的 \mathrm{e}^{-\int k} 是真透过率（无这项 MC 噪声）；htrdr 估计无偏但方差大。
- **不是同一估计量**；期望在「k 分片常数且 DDA 不漏格」时才对齐。
- 不规则 Z 被均匀化后，GPU 的 analog \tau 与 htrdr 线积分可以有几何差。



### 2.4 气 vs 粒

文档：合场碰撞后 **掷一次** k_{s,\mathrm{gas}}/K_s。

GPU：谁先到用哪路相函数（`ATM_EVENT_SCATTER_GAS` → Rayleigh，否则 HG）。不再二次 Bernoulli（`ATM_KERNEL_REV 40`）。与文档「只掷一次」意图一致，实现是两时钟而非合场后再抽。

### 2.5 NEE / 路径权

直射：相机方向在太阳锥内则 L_\odot\mathrm{Tr}(K_{\mathrm{ext}})，miss 不再加日盘 — 与文档一致。

体积 NEE：`phase * L * Tr / pdf`，HG / Rayleigh — 公式与文档一致。

地面 NEE：CIE 沙面 R/\pi\cdot(\mathbf{n}\cdot\boldsymbol{\omega}_\odot) — 与文档 Lambert 一致。

差异：

- 体积 NEE 遮挡：**只** mesh（`atm_sun_visibility_volume`），不用 GS 影。文档只有三角地面。
- 表面 NEE：`atm_sun_visibility` = mesh **再加 GS**（若 `gs_enabled`）。文档无 GS。`--ground mesh` 时 GS 关，与文档一致。
- `path_tmax` GPU 默认 **1.2×10⁵ m**；文档太阳可见性 `range` 到 `FLT_MAX`。斜射长路径可能被提前截断。
- 表面 Lambert：GPU `atm_lambert_bounce` **权乘 R 后必弹**（R\approx0 才停）。文档以 P=R 俄罗斯轮盘，存活不乘 R。Lambert 下期望相同，方差不同。
- GPU 额外：非体积碰撞且 `bounce>=8` 再做权 RR。文档没有这条。



### 2.6 长波

内核有 `ATM_SPECTRAL_LW`（K_{\mathrm{ext}} 游走、吸收处 Planck）。`render_mountain_gs_gpu.py` **固定 CIE 短波。** GS 命中时 LW 温度写死 **288.15 K**，不是 HTCP / mtl 300 K。不能当文档 §9 的实现。

---



## 3–7. 采样细节


| 项目               | 文档                                      | GPU                                                                       |
| ---------------- | --------------------------------------- | ------------------------------------------------------------------------- |
| CIE 配色           | Wyman 2013，pdf=`rcp_integral`           | 同拟合、同 pdf（×10⁹ 成 m⁻¹）                                                     |
| 谱段 / k 点         | `htsky_find_spectral_band` + quadrature | `atm_sw_find_band` + `sw_quad_cdf`                                        |
| g(\lambda)       | `htmie_fetch_asymmetry` 线性              | `mie_g_nm` LUT 线性插值                                                       |
| Rayleigh / HG 公式 | 与 `ssf` 相同形式                            | `atm_rayleigh_phase` / `atm_hg_phase`                                     |
| 沙 BRDF           | `mrumtl_fetch_brdf` 节点间掷一次              | `atm_mrumtl_fetch_R` 同规则                                                  |
| 云 XY 周期          | `world_to_cloud`                        | 体素 `atm_imod` XY；Z 有限 slab                                                |
| 气体+云射线           | 先气、再云盒、再气；云 `-r` 时 slab                 | 两路独立 walk，XY 周期、Z 裁云盒                                                     |
| 相机               | `scam` 针孔，胶片抖动                          | `make_persp_rays` 同 pos/tgt/fov=30；**默认** `--camera ortho` 不是 mountain.sh |


---



## 8. 短波管线对照

文档伪代码在 GPU `traceAtmospherePath` 里大致都有，但插入了文档没有的步骤：

1. **GS closest**（可关）
2. **z>850 跳过 GS / 裁体积**（仅 GS 模式）
3. 透过率 analog 而非分解追踪
4. `atm_add_sky_surface`：非 CIE 时垫 `0.08 * L_sun`。文档 §5.2 / §15：无环境光贴图、无 miss 垫色。**CIE 开启时该函数直接 return**，对齐 `mountain.sh`。
5. bounce 上限 + 表面二次 RR

---



## 9. 文档有、GPU 没有

- 长波图像 / 亮温
- 薄透镜、通量图、海面镜面、城市多材料
- htsky 八叉树 cache（`-O`）、\tilde\tau、`-V`
- 云内水汽插值
- 不规则 Z 原生查询
- 灵敏度、前向通量、论文 §5 算法（文档也标明不是 GPU 范围）



## GPU 有、文档没有

- 3DGS 地面与 z=850 裁剪
- PTIR Disney BRDF 回退
- 程序化 fbm 云
- `--no-clouds`、`--crop`
- 砖 majorant + 双时钟体积
- analog DDA \mathrm{e}^{-\tau}
- `ATM_KERNEL_REV` / 清 OptiX cache（改核必须）

---



## 10. 文件对照


| 文档模块                                                  | GPU                                                                 |
| ----------------------------------------------------- | ------------------------------------------------------------------- |
| `atmosphere_compute_radiance_sw.c`                    | `atmospherePathTracer.cuh` `traceAtmospherePath`                    |
| `scattering_hit_filter` / `transmissivity_hit_filter` | `atmosphere.cuh` Woodcock + analog DDA（**不是** decomposition filter） |
| `htsky` 八叉树                                           | 无；`AtmVoxelGrid` + 8³ `kext_max`                                    |
| `htrdr_atmosphere_sun.c`                              | `atm_sample_sun` / `atm_dir_in_sun_cone`                            |
| `htrdr_atmosphere_ground.c`                           | OptiX mesh slab；可选 GS                                               |
| `htrdr_ran_wlen_cie_xyz.c`                            | `atm_cie_*` + `atmosphere.py` `build_cie_xyz_tables`                |
| `mountain.sh`                                         | `tools/render_mountain_gs_gpu.py` + `tools/mountain_gs_scene.py`    |


---



## 11. 对齐时不要搞混的点

1. **默认相机是 ortho，不是** `mountain.sh`**。** 要透视必须 `--camera mountain`。
2. **默认 spp=16。** 文档 256。
3. `--ground gs` **不是 htrdr 图元。** 比文档多一层面、多 850 m 逻辑。
4. **透过率估计量不同**（analog vs 0/`e^{-Tmin}`），不能把像素噪声差都当成 k 场错误。
5. **非 CIE 的 0.08 天空垫** 违反文档；CIE `mountain` 路径已关掉。
6. 改 `atmosphere*.cuh` 后清 `/var/tmp/OptixCache_*` 并提高 `ATM_KERNEL_REV`。

