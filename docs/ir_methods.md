# GPU 短波 / 中波 / 长波成像

> 准绳：当前 GPU OptiX 核（`ATM_KERNEL_REV 79`）+ 主机脚本**实际在算什么**，不是「应该算什么」。  
> CIE 可见光 / 山体 GS 见 `methods.md`。本文只写 FIRE 正视三条红外：短波 **1600 nm**、中波 **4000 nm**、长波 **10500 nm**。  
> 单位：长度 1 = 1 m；温度 K；\(B_\lambda\)、\(L_\lambda\) 为 \(\mathrm{W}\,\mathrm{sr}^{-1}\,\mathrm{m}^{-2}\,\mathrm{m}^{-1}\)。  
> 改核后清 `/var/tmp/OptixCache_*`，否则跑到旧 PTX。不要往 `AtmParams` 前半段插字段。不要给红外加 `ATM_WRAP_XY` 当地面补丁。

主循环：`atmospherePathTracer.cuh` 的 `traceAtmospherePath`。后向 Monte Carlo：传感器出射，三角网格、**1D 气体层**、**3D 云液滴体素**竞争自由程。

---

## 0. 三条链对照

| | 短波 SWIR | 中波 MWIR | 长波 LWIR |
|--|--|--|--|
| dump λ | **1600 nm** | **4000 nm** | **10500 nm** |
| 核模式 | `ATM_SPECTRAL_SW` | `ATM_SPECTRAL_LW` | `ATM_SPECTRAL_LW` |
| 不是 | 热成像、CIE、1.4–1.8 μm 相机 | 可见光中波、`-s sw=`、3.7–4.8 μm 窗 | 可见长波、`-s sw=`、8–12 μm 窗 |
| htrdr | `-s sw=1600,1601` | `-s lw=4000,4001` | `-s lw=10500,10501` |
| 太阳 | 开。\(L_\odot=B_\lambda(5778\,\mathrm{K})\) | **关**（对齐 htrdr LW） | **关** |
| 热发射 \(B_\lambda(T)\) | **不算** | 吸收处 + 表面 Kirchhoff | 同左 |
| 云游走 | \(k_s\)，再 analog \(\mathrm{Tr}(k_a)\) | \(k_{\mathrm{ext}}\)，Bernoulli \(\omega\) | 同左 |
| 气游走 | \(k_s\)；盒内 \(x_{\mathrm{H_2O}}\) 插值 | \(k_s=0\) ⇒ 走 1D \(k_a\) | 同左 |
| 气体表 | 含 1600 nm 的 RRTMG **SW** 档 | LW 3846–4202 nm | LW 10204–12195 nm |
| `mie_omega` | 1（`walk_ks=1` 时 Bernoulli 用不到） | 0（用体素 \(\omega\)） | 0 |
| miss | 0（太阳只走 NEE / 相机锥） | 0 | 0 |
| 默认 spp | 128 | 64 | 64 |
| 32 spp 无飞机 GPU/htrdr 均值 | **1.010** | **0.999** | **1.001** |

htrdr 输出 \(\int L_\lambda d\lambda\)（1 nm 窗），除以 \(10^{-9}\,\mathrm{m}\) 才是 \(L_\lambda\)。

**光谱约定（htrdr，不是相机，不是 LBL 闭合）：**

- \(B_\lambda(T)\)、水面 ρ、HG 的 \(g\)：dump λ（路径不抽 3.7–4.8 / 8–12）。
- 气体 \(k\)：含 dump λ 的 RRTMG 档，按权重抽 iquad。
- 云：`htmie_compute_xsection_{scattering,absorption}_average` 对该档 **分别** 做均匀 λ 线性平均，得到 \(\overline{C}_s,\overline{C}_a\)。不是 \(\overline{C}_{\mathrm{ext}}\) 再拆，也不是均匀波数平均。体素仍存 \(C(\lambda)\)，路径乘 \(\overline{C}/C(\lambda)\)。
- \(\omega=\overline{C}_s/(\overline{C}_s+\overline{C}_a)\)。这不等于 \(\langle\omega(\lambda)\rangle\)（10500 nm 档：0.481 vs 0.473）。
- \(g=g(\lambda_{\mathrm{dump}})\)。htrdr 算过 \(g_{\mathrm{avg}}\) 但短波 setup **不用**（`methods.md` §6.1）。截面档平均、相函数单色，是同一条约定。
- 因此发射是 \(B_\lambda(\lambda_{\mathrm{dump}},T)\)，消光是档平均 \(k\)：RTE 严格能量不闭合。光学厚、LTE 极限下像素趋近 \(B_{\mathrm{dump}}\)，偏差小；光学薄时发射 \(\propto k_a B\) 会偏。三条红外对齐 htrdr 就停在这条约定。

**不要把「走 \(k_{\mathrm{ext}}\)」读成「有一份三维大气 \(k_{\mathrm{ext}}\)」。** 云体素走 \(k_a+k_s\)。LW 气体被强制 \(k_s=0\)，故气体 \(k_{\mathrm{ext}}\equiv k_a\)。

---

## 1. 共用场景（FIRE 正视）

入口：`tools/render_fire_shahed_bands.py`。对照脚本：`tools/render_htrdr_fire_swir1600.sh` / `_mwir.sh` / `_lwir.sh`。`--no-uav` 去掉 An-124。

| | |
|--|--|
| 云 | HTCP `L25.1.FIRE.012`，511×511×70，\(z\sim0\)–700 m。正视柱按 **550 nm** 平均 τ 选（(265,394)，xy=(13275, 19725) m，τ≈11） |
| 云 \(k_{s,a}\) | 体素先按 dump λ 的 \(C(\lambda)\) 写出，路径再乘 \(\overline{C}(\Delta\nu)/C(\lambda)\)（htrdr 谱段平均截面）。\(g=g(\lambda)\) |
| 气体 | pack 默认 `ecrad_opt_prop.txt` 1D 列（地面 101300 Pa / 294.2 K，顶 ~88 km）。**不是 FIRE 探空** |
| 相机 | 正交 \((x,y,2000)\) m，胶片高 400 m，512²，look \((0,0,-1)\)，GSD≈0.78 m |
| 水面 | 600 m 两三角，Lambert ρ=0.04，T=290 K。红外绑常数 LUT，**不**绑 `blue_water`（2.08 μm 后 ρ=0） |
| 机体 | 仅 `--uav`。红外：ρ_body=0.10、ρ_eng=0.08、T_body=300 K、T_eng=550 K。短波仍用可见光 ρ |
| 网格周期 | `ntri≤65536` 关 XY 周期；后续 bounce **仍追网格** |
| 太阳几何 | az=340°、el=25°，有限锥 `cos_half=cos(4.6524e-3)` |
| `path_tmax` | \(1.2\times10^5\) m；`max_bounces=256` |
| 体积 IS | 三条都跳过（IS 读 host 均值，看不到路径 iquad） |

Planck（设备 `expf-1`，float）：

\[
B_\lambda(T)=\frac{2hc^2}{\lambda^5(e^{hc/\lambda kT}-1)}
\]

T≤0 → 0。**按位置**：点在 HTCP 有限 AABB 内（体素或气体碰撞都一样）用 HTCP \(T_{ijk}\)；盒外气体用 HTGOP 层 T。网格 T：每三角形；GS（本实验关）写死 288.15 K。水面 290 K 与 HTGOP 地面 294.2 K 仍是两套（海表 vs 名义廓线地面）。

---

## 2. 短波 1600 nm（`SPECTRAL_SW`）

**RTE（无热源）：** 走散射，吸收只进透过率。\(B_\lambda(300\,\mathrm{K})\) 在 1.6 μm 可忽略，核里吸收事件 **不**加 Planck。

### 2.1 主机

1. `load_htcp_voxels(..., 1600)`：体素按 1600.0 nm 的 \(C(\lambda)\) 写出。`mie_cs_ref=C(1600)`，再乘该 SW 档 \(\overline{C}/C(1600)\)（htrdr \(\overline{C}(\Delta\nu)q_l\)）。
2. `load_ecrad_sw_visible(1600,1600)`：只绑 **含 1600 nm 的那一档** RRTMG SW。每路径抽该档一个 iquad，`gas.ka/ks` 指向该列。
3. HTCP `RVT→x_{\mathrm{H_2O}}\)。云盒子内按 HTGOP 湿度结插该 iquad 的 \(k_a,k_s\)；盒子外仍是该 iquad 的 `ka_nominal`。缺 RVT 则 abort。
4. `sun_radiance=1` 是占位；`atm_sw_bind_wavelength` 写成 \(B_\lambda(5778\,\mathrm{K})\)。
5. `walk_ks=1`。水面/机体：ρ=0.04 LUT 与可见光机体，**无 SWIR 光谱**。

### 2.2 一条路径

1. 抽 iquad，绑气。太阳辐亮度 = Planck 5778 K @ 1600 nm。
2. 若 **相机方向**落在太阳锥内：\(L\mathrel{+}=L_\odot\,\mathrm{Tr}(k_{\mathrm{ext}})\)（直视太阳）。
3. 竞争风险：气层 \(k_s\) Woodcock（盒内 majorant \(\hat k_s=\max_t k_s^{\mathrm{tab}}\)）与云砖 \(k_s\) Woodcock，取更近。
4. 每段乘 analog \(\mathrm{Tr}(k_a)=\exp(-\int k_a ds)\)。盒内 \(k_a\) 按体素 XY 切开。
5. 散射：云 HG；气 Rayleigh。体积 NEE：\(\mathrm{phase}\times L_\odot\times\mathrm{Tr}(k_{\mathrm{ext}})\)（mesh 遮挡 + analog 消光）。
6. 表面：无 Kirchhoff 热项。Lambert NEE \( (\rho/\pi)(\mathbf{n}\cdot\boldsymbol{\omega}_\odot)L_\odot\mathrm{Tr}(k_{\mathrm{ext}})\)。`atm_add_sky_surface` **空操作**（rev 76，htrdr 无 0.08 天空垫）。
7. miss：停。不在 miss 上再加太阳盘。
8. RR：仅非体事件且 bounce≥8。体积永不 RR。

### 2.3 拼法上限

云 \(k\) 是该 SW 档 \(\overline{C}\)；\(B_\lambda\)、\(g\)、ρ 在 1600.0 nm。气是该 SW **整段** quadrature。不是 1.4–1.8 μm 相机。这是 htrdr `-s sw=1600,1601` 同一套，不是线-线 LBL。

对照：`scenes/fire_gs/swir1600/`。rev 79 32 spp 均值比 **1.010**。原始中位比 1.087 是噪声（信号 cv≈9.6%，噪声 cv≈130%，\(\mathrm{corr}\approx\mathrm{Var}S/(\mathrm{Var}S+\mathrm{Var}n)\approx0.005\)）。20 m 平滑后中位比 1.007，相关 0.891。

---

## 3. 中波 4000 nm（`SPECTRAL_LW`，太阳关）

**RTE（热，无太阳）：** 吸收处发射 \(B_\lambda(T)\)，散射积分相函数。htrdr `compute_radiance_lw`。不是可见光、不是 `-s sw=`。白天 4 μm 太阳与热出射可同量级——那是**另一份混源实验**，这条对照关太阳。

### 3.1 主机

1. 云体素 dump **4000 nm**，路径乘该 LW 档 \(\overline{C}/C(4000)\)（\(\omega\) 随 \(C_a,C_s\) 一起变）。
2. `load_ecrad_gas_1d_lw`：LW 谱段 **3846–4202 nm** 的 pdf 加权 \(k_a\) 作 1D 回退，\(k_s=0\)。禁止对红外调 `load_ecrad_gas_1d`（会落到 SW band 0）。
3. `sun_radiance=0` → `atm_lw_bind_path` 保持太阳 0。`r_lam` **不用**（不在 3.7–4.8 μm 均匀抽 λ）。
4. `mie_omega=0`。水面 ρ=0.04、T=290 K；机体红外灰体。

### 3.2 一条路径

1. 用 dump λ=4000 nm 找 LW 档，抽 iquad，`gas.ka` 指向该列。云 `path_ks_scale=\overline{C}_s/C_s(4000)`，`path_ka_scale` 同理。
2. **不**乘 analog \(\mathrm{Tr}(k_a)\)。碰撞密度已含所走的 \(k\)。
3. 云：砖 \(\hat k=\max(k_a+k_s)\)，接受 \(k_{\mathrm{ext}}\)，Bernoulli \(\omega=k_s/k_{\mathrm{ext}}\)。
4. 气：`walk_ks=0` 且主机 \(k_s=0\) ⇒ 走 \(k_a\)。接受后 \(k_a/k_{\mathrm{ext}}=1\) → 只吸收，无 Rayleigh。
5. 吸收：\(L\mathrel{+}=T\,B_\lambda(T)\)，停。盒内（云滴或气体）T=HTCP；盒外气体 T=HTGOP。
6. 散射：只云 HG。太阳 NEE 因 \(L_\odot=0\) 不贡献。
7. 网格：\(L\mathrel{+}=T(1-\rho)B_\lambda(T_{\mathrm{tri}})\)，Lambert 弹射。
8. miss → 0。无 CMB、无大气顶向下长波（与 htrdr LW 一致）。

### 3.3 对照

`scenes/fire_gs/mwir4000/`。rev 79 32 spp：均值 **0.999**。原始像素相关 0.006 = 信号 cv 0.4% 对噪声 5.4%。20 m 相关 **0.911**，平滑像素比 p5–p95 = 0.997–1.002。

---

## 4. 长波 10500 nm（`SPECTRAL_LW`，太阳关）

与 §3 **同一套核**，只换 dump λ。不是可见长波、不是 8–12 μm 窗平均。

1. 云体素 dump **10500 nm**，路径乘该档 \(\overline{C}/C(10500)\)。档宽，\(C(\lambda)\) 与 \(\overline{C}\) 差比中波大。
2. 气绑 LW **10204–12195 nm**。1D \(\tau_a\approx0.54\)。
3. `atm_lw_bind_path` 仍不抽窗口 λ。太阳 0。
4. 层积云接近灰体：像素几乎是 \(B_\lambda(T_{\mathrm{cloud}})\)，平滑后起伏 ~0.07%。

对照：`scenes/fire_gs/lwir10500/`。rev 79 \(\overline{C}\) 32 spp：均值 7.583×10⁶ / 7.575×10⁶ = **1.001**；分位比 p1–p99 全在 1.001 附近。像素相关 ≈0。GPU 噪声 σ/μ = 1.12%，与 htrdr 1.13% 齐。20 m 相关 0.52。旧 dump-\(C\) 均值比 1.000。

---

## 5. 核分叉（SW vs LW）

```
walk_ks = (spectral_mode != LW)
atm_segment_abs_weight = isLW ? 1 : analog Tr(ka)
```

| 事件 | SW | LW |
|--|--|--|
| 云接受 | \(k_s\) | \(k_{\mathrm{ext}}\) 再 \(\omega\) |
| 气接受 | \(k_s\)（盒内 3D \(x_{\mathrm{H_2O}}\)） | \(k_a\)（1D `ka_nominal`） |
| 吸收 | 吞吐 × \(\mathrm{Tr}(k_a)\) 后停，**不加** \(B\) | 吞吐=1，\(L\mathrel{+}=B_\lambda(T)\)，停 |
| 太阳 | Planck 5778；NEE + 相机锥 | host=0 则全关 |
| 表面热 | 无 | \((1-\rho)B_\lambda(T_{\mathrm{tri}})\) |
| 天空垫 0.08 | 关 | 关 |

LW 若再乘 \(\mathrm{Tr}(k_a)\)，权重变成 \(\mathrm{e}^{-\tau_s-2\tau_a}\)。rev 72 起 `isLW` 上 analog=1。

体积 NEE 用 `atm_sun_visibility_volume`（mesh 遮挡 + analog \(\mathrm{Tr}(k_{\mathrm{ext}})\)）。表面 NEE 用 `atm_sun_visibility`（GS+mesh；FIRE 无 GS）。山体 CIE 的 0.08 天空垫 **不**走这三条红外。

---

## 6. 物理评审

分两层：**估计量是否在所采用的介质上无偏实现 RTE**，以及 **介质/光谱/边界是否等于真实大气或相机**。前者对了不等于后者对。

### 6.1 符合（在给定介质上）

1. **Planck \(B_\lambda\) SI**，float `expf-1`。1.6 μm、300 K 热项 ~0，不算热是对的。
2. **不透明灰体 Kirchhoff** \(\varepsilon=1-\rho\)，发射 + Lambert 反射，能量在 \(\rho\in[0,1]\) 上自洽。
3. **LW 云走 \(k_{\mathrm{ext}}\)**，吸收停、散射继续，不把 \(k_a\) 算两遍。与 htrdr LW 形式一致。
4. **SW 分解追踪**（走 \(k_s\) + analog \(\mathrm{Tr}(k_a)\)）是标准短波估计量；与 LW 热核分开是对的。
5. **LW 气体无 Rayleigh**（\(k_s=0\)）在热红外分子散射可忽略的意义上成立。
6. **miss=0** 与 htrdr LW 一致（无宇宙微波、无 TOA 向下长波）。
7. **局地热动平衡**：吸收处用当地 \(T\) 的 \(B_\lambda\)，无荧光/非 LTE。
8. **跳过体积 IS**：IS 读不到路径 iquad，跳过避免用错 \(k\)。
9. **竞争风险**（气一层 + 云砖各抽一次、取近）对组合泊松过程期望无偏，只要两边的 \(k\) 是同一物理场。

### 6.2 不符合、不完备、或与真实/htrdr FIRE 不一致

按严重程度。这些不是「再清一次 OptiX 缓存」能消失的。

**介质不是场景大气**

1. **没有整柱三维气体。** LW/MWIR 只用 HTGOP `ka_nominal` 1D 水平均匀列。htrdr 在云内用 LES \(x_{\mathrm{H_2O}}\) 插该层气体 \(k_a\)。GPU 把气体光学厚度的三维结构抹掉了。SWIR 只在 HTCP **盒子内**插湿度；盒子外仍是名义廓线。
2. **海表 / 机体 / 名义地面仍三套 T：** 水面 290 K vs HTGOP 地面 294.2 K vs 机体 300/550 K。云盒子内 rev 78 已按位置统一为 HTCP（气体碰撞不再读 HTGOP）。海表这件事和云内 Planck 不是同一条。
3. **无气溶胶三维场、无冰相、无降水、无羽流。** 发动机 550 K 只是短舱三角。
4. **体素 fetch XY 取模**（`atm_wrap_xy_cell`），Woodcock 只裁 Z 板；htrdr 对照脚本 **未**开 `-r`。正视 400 m 胶片在 25 km 域内，太阳 25° 穿云水平尺度约 1 km，本实验通常碰不到周期边。斜向出域则与「有限云」不一致。不要为此改成无限 XY 当 nadir 地面补丁。

**光谱不是相机、也不是单色谱线**

5. **单色 dump λ 上的 \(B_\lambda,\rho,g\) + 谱段 \(k\)：** 气体 iquad、云 \(\overline{C}(\Delta\nu)\)，与 htrdr `-s lw=4000,4001` / `sw=1600,1601` 相同。不是滤光片 \(\int\eta(\lambda)L_\lambda d\lambda\)，也不是 8–12 / 3.7–4.8 μm 窗口图。不是 LBL \(k(\lambda_{\mathrm{dump}})\)。
6. **相函数是 HG**，不是 LUT 的 Mie \(P(\theta)\)。\(g\) 来自同一 Mie LUT，但高阶矩丢了。
7. **太阳是 \(B_\lambda(5778\,\mathrm{K})\) 圆盘**，不是 Kurucz / 实测 TOA、没有 Fraunhofer。SWIR 均值比 1.064 含这一项与气 \(k\) 谱段化的残差。
8. **MWIR 关太阳。** 对「对齐 htrdr LW」正确；对「白天 4 μm 相机」物理不完备（太阳与热同量级）。

**表面不是真实光学**

9. **水面 ρ=0.04 灰 Lambert**，不是光谱海水、不是菲涅尔（本实验 `mat_id` 不走 Schlick 水）。短波泡沫/白冠没有。
10. **机体红外 ρ/T 是经验数**，无材料光谱。短波机体仍用可见光 ρ，SWIR 机体错。
11. **Kirchhoff 要求不透明；** 薄漆/半透明短舱不成立。ε 与载荷脚本里的 0.5 不是同一套。

**估计量与 htrdr 求解器仍有差（期望在 \(k\) 一致时应接近，方差/纹理不必）**

12. **一个组合泊松 vs 一棵八叉树。** 竞争风险 majorant 更松，GPU LWIR 32 spp 噪声 σ 比 htrdr 高约 27%。分位可对齐，像素图对不齐。
13. **SW analog \(\mathrm{Tr}(k_a)\)** 不是 htrdr 的 residual / δ-tracking。短波 **NEE** 的透过率还是 analog \(\mathrm{Tr}(k_{\mathrm{ext}})\)，与路径分解不是同一套跟踪。
14. **俄罗斯轮盘** 只打 bounce≥8 的表面，体积从不 RR。与 htrdr SW「只对 BSDF 做 RR」同类，但阈值不同。

**输出不是亮度温度**

15. npy 是该估计量下的 \(L_\lambda\)（三条都钉 dump λ，**不再**窗口抽 λ）。PNG 分位拉伸或固定 ±% 灰度，**不是** \(T_B\)。旧 LWIR 1–99.5%+√ 把 1.4% 的 MC 噪声拉成全对比度。

**不应当成 bug 的近似**

- LW 无分子散射：热红外成立。
- SW 不算热：1.6 μm 成立。
- miss=0：与 htrdr LW 的实验定义一致，不是 CMB 模型。
- float Planck：本三条 λ 与 280–5800 K 下 `expf` 够用。

---

## 7. 32 spp 对照摘要（rev 79，云 \(\overline{C}(\Delta\nu)\)）

原始像素相关 ≈0 **不是核偏**：独立 MC，\(\mathrm{corr}=\mathrm{Var}S/(\mathrm{Var}S+\mathrm{Var}n)\)。短波信号 cv 9.6% / 噪声 130% → 期望 corr≈0.005（测到 0.004）。中波 0.42% / 5.4% → 期望 0.006（测到 0.006）。长波信号更弱。要对场，用 20 m 平滑。

| | 均值比 | 20 m 中位比 | 原始相关 | 20 m 相关 | 噪声 σ/μ GPU / htrdr |
|--|--|--|--|--|--|
| SWIR 1600 | **1.010** | **1.007** | 0.004 | **0.891** | 130% / 138% |
| MWIR 4000 | **0.999** | **0.999** | 0.006 | **0.911** | 5.44% / 5.47% |
| LWIR 10500 | **1.001** | **1.001** | ≈0 | 0.52 | 1.12% / 1.13% |

短波剩 ~1% 均值是均匀的（四象限 1.005–1.014），不是截面没乘 \(\overline{C}\)，也不是 HG/5778（两边一样）。不改核去「对齐像素图」。

图：`scenes/fire_gs/swir1600/compare_swir1600.png`、`mwir4000/compare_mwir4000.png`、`lwir10500/compare_lwir10500.png`（上排原始噪声，下排 20 m）。

---

## 8. 载荷脚本

`shahed_ir_payload.py` 读 `record.json`，不重渲染。spp ≠ 积分时间。对比度模型用 300 m 晕与 ε=0.5，与网格 Kirchhoff 不是同一个 ε。

---

## 9. 不要做的事

- 不要给 SWIR 走 \(k_{\mathrm{ext}}\) 或 `mie_omega=0`（会与 analog \(\mathrm{Tr}(k_a)\) 双重吸收）。
- 不要给红外 λ 调 `load_ecrad_gas_1d`。
- 不要绑 `blue_water` 当 1.6–12 μm 海水。
- 不要把湿度表绑到 CIE。
- 不要把 MWIR 改回 `-s sw=` 或 CIE。
- 不要把 LWIR 改回 8–12 μm 均匀抽 λ。
- 不要为「看起来有大气」编 3D \(q_v\) 或第三块沙面。
- 不要用无限 XY / `ATM_WRAP_XY` 修 nadir 地面。
- 改核必须清 OptiX 缓存并升 `ATM_KERNEL_REV`。

---

## 10. 文件锚点

| | |
|--|--|
| `ATM_KERNEL_REV` | `gpu/threedgptir_tracer/src/kernels/cuda/referenceOptix.cu` |
| 路径、analog、IS、太阳 NEE、天空垫 | `atmospherePathTracer.cuh` |
| Planck、Woodcock、`atm_lw_bind_path` / `atm_sw_bind_wavelength` | `atmosphere.cuh` |
| `load_ecrad_gas_1d_lw` / `require_sw_band` / `require_sw_h2o` | `tools/mountain_gs_scene.py` |
| FIRE 三档 | `tools/render_fire_shahed_bands.py` |
| htrdr 对照 | `tools/render_htrdr_fire_swir1600.sh`、`_mwir.sh`、`_lwir.sh` |
