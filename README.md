# cloudGS

GPU 短波大气路径追踪：对齐 Villefranque et al. 2019（htrdr）的无偏 Monte Carlo，在 OptiX 上把同一套 `mountain.sh` 场景跑到比 48 核 CPU 更快。

论文原文是 CPU 库、无 GPU（JAMES / arXiv:1902.01137）。本仓库求解器仍是：Woodcock 体素云 + 1D 气体、太阳盘 NEE、CIE 1931 图像、HG 相函数、周期山地 mesh。改的是**实现**，不是换一套辐射物理。

## 相对论文 / htrdr 的改进

论文与其后的 htrdr CPU 把每个像素的 CIE X/Y/Z **拆成三条独立路径**（Table C1 还是每通道 4096 spp）。路径在体积里和 830 万三角的山上分叉，CPU 线程各走各的，warp 这一层不存在。

本仓库在 **RTX 4090** 上、同一相机（ARMCu / `mountain.sh`，1280×720，太阳方位 225°、高度角 30°）做了这些实现层加速，辐射量仍对冻结的 htrdr spp256 参考（全图 Y 比 ≈ 0.995）：

| 相对原文的点 | 论文 / htrdr CPU | 本仓库 GPU |
|---|---|---|
| 硬件 | C 库，发表时无 GPU；对照用 48 线程 | OptiX 8 + RTX 4090 |
| CIE | 每通道独立抽 λ，spp 要乘 3 | **一条路径同时累加 X/Y/Z**（CMF 混合 MIS） |
| 地面 | Embree 三角形 | 瘦 payload 的 mesh GAS（7 字，不再走 PTIR 32-payload anyhit） |
| 线程 | CPU 核 | **SER**：打山 / 透云 / 不打山 2-bit 重排，减少 warp 发散 |
| 地面图元 | `mountain.obj` | 同 mesh；GS 默认关（`ATM_ENABLE_GS=1` 才开） |

同路径 CIE 去掉品红/绿斑（那是 X 与 Z 独立噪声），并少做 2/3 的光谱路径。默认不要开 `--adaptive`：按方差加样本目前会把全图 Y 拉低约 3%。

## 测速（`mountain.sh` 1280×720，max_bounces=256）

对照：冻结 htrdr CPU **48 线程、spp=256、18 min 32 s（1112 s）**。GPU 与它对同一张 `htrdr_ref_mountain_1280x720x256`。

| 实现 | spp | 墙钟 | 相对 htrdr CPU | 全图 Y GPU/htrdr |
|---|---:|---:|---:|---:|
| htrdr CPU（论文软件，48 核） | 256 × 3 通道 | 1112 s | 1× | 1 |
| 本仓库 GPU，CIE 拆通道 + SER | 256 | 670 s | **1.7×** | 0.995 |
| 本仓库 GPU，**同路径 CIE + SER** | 256 | **249 s** | **4.5×** | 0.996 |

中间阶梯（同一 GPU，云窗 `--crop 350 40 950 240` spp=4）说明 SER 的贡献：32-payload 时代约 24 s → 瘦 payload 约 24 s 量级上再削一截 → SER 后 **2.5 s（约 10×）**。全图在 SER 之后才稳定快过 48 核 htrdr。

吞吐：同路径后约 **104 万 camera-ray/s**（每条路径写满 XYZ）。旧拆通道约 37 万 camera-ray/s，但每条还要 ×3 光谱 launch。

成像随 spp：均值从 spp=32 就对齐（Y≈0.995）；脏的是方差。spp32→256 火蝇 p995 从 ~47 收到与 htrdr 相当的 ~32。对比图见 `scenes/mountain_gs/compare_spp256_cie_samepath.png`。

论文 Table C1 用每通道 **4096** spp，本对照用 Starter Pack 的 `mountain.sh` **256** spp，几何相同，样本数不同，不能直接和论文插图比噪点。

## 布局

- `gpu/threedgptir_tracer/` — OptiX 大气路径（`ATM_KERNEL_REV` 升核须清 `/var/tmp/OptixCache_*`）
- `tools/render_mountain_gs_gpu.py` — `mountain.sh` / 正交 / 云上相机
- `scenes/mountain_gs/` — mesh、htrdr 冻结参考、GPU 对照图
- `docs/` — 论文笔记（`docs/README.md`）与 htrdr 源码对照（`docs/methods.md`）

## 渲染

```bash
# 验收相机，默认 mesh 地面、同路径 CIE
python tools/render_mountain_gs_gpu.py --camera mountain --ground mesh --spp 256 --max-bounces 256

# 云窗 Y 门（相对 htrdr 应 ≈ 0.99）
python tools/render_mountain_gs_gpu.py --camera mountain --ground mesh --spp 4 --crop 350 40 950 240

# 旧 CIE 三通道 launch
ATM_CIE_SPLIT=1 python tools/render_mountain_gs_gpu.py --camera mountain --spp 32
```

关 SER：`ATM_SER=0`。开 GS：`ATM_ENABLE_GS=1`。
