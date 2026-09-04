<div align="center">

# lsfg-vk Custom

### Fixed Target + D3B3 Dual-GPU development fork for Linux / Vulkan

[![Fork](https://img.shields.io/badge/Fork-2.0.0--fixed--target-0ea5e9?style=for-the-badge)](https://github.com/wizardkof/lsfg-vk-custom)
[![Upstream](https://img.shields.io/badge/Upstream-2.0.0--dev28-6366f1?style=for-the-badge)](https://github.com/PancakeTAS/lsfg-vk)
[![Platform](https://img.shields.io/badge/Platform-Linux-f59e0b?style=for-the-badge\&logo=linux\&logoColor=white)](https://github.com/wizardkof/lsfg-vk-custom)
[![API](https://img.shields.io/badge/API-Vulkan-ac162c?style=for-the-badge\&logo=vulkan\&logoColor=white)](https://www.vulkan.org/)
[![Default Branch](https://img.shields.io/badge/Default%20Branch-fixed--target-22c55e?style=for-the-badge)](https://github.com/wizardkof/lsfg-vk-custom/tree/fixed-target)
[![License](https://img.shields.io/github/license/wizardkof/lsfg-vk-custom?style=for-the-badge)](LICENSE.md)

**A development fork of [PancakeTAS/lsfg-vk](https://github.com/PancakeTAS/lsfg-vk) focused on explicit Fixed Target frame generation and a production-safe cross-device / Dual-GPU Vulkan frame-generation architecture.**

Same-GPU operation remains the safe default while the D3B3 Dual-GPU pipeline is developed, source-qualified and hardware-qualified in explicit stages.

<br>

[Status](#-project-status) ·
[Features](#-feature-matrix) ·
[D3B3 Progress](#-d3b3-development-progress) ·
[Architecture](#-target-dual-gpu-architecture) ·
[Validation](#-validation-model) ·
[Installation](#-installation) ·
[Roadmap](#-roadmap) ·
[Limitations](#-current-limitations)

</div>

---

# 🚀 Project status

|                                           |                                                             |
| :---------------------------------------- | :---------------------------------------------------------- |
| **Fork version**                          | `2.0.0-fixed-target`                                        |
| **Upstream baseline**                     | `2.0.0-dev28`                                               |
| **GitHub default branch**                 | `fixed-target`                                              |
| **Current development line**              | `D3B3 / PRE-A6B`                                            |
| **Last published qualification branch**   | `d3b3-prea6b-pfsr-qualified-20260831`                       |
| **Last published qualification commit**   | `633901221784220ffe4beaf2a3de0e1503c35b6c`                  |
| **Platform**                              | Linux                                                       |
| **Graphics API**                          | Vulkan                                                      |
| **Language standard**                     | C++20                                                       |
| **Safe production default**               | Same-GPU                                                    |
| **Fixed Target**                          | ✅ Implemented                                               |
| **Adaptive 1x bypass**                    | ✅ Implemented                                               |
| **Cross-device diagnostic probes**        | ✅ Implemented / hardware tested                             |
| **D3B3 production foundations**           | ✅ Implemented                                               |
| **Strict all-virtual X2–X5 scheduler**    | ✅ PRE-ROOT source-qualified                                 |
| **Root production X2–X5 routing**         | 🔒 Disabled                                                 |
| **Strict mixed X2–X5 routing**            | 🔒 Deferred                                                 |
| **Continuous production Dual-GPU FG**     | 🔒 Not enabled                                              |
| **Current engineering stage**             | `PRE-A6B`                                                   |
| **Current engineering gate**              | 🟡 Global WSI teardown / exact per-swapchain destroy safety |
| **Final Dual-GPU hardware qualification** | 🔒 Not run                                                  |

> [!IMPORTANT]
> **The GitHub default branch remains `fixed-target`, but the active engineering line has progressed into D3B3 / PRE-A6B.**
>
> The last published qualification checkpoint is:
>
> ```text
> branch: d3b3-prea6b-pfsr-qualified-20260831
> commit: 633901221784220ffe4beaf2a3de0e1503c35b6c
> ```
>
> Development after that checkpoint includes later scheduler and lifetime qualification work that has not yet been published as a new qualification branch/tag.

> [!CAUTION]
> **Dual-GPU frame generation is not a normal user-facing production feature yet.**
>
> The cross-device transport foundations and strict PRE-ROOT X2–X5 scheduler exist, but Root routing, final assembly, hardware qualification and production enablement are intentionally still disabled.

---

# ✨ Feature matrix

| Feature                                                 | State | Current status                                    |
| :------------------------------------------------------ | :---: | :------------------------------------------------ |
| **Same-GPU lsfg-vk operation**                          |   ✅   | Preserved as safe default                         |
| **Adaptive frame generation**                           |   ✅   | Existing upstream-style behavior                  |
| **Adaptive 1x bypass**                                  |   ✅   | Layer/profile remains active, generation bypassed |
| **Fixed Target mode**                                   |   ✅   | Explicit output cadence through `target_fps`      |
| **Physical-device diagnostics**                         |   ✅   | Stable device discovery / identity                |
| **DMA-BUF cross-device probe**                          |   ✅   | Isolated diagnostic path                          |
| **SYNC_FD cross-device probe**                          |   ✅   | Isolated binary-semaphore bridge                  |
| **D3B3 typed ownership model**                          |   ✅   | Implemented                                       |
| **A → B transport foundations**                         |   ✅   | Implemented                                       |
| **B → A return foundations**                            |   ✅   | Implemented                                       |
| **Destination Return Timeline Authority**               |   ✅   | Source-qualified                                  |
| **Prepared Frame Schedule Reservation**                 |   ✅   | Source-qualified                                  |
| **Generated Frame Preparation Reservation**             |   ✅   | Source-qualified                                  |
| **Strict all-virtual X2–X5 scheduler**                  |   ✅   | PRE-ROOT source-qualified                         |
| **Generation-safe virtual runtime binding**             |   ✅   | Source-qualified                                  |
| **Strong runtime lifetime authority**                   |   ✅   | Source-qualified                                  |
| **Physical WSI lifecycle generation**                   |   ✅   | Source-qualified                                  |
| **Per-swapchain WSI teardown without device-wide idle** |   🟡  | Current PRE-A6B gate                              |
| **PRE-A6B full contract**                               |   🔒  | Future gate                                       |
| **PRE-A6B Production Assembly**                         |   🔒  | Future gate                                       |
| **A6B qualification**                                   |   🔒  | Future gate                                       |
| **Root X2–X5 integration**                              |   🔒  | Not enabled                                       |
| **Normal production D3B3 routing**                      |   🔒  | Not enabled                                       |
| **Strict mixed Native/D2/Virtual X2–X5**                |   🔒  | Deferred                                          |
| **Final Dual-GPU UI controls**                          |   🔒  | Not implemented                                   |
| **Continuous Dual-GPU 2x–5x hardware qualification**    |   🔒  | Not run                                           |

Legend:

* ✅ **Implemented / qualified for its current development gate**
* 🟡 **In development or corrective qualification**
* 🧱 **Foundation exists but is not production-exposed**
* 🔒 **Future gate / intentionally not enabled**
* 🚫 **Not claimed / unsupported**

---

# 🧭 What this fork is

This project originally started as the **Fixed Target** development fork of lsfg-vk.

Its scope has since expanded.

The current development line is focused on two major areas:

1. **Fixed Target frame-generation cadence**
2. **Production-safe D3B3 cross-device / Dual-GPU frame generation**

The goal is not simply to make two Vulkan devices exchange a buffer.

The goal is to eventually support a continuous production pipeline with explicit synchronization, lifetime, WSI ownership, recreate handling and failure semantics.

---

# ⚡ Fixed Target

Fixed mode targets an explicit output cadence.

Example:

```toml
frame_generation_mode = "fixed"
target_fps = 60
```

Unlike Adaptive mode, Fixed mode does not use `multiplier` as its primary cadence control.

For example:

```toml
frame_generation_mode = "fixed"
target_fps = 60
multiplier = 1
```

does **not** mean Fixed mode is bypassed.

`target_fps` describes the desired Fixed output cadence.

It does not directly limit or cap the application's source FPS.

---

# 🔁 Adaptive mode

Adaptive mode uses `multiplier`.

Example:

```toml
frame_generation_mode = "adaptive"
multiplier = 2
```

The development model supports:

```text
1x = generation bypass
2x = one generated frame
3x = two generated frames
4x = three generated frames
5x = four generated frames
```

## Adaptive 1x bypass

```toml
frame_generation_mode = "adaptive"
multiplier = 1
```

Adaptive 1x keeps:

* the Vulkan layer active;
* the selected profile active;
* the backend infrastructure available;

but bypasses generated-frame production.

It is **not** equivalent to completely unloading lsfg-vk.

---

# 🧩 D3B3 development progress

The D3B3 work is the production-oriented cross-device frame-generation line.

It is developed in explicit gates rather than enabling experimental code directly in the normal application path.

---

## D3B3 runtime foundations

The development line contains production-oriented foundations for:

* ✅ typed frame transport;
* ✅ source ownership;
* ✅ source-ready synchronization;
* ✅ application present-wait ownership;
* ✅ returned-for-graphics synchronization;
* ✅ A → B transport;
* ✅ B → A return;
* ✅ asynchronous runtime cores;
* ✅ production runtime ownership;
* ✅ non-blocking retirement foundations;
* ✅ explicit source-layout handling;
* ✅ explicit first-use handling;
* ✅ hidden WSI acquisition ownership;
* ✅ device-retirement integration.

The project deliberately avoids using generic integers or loosely associated Vulkan handles as the production ownership model.

---

# 🗓 Prepared Frame Schedule Reservation

## PFSR

✅ **Source-qualified**

Prepared Frame Schedule Reservation establishes resources and ownership before frame-generation execution begins.

The authority covers concepts such as:

* source frame lifetime;
* generated-frame schedule;
* producer synchronization;
* destination-return synchronization;
* completion ownership;
* pre-reserved resources needed after bridge acceptance.

The goal is to avoid discovering indispensable resources only after GPU work has already been accepted.

---

# 🎬 Generated Frame Preparation Reservation

## GPR

✅ **Source-qualified**

GPR extends PFSR into generated-frame and physical-presentation preparation.

Qualified development behavior includes:

* one real PFSR per logical source frame;
* one exact bridge-consumer authority;
* bridge semaphore single consumption;
* generated-frame preparation for `2x` through `5x`;
* source-copy lifetime tracking;
* destination-return ownership;
* physical Acquire authority;
* Acquire failure recovery;
* physical image lease tracking;
* internal present-fence ownership;
* producer-completion authority;
* reacquire retirement;
* physical WSI lifecycle generation checks;
* old/new raw-handle isolation.

GPR itself is not intended to perform independent generated presents per logical entry.

---

# 🎞 Strict all-virtual X2–X5 scheduler

✅ **PRE-ROOT source-qualified**

The strict all-virtual scheduler is a whole-batch presentation primitive.

For multiplier `M`:

```text
generated phases = M - 1
```

The model is:

```text
Generated phase 0
    └── ONE whole-batch QueuePresentKHR

Generated phase 1
    └── ONE whole-batch QueuePresentKHR

...

Final original phase
    └── ONE whole-batch QueuePresentKHR
```

Therefore:

```text
2x
├── 1 generated whole-batch present
└── 1 final whole-batch present

3x
├── 2 generated whole-batch presents
└── 1 final whole-batch present

4x
├── 3 generated whole-batch presents
└── 1 final whole-batch present

5x
├── 4 generated whole-batch presents
└── 1 final whole-batch present
```

The scheduler does **not** issue one generated `QueuePresentKHR` per logical entry.

---

## Scheduler authorities already qualified

The PRE-ROOT scheduler development has closed the following areas:

* ✅ exact internal vs application-visible result separation;
* ✅ generated hidden-result handling;
* ✅ final application-visible result mapping;
* ✅ monotonic `VK_ERROR_DEVICE_LOST` escalation;
* ✅ post-commit allocation-error handling;
* ✅ cleanup / recovery failure propagation;
* ✅ move-only sealed scheduler entries;
* ✅ exact GPR / final-original association;
* ✅ application slot / image-index sealing;
* ✅ physical WSI lifecycle authority;
* ✅ canonical lifecycle host mutex;
* ✅ pre-present lifecycle revalidation;
* ✅ duplicate physical swapchain rejection;
* ✅ raw-handle reuse generation isolation;
* ✅ strong `VirtualSwapchainRuntime` lifetime;
* ✅ terminal physical lifecycle invalidation;
* ✅ non-resurrection of invalidated lifecycle state;
* ✅ generation-safe virtual-runtime binding authority;
* ✅ immutable logical `VkDevice` provenance;
* ✅ strict virtual-topology eligibility.

> [!IMPORTANT]
> The scheduler is currently a **PRE-ROOT production primitive**.
>
> It is intentionally not called by normal public Root X2–X5 routing yet.

---

# 🛡 Safety model

Cross-device frame generation introduces several distinct lifetime and synchronization domains.

This fork treats them explicitly.

---

## Physical WSI lifecycle authority

Physical WSI state tracks:

* exact `VkSwapchainKHR`;
* lifecycle generation;
* active / invalidated state;
* canonical host synchronization mutex;
* old/new raw-handle generation isolation.

An invalidated old lifecycle is not allowed to become active again.

This prevents stale asynchronous work from silently attaching itself to a reused raw swapchain value.

---

## Virtual runtime binding authority

Raw non-dispatchable Vulkan handle equality is not used as the sole runtime identity.

The development line uses a layer-owned immutable binding authority:

```text
VirtualSwapchainRuntime
        │
        ├── generation-safe binding
        │
        ▼
Virtual Swapchain context
        │
        ▼
exact logical VkDevice
```

A new runtime generation receives a new authority even if Vulkan or a test environment reuses numerical handle values.

---

## Strong runtime lifetime

Scheduler entries retain the exact virtual runtime needed by their logical work.

This means asynchronous work does not depend on the original caller keeping an unrelated reference alive.

---

## Application vs internal synchronization

The architecture distinguishes:

* application present waits;
* bridge synchronization;
* internal generated-frame semaphores;
* physical Acquire synchronization;
* layer-owned present fences;
* borrowed application present fences;
* producer-completion synchronization;
* destination-return synchronization.

These are not treated as interchangeable resources.

---

# 🖥 Target Dual-GPU architecture

The intended production architecture is:

```text
                 Application
                     │
                     │ original frame
                     ▼
          ┌─────────────────────┐
          │ GPU A               │
          │ Application Graphics│
          └──────────┬──────────┘
                     │
                     │ A → B transport
                     ▼
          ┌─────────────────────┐
          │ GPU B               │
          │ LSFG Generation     │
          └──────────┬──────────┘
                     │
                     │ generated frame return
                     ▼
          ┌─────────────────────┐
          │ GPU A               │
          │ Physical WSI        │
          └──────────┬──────────┘
                     │
                     ▼
                  Display
```

`GPU A` and `GPU B` describe roles.

They are not intended to mean one hard-coded vendor combination.

The final architecture should only activate a device pair when the required Vulkan capabilities are available.

---

# 🔬 Cross-device development probes

The CLI currently exposes diagnostic commands such as:

```bash
lsfg-vk-cli devices
lsfg-vk-cli interop
lsfg-vk-cli interop-buffer-probe --allocator PATH --device-a INDEX --device-b INDEX
lsfg-vk-cli interop-sync-fd-probe --allocator PATH --device-a INDEX --device-b INDEX
```

These diagnostic paths cover:

* Vulkan physical-device discovery;
* external-buffer capability inspection;
* cross-device DMA-BUF `VkBuffer` import/use;
* binary semaphore `SYNC_FD` synchronization between Vulkan devices.

---

## Development hardware validation

The isolated DMA-BUF and `SYNC_FD` probe paths have been validated on the development hardware combination:

```text
AMD / RADV
    +
NVIDIA RTX 3060
```

This validates those specific diagnostic primitives on that tested device/driver combination.

> [!CAUTION]
> It does **not** prove:
>
> * PCIe P2P;
> * physical zero-copy;
> * universal cross-vendor compatibility;
> * universal DMA-BUF modifier support;
> * shared `VkImage` compatibility on every driver;
> * production frame pacing;
> * final latency characteristics;
> * final continuous Dual-GPU frame generation.

---

# 🧪 Validation model

This project deliberately separates several meanings of “working”.

## Implemented

The source architecture or feature exists.

## Source-qualified

The implementation has completed its current source-level correctness gate and associated regression matrix.

## Hardware-qualified

The exact production path has been exercised on real hardware.

These states are **not equivalent**.

---

## Current validation matrix

| Area                                          | State |
| :-------------------------------------------- | :---: |
| Fixed Target                                  |   ✅   |
| Adaptive 1x bypass                            |   ✅   |
| Cross-device device discovery                 |   ✅   |
| DMA-BUF buffer probe                          |   ✅   |
| `SYNC_FD` synchronization probe               |   ✅   |
| D3B3 typed runtime foundations                |   ✅   |
| Destination Return authority                  |   ✅   |
| PFSR                                          |   ✅   |
| GPR                                           |   ✅   |
| Strict PRE-ROOT all-virtual X2–X5 scheduler   |   ✅   |
| Scheduler result / terminal authority         |   ✅   |
| Physical WSI lifecycle generation             |   ✅   |
| Strong runtime lifetime                       |   ✅   |
| Generation-safe virtual binding               |   ✅   |
| Logical-device provenance                     |   ✅   |
| Strict virtual-topology eligibility           |   ✅   |
| Global virtual WSI teardown                   |   🟡  |
| PRE-A6B full contract                         |   🔒  |
| Production Assembly                           |   🔒  |
| A6B                                           |   🔒  |
| Root X2–X5 production routing                 |   🔒  |
| Normal D3B3 route                             |   🔒  |
| Final continuous Dual-GPU hardware validation |   🔒  |

---

# 🚧 Current engineering gate

## PRE-A6B Global WSI Teardown

🟡 **In progress**

The current work is focused on safe virtual-swapchain destruction without using a device-wide idle as the lifetime authority for one swapchain.

The target model is:

```text
Virtual Swapchain A
        │
        │ destroy requested
        ▼
Exact A-only destroy-safety authority
        │
        ├── wait only for A's device use
        │
        ├── do not idle unrelated Swapchain B
        │
        └── preserve exact WSI lifetime
        ▼
safe synchronous downstream destroy
```

The final implementation must satisfy both goals:

```text
No broad vkDeviceWaitIdle for one virtual swapchain
```

and:

```text
Never destroy a swapchain while device execution still uses its acquired images
```

The first no-idle attempt removed the broad wait but did not yet establish a complete exact per-swapchain synchronous destroy-safety barrier.

That corrective work is the current PRE-A6B gate.

---

# 🔒 What is intentionally still disabled

The following paths are **not** normal production features yet:

* Root X2–X5 scheduler routing;
* normal D3B3 frame-generation routing;
* strict mixed Native/D2/Virtual X2–X5;
* automatic production Dual-GPU device selection;
* Dual-GPU configuration through the normal UI;
* final continuous 2x–5x hardware route.

This is intentional.

The project does not enable a path merely because its lower-level primitive exists.

---

# 🗺 Roadmap

## Completed foundations

* [x] Fixed Target mode
* [x] Adaptive 1x bypass
* [x] Stable physical-device diagnostics
* [x] External-memory capability discovery
* [x] DMA-BUF buffer probe
* [x] `SYNC_FD` semaphore probe
* [x] D3B3 finite cross-device orchestration foundations
* [x] Production D3B3 runtime adapter
* [x] Async runtime cores
* [x] Non-blocking retirement foundations
* [x] Typed ownership authorities
* [x] Source-layout / first-use authority
* [x] Destination Return Timeline authority
* [x] Prepared Frame Schedule Reservation
* [x] Generated Frame Preparation Reservation
* [x] Strict all-virtual X2–X5 whole-batch scheduler
* [x] Scheduler result / terminal qualification
* [x] Entry composition qualification
* [x] Strong runtime lifetime
* [x] Physical lifecycle non-resurrection
* [x] Generation-safe runtime binding
* [x] Logical-device provenance
* [x] Strict virtual-topology eligibility

---

## Current PRE-A6B work

* [ ] Exact per-swapchain synchronous destroy-safety barrier
* [ ] Virtual swapchain destroy without device-wide idle
* [ ] Rapid destroy / recreate qualification
* [ ] Multi-swapchain same-device teardown isolation
* [ ] Application allocator teardown lifetime
* [ ] Remaining global WSI lifetime matrix
* [ ] Remaining PRE-A6B error handling

---

## Remaining PRE-A6B gates

* [ ] Generic application present-fence closure
* [ ] Aborted logical-present authority
* [ ] Remaining legacy / fixed / D2 host-wait review
* [ ] Final OOM / DeviceLost / OutOfDate / SurfaceLost matrix
* [ ] Observer-transparency qualification
* [ ] Global destroy / recreate matrix
* [ ] **PRE-A6B Async Adapter Contract — Full Pass**

---

## Production construction

* [ ] **PRE-A6B Production Assembly**
* [ ] **A6B qualification**

---

## Root integration

Only after the previous gates are complete:

* [ ] Map application `VkSwapchainKHR` to exact qualified runtime / owner
* [ ] Map `VkQueue` to canonical queue metadata / mutex / device
* [ ] Connect the strict all-virtual X2–X5 scheduler
* [ ] Add requested vs effective multiplier routing
* [ ] Add safe fallback decisions
* [ ] Preserve 1x / same-GPU behavior
* [ ] Keep unsupported mixed batches on safe fallback
* [ ] Enable normal D3B3 route only after qualification

---

## Hardware qualification

Final hardware work must include:

* [ ] Same-GPU 1x regression
* [ ] Dual-GPU 2x
* [ ] Dual-GPU 3x
* [ ] Dual-GPU 4x
* [ ] Dual-GPU 5x
* [ ] Runtime multiplier switching

Target example:

```text
1x → 2x → 5x → 3x → 1x
```

Also:

* [ ] resize;
* [ ] fullscreen transition;
* [ ] rapid recreate;
* [ ] `VK_ERROR_OUT_OF_DATE_KHR`;
* [ ] `VK_ERROR_SURFACE_LOST_KHR`;
* [ ] DeviceLost behavior;
* [ ] multiple swapchains;
* [ ] same raw handle reuse;
* [ ] frame pacing;
* [ ] latency analysis;
* [ ] long-running stability.

---

## User-facing completion

* [ ] Production Dual-GPU configuration
* [ ] Device-pair selection
* [ ] Compatibility diagnostics
* [ ] UI exposure
* [ ] Packaging
* [ ] Release artifact
* [ ] Upgrade / rollback documentation

---

# 🎯 Definition of Dual-GPU complete

This project will only describe Dual-GPU frame generation as production-ready when the complete route:

```text
Application
    │
    ▼
GPU A original source
    │
    ▼
A → B transport
    │
    ▼
GPU B LSFG generation
    │
    ▼
B → A return
    │
    ▼
GPU A physical presentation
```

works continuously with:

* correct Vulkan synchronization;
* explicit lifetime ownership;
* generation-safe WSI handling;
* safe swapchain destroy / recreate;
* no stale raw-handle use;
* no semaphore-generation aliasing;
* no device-wide waits in the normal runtime path;
* correct error semantics;
* runtime 1x–5x switching;
* multi-swapchain safety;
* acceptable pacing;
* acceptable latency;
* real hardware qualification.

Until then:

> **Same-GPU remains the safe production default.**

---

# 🚫 What this project does not currently claim

The project does **not** currently claim:

* production-ready Dual-GPU frame generation;
* that Root X2–X5 routing is enabled;
* that normal D3B3 routing is enabled;
* universal multi-GPU compatibility;
* universal AMD → NVIDIA compatibility;
* universal NVIDIA → AMD compatibility;
* universal Intel combinations;
* PCIe peer-to-peer access;
* physical zero-copy;
* universal DMA-BUF modifier compatibility;
* universal external-image support;
* final performance numbers;
* final latency numbers;
* strict mixed Native/D2/Virtual X2–X5 support;
* complete 2x–5x hardware qualification.

A capability is only promoted to this README as production-ready after its corresponding qualification gate is completed.

---

# 📦 Installation

> [!IMPORTANT]
> **This repository is an active development fork.**
>
> The official upstream release package from `PancakeTAS/lsfg-vk` does **not** automatically contain the development changes described in this repository.
>
> If you want this fork specifically, build or install from this repository / one of its published fork artifacts when available.

---

## Clone the public fork

```bash
git clone https://github.com/wizardkof/lsfg-vk-custom.git
cd lsfg-vk-custom
```

The GitHub default branch is:

```text
fixed-target
```

To explicitly select it:

```bash
git switch fixed-target
```

> [!NOTE]
> Qualification branches may exist separately from the default branch.
>
> The last published qualification branch currently documented by the project is:
>
> ```text
> d3b3-prea6b-pfsr-qualified-20260831
> ```
>
> Later local PRE-A6B development may not yet be published to a new qualification branch.

---

# 🛠 Build from source

The project uses CMake and C++20.

The current top-level build options include:

```text
LSFGVK_BUILD_VK_LAYER
LSFGVK_BUILD_UI
LSFGVK_BUILD_CLI
LSFGVK_BUILD_TESTS
LSFGVK_INSTALL_DEVELOP
LSFGVK_INSTALL_XDG_FILES
```

Current defaults are:

```text
Vulkan layer = ON
CLI          = ON
UI           = OFF
Tests        = OFF
```

---

## Basic Release build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLSFGVK_BUILD_VK_LAYER=ON \
  -DLSFGVK_BUILD_CLI=ON

cmake --build build -j
```

---

## Build with GUI

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLSFGVK_BUILD_VK_LAYER=ON \
  -DLSFGVK_BUILD_CLI=ON \
  -DLSFGVK_BUILD_UI=ON

cmake --build build -j
```

The GUI requires Qt6, Qt6 Quick and Qt6 Quick Controls.

---

## Build regression tests

```bash
cmake -S . -B build-tests \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLSFGVK_BUILD_TESTS=ON

cmake --build build-tests -j
ctest --test-dir build-tests --output-on-failure
```

> [!NOTE]
> Development / qualification branches may require additional build dependencies used by the current Vulkan, DMA-BUF, WSI or test paths.

---

# 🖼 GUI dependencies

## Debian / Ubuntu

```bash
sudo apt install \
  qt6-qpa-plugins \
  libqt6quick6 \
  qml6-module-qtquick-controls \
  qml6-module-qtquick-layouts \
  qml6-module-qtquick-window \
  qml6-module-qtquick-dialogs \
  qml6-module-qtqml-workerscript \
  qml6-module-qtquick-templates \
  qml6-module-qt-labs-folderlistmodel
```

## Arch Linux / CachyOS

```bash
sudo pacman -S qt6-declarative qt6-base
```

## Fedora

```bash
sudo dnf install qt6-qtdeclarative qt6-qtbase
```

---

# 🖥 Graphical configuration

When installed, the configuration application can be launched as:

```bash
~/.local/bin/lsfg-vk-ui
```

The GUI provides profile-based configuration.

Typical concepts include:

* Global Settings
* Profile Settings
* executable matching through `active_in`
* frame-generation mode
* multiplier
* Fixed Target cadence

---

# 📝 Manual configuration

The normal configuration file is:

```text
~/.config/lsfg-vk/conf.toml
```

Profiles use `active_in` to select applications.

Example Adaptive configuration:

```toml
frame_generation_mode = "adaptive"
multiplier = 2
```

Example Adaptive 1x bypass:

```toml
frame_generation_mode = "adaptive"
multiplier = 1
```

Example Fixed Target:

```toml
frame_generation_mode = "fixed"
target_fps = 60
```

Validate configuration with:

```bash
~/.local/bin/lsfg-vk-cli validate
```

See [Configuration](docs/Configuration.md) for the detailed configuration model.

---

# 📊 Benchmarking

The CLI includes benchmarking support:

```bash
~/.local/bin/lsfg-vk-cli benchmark
```

By default, the benchmark runs for 10 seconds.

Use:

```bash
~/.local/bin/lsfg-vk-cli benchmark --help
```

for available options.

---

# 🧰 Diagnostic commands

Cross-device development diagnostics include:

```bash
lsfg-vk-cli devices
lsfg-vk-cli interop
lsfg-vk-cli interop-buffer-probe --allocator PATH --device-a INDEX --device-b INDEX
lsfg-vk-cli interop-sync-fd-probe --allocator PATH --device-a INDEX --device-b INDEX
```

These are development / diagnostic tools.

They do not by themselves enable production Dual-GPU frame generation.

---

# ⚠️ Current limitations

* Fixed Target does not directly cap application source FPS.
* Adaptive 1x does not completely unload the layer/backend.
* Same-GPU remains the safe normal runtime.
* Production Dual-GPU frame generation is not enabled.
* Root X2–X5 routing remains disabled.
* Normal D3B3 routing remains disabled.
* Strict mixed Native/D2/Virtual X2–X5 is deferred.
* Global virtual WSI teardown is still under PRE-A6B corrective qualification.
* Final continuous Dual-GPU 2x–5x hardware validation has not been completed.
* Current cross-device probe validation applies only to the tested environments.
* DMA-BUF probe success does not imply PCIe P2P.
* DMA-BUF probe success does not imply physical zero-copy.
* No universal external-memory / modifier compatibility claim is made.
* Final performance and latency characteristics are not yet claimed.

---

# 🌳 Branch model

The repository currently has two concepts that should not be confused.

## GitHub default branch

```text
fixed-target
```

This remains the public default branch of the repository.

## Development / qualification line

The deeper D3B3 work progressed through dedicated development and qualification branches.

The last published qualification checkpoint currently documented is:

```text
branch:
d3b3-prea6b-pfsr-qualified-20260831

commit:
633901221784220ffe4beaf2a3de0e1503c35b6c
```

Later scheduler and PRE-A6B lifetime work has continued beyond that checkpoint.

Until a new checkpoint is intentionally committed / tagged / pushed, the README should not imply that those later local changes already exist on a public qualification branch.

---

# 🔗 Upstream

This project is based on:

**PancakeTAS/lsfg-vk**

https://github.com/PancakeTAS/lsfg-vk

The fork tracks experimental development separately.

Upstream same-GPU behavior is preserved wherever possible while Fixed Target and D3B3 functionality are developed behind explicit qualification gates.

For the upstream stable / official release, use the upstream repository and its official releases.

---

# ❤️ Credits

* [PancakeTAS/lsfg-vk](https://github.com/PancakeTAS/lsfg-vk) — upstream Vulkan layer
* [Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) — frame-generation technology
* Khronos Vulkan ecosystem
* Mesa / RADV
* NVIDIA Linux Vulkan driver ecosystem

---

# 📄 License

See [LICENSE.md](LICENSE.md).

---

<div align="center">

### Current objective

**Production-safe Dual-GPU frame generation without sacrificing lsfg-vk's same-GPU reliability.**

Fixed Target is already part of the fork.

D3B3 Dual-GPU development remains gated until synchronization, lifetime, WSI teardown, Root integration and hardware qualification are complete.

</div>
