# Configuration Options

Configuring lsfg-vk is done either through **lsfg-vk-ui** (graphical interface) or by manually editing the configuration file located at `~/.config/lsfg-vk/conf.toml`.

Regardless of the method you choose, the concept of profiles remains the same.
- **Profiles**: Profiles allow you to create different sets of configurations for different applications or use cases. A profile can automatically be selected through the "active_in" property.
- **Profile Settings**: Settings related to a profile are stored under the "Profile Settings" or `[[profile]]` section.
- **Global Settings**: Any settings in the "Global Settings" or `[global]` section apply to all profiles.

### All Configuration Options

Below is a list of all available **global** configuration options:
- **Path to Lossless Scaling / `dll`**: By default, lsfg-vk will search certain directories for Lossless Scaling. If you have Lossless Scaling installed in a custom location, you can specify the full path to the "Lossless.dll" file inside of Lossless Scaling here.
- **Allow half-precision / `allow_fp16`**: If enabled, this will allow lsfg-vk to take advantage of half-precision shader operations if supported by the GPU. This has a giant performance uplift on AMD GPUs, but does not affect NVIDIA GPUs (GTX 1000-series or older cards will actually see a big performance **decrease**). This option **does not** influence quality. (Default: `true`)

Next is a list of all available **profile** configuration options:
- **Profile Name / `name`**: The name of the profile, displayed in **lsfg-vk-ui**. Additionally, this is used when selecting a profile through the `LSFGVK_PROFILE` environment variable.
- **Active In / `active_in`**: A list of 1) linux binary names, such as `mpv`, 2) windows executables, such as `GenshinImpact.exe` and 3) process names, such as `GameThread`. It is also possible to specify the last part of a path (e.g. `Ghostrunner2/Binaries/Win64/Ghostrunner2-Win64-Shipping.exe`). When a process matching one of these rules is detected, this profile will be activated.
- **Frame Generation Mode / `frame_generation_mode`**: Selects how output frames are scheduled. `adaptive` preserves the normal multiplier-driven behavior. `fixed` targets an explicit output frame rate using `target_fps`. (Default: `adaptive`)
- **Target FPS / `target_fps`**: Output cadence target used by Fixed mode. It must be greater than `0` when `frame_generation_mode = "fixed"`. It does not directly cap the application's source FPS and is ignored by Adaptive mode.
- **Multiplier / `multiplier`**: The Adaptive frame generation multiplier. Supported Adaptive values in this fork are `1` through `5`. In Adaptive, a value of `1` keeps the layer/profile active but bypasses frame generation, while a value of `3` generates 2 additional frames for every application frame. Fixed mode ignores the multiplier; in particular, `multiplier = 1` does not bypass Fixed. (Default: `2`)
- **Flow Scale / `flow_scale`**: The resolution scale at which the motion vectors are calculated. A lower value means better performance, but worse quality. (Default: `1.0`)
- **Performance Mode / `performance_mode`**: When enabled, a significantly lighter frame generation model is used. This has a minor quality impact, but greatly improves performance. 
(Default: `false`)
- **Pacing Mode / `pacing`**: This option is explained in greater detail below. Supported values are **None / `none`**.
- **GPU / `gpu`**: The GPU to use for frame generation. This MUST currently be the **same GPU** as the one being used by the application because **Dual GPU is not supported yet**. If omitted (the UI's **Default** option), lsfg-vk correlates the backend with the application's physical device using its Vulkan `deviceUUID`, with the full PCI address as a fallback when available. An explicit GPU can still be identified by name (e.g. `NVIDIA GeForce RTX 3080`), uppercase-only ID (e.g. `0x10DE:0x2C02`), legacy PCI bus ID (e.g. `3:0.0`), or domain-qualified PCI address (e.g. `0:3:0.0`).

The "Multiplier", "Flow Scale", "Performance Mode" and Fixed `target_fps` options can be **hot-reloaded**. `frame_generation_mode` can also switch between Adaptive and Fixed in the same process when the swapchain supports the dual-mode topology. On fallback hardware, Adaptive may use the legacy real/FIFO path and can switch to synchronous Fixed/FIFO; a Fixed virtual swapchain that could not declare compatible dual present modes will conservatively block a switch back to Adaptive until the application recreates its swapchain. Options such as "Pacing Mode" still require a swapchain recreation. Removing the active profile is not a supported hot-disable operation: the layer retains the last active profile and its global settings until the process restarts, preserving the existing virtual swapchain images.

### Adaptive and Fixed Examples

Adaptive 1x bypass keeps the profile/layer active without generating frames:

```toml
[[profile]]
name = "Adaptive bypass"
active_in = "Game.exe"
frame_generation_mode = "adaptive"
multiplier = 1
target_fps = 0
```

Fixed mode ignores `multiplier` and targets an explicit output cadence. The
example remains active Fixed frame generation even with `multiplier = 1`:

> [!IMPORTANT]
> In the asynchronous virtual path, Fixed mode controls the output cadence and
> does not directly impose a lower source frame-rate cap on the application. If
> the application already renders at `target_fps`, no intermediate frames are
> needed. Synchronous/fallback Fixed paths may pace the source directly. To
> trade source rendering load for generated frames in the asynchronous path,
> cap the application's source FPS below `target_fps` using the game or an
> external limiter.

> [!NOTE]
> The `--multiplier` option of `lsfg-vk-cli benchmark` and `lsfg-vk-cli debug`
> controls those tools' direct backend interpolation workload. Those commands
> do not create the Vulkan-layer swapchain/pacer topology, so their multiplier
> is separate from profile `frame_generation_mode`, Adaptive 1x bypass, and
> Fixed `target_fps`.

```toml
[[profile]]
name = "Fixed 120 FPS"
active_in = "Game.exe"
frame_generation_mode = "fixed"
target_fps = 120
multiplier = 1
```

### Pacing Modes

**Pacing modes** determine how lsfg-vk synchronizes frame generation with the application's frame rate.

Traditionally, lsfg-vk did not have frame pacing and would present frames to the screen as soon as they are generated. This approach is flawed, because frames are generated and presented much quicker than your screen refreshes, causing frames to be skipped. This effect is countered by force-enabling V-Sync, as the compositor will then wait for the next screen update, before presenting the next frame.

Enabling V-Sync is not a "get-out-of-jail-free" card, because it introduces input latency. Additionally, not every compositor (such as gamescope) respects the V-Sync setting, leading to the same issues as before. As a result of this, additional pacing modes have been introduced to properly handle frame pacing.

Here are all available pacing modes:
- `none`: Traditional lsfg-vk behavior. Forces V-Sync. Might require workarounds on some compositors.
- *... there are no other pacing modes yet ...*

### Environment Variables

The following environment variables affect lsfg-vk:
- `DISABLE_LSFGVK`: If set, lsfg-vk will be completely disabled.
- `LSFGVK_CONFIG`: Path to the configuration file.
- `LSFGVK_PROFILE`: Name of the profile to use. If set, this will override automatic profile detection.

If you do not wish to use a configuration file, you can also set configuration options through environment variables. To do this, set `LSFGVK_ENV=1` and then any of the following variables:
- `LSFGVK_DLL_PATH`: Path to Lossless Scaling DLL.
- `LSFGVK_NO_FP16`: If set to `1`, half-precision will be disabled.
- `LSFGVK_MULTIPLIER`: Adaptive frame generation multiplier (`1` through `5`; `1` is bypass).
- `LSFGVK_FRAME_GENERATION_MODE`: `adaptive` or `fixed`.
- `LSFGVK_TARGET_FPS`: Positive output FPS target required by Fixed mode.
- `LSFGVK_FLOW_SCALE`: Flow scale value.
- `LSFGVK_PERFORMANCE_MODE`: If set to `1`, performance mode will be enabled.
- `LSFGVK_PACING`: Pacing mode to use.
- `LSFGVK_GPU`: GPU to use for frame generation.
