# lsfg-vk — Fixed Target / Dual-GPU Development Fork

This repository is the **fixed-target** development fork of lsfg-vk.

- Fork version: **2.0.0-fixed-target**
- Upstream baseline: **2.0.0-dev28**
- Development branch: **fixed-target**

The fork preserves same-GPU operation as the safe default while developing
explicit fixed-cadence output and isolated cross-device transport probes.

**Lossless Scaling** is a Windows-exclusive program featuring various algorithms for scaling and interpolating programs.

**lsfg-vk** is a Vulkan layer that hooks into Vulkan applications and generates additional frames using Lossless Scaling's frame generation algorithm.

>[!CAUTION]
> You are reading the README for the upcoming version 2.0 of lsfg-vk. For the stable version 1.x, [please read here](https://github.com/PancakeTAS/lsfg-vk/tree/ff1a0f72a7d6d08b84d58b7b4dc5f05c9f904f98)

## Installation
>[!TIP]
> If you are on a Steam Deck or similar handheld, consider using the [Decky plugin for lsfg-vk](https://github.com/xXJSONDeruloXx/decky-lsfg-vk). This is an easy way to install and configure lsfg-vk on the Steam Deck.
> Please keep in mind that it is not officially supported and support questions should be directed to the plugin's repository & discord.

1. Before proceeding, please make sure you have [Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) downloaded on Steam.
2. Head to the [GitHub Releases](https://github.com/PancakeTAS/lsfg-vk/releases) and download the file named "lsfg-vk-2.0.0-linux.tar.xz".
3. Open a terminal in the folder where you downloaded the file and run the following:
```bash
tar -xvf lsfg-vk-2.0.0-linux.tar.xz -C ~/.local
```
This will extract lsfg-vk to `~/.local`. Please **keep track of the files that were extracted**, in case you want to uninstall lsfg-vk later.

4. The graphical interface requires Qt6, Qt6 Quick and Qt6 Quick Controls 2 in order to run. If you do not have these installed, install the following packages:
```bash
sudo apt install qt6-qpa-plugins libqt6quick6 qml6-module-qtquick-controls qml6-module-qtquick-layouts qml6-module-qtquick-window qml6-module-qtquick-dialogs qml6-module-qtqml-workerscript qml6-module-qtquick-templates qml6-module-qt-labs-folderlistmodel # On Debian/Ubuntu-based systems
sudo pacman -S qt6-declarative qt6-base # On Arch-based systems
sudo dnf install qt6-qtdeclarative qt6-qtbase # On Fedora
```

5. (Optional) If you wish to use lsfg-vk within Flatpak applications, see the [Flatpak Guide](docs/Flatpak-Guide.md).

## Usage
In order to start using lsfg-vk, you will need to configure it. This can either be done using the GUI application, or manually.

### Graphical Configuration
Start 'lsfg-vk Configuration Window' from your application launcher, or run `~/.local/bin/lsfg-vk-ui` in a terminal:
- On the left side, you will see a list of profiles. Each profile has its own settings.
- All properties in the "Global Settings" section apply to all profiles.
  - Should Lossless Scaling be installed in a non-standard location, you can specify the path here.
- Select a profile and configure the "Profile Settings" section to your liking.
  - When editing the "Active In" list, you can add a game using its executable name (e.g. `Game.exe`, `mpv`).
- Please see the [documentation](docs/Configuration.md) for detailed information on each setting.
- Once you are done configuring, simply starting a game that matches one of the profiles will automatically apply the settings.

### Manual Configuration
The default configuration is located in `~/.config/lsfg-vk/conf.toml`. It will be created automatically when any Vulkan application is started.
- In the `[global]` section, you can change where Lossless Scaling is installed, as well as other global settings.
- Each profile is defined in its own `[[profile]]` section.
- The `active_in` array/string defines which applications the profile is active in. You can add applications using their executable name (e.g. `Game.exe`, `mpv`).
- Please see the [documentation](docs/Configuration.md) for detailed information on each setting.
- Once you are done configuring, simply starting a game that matches one of the profiles will automatically apply the settings.

You can validate the configuration using `lsfg-vk-cli`:
```bash
~/.local/bin/lsfg-vk-cli validate
```

### Frame Generation Modes

Adaptive mode uses `multiplier` to select frame generation from 2x through 5x.
Adaptive 1x keeps the profile and layer active but bypasses frame generation:

```toml
frame_generation_mode = "adaptive"
multiplier = 1
```

Fixed mode instead targets an explicit output cadence. It ignores `multiplier`,
so `multiplier = 1` does not bypass Fixed mode:

```toml
frame_generation_mode = "fixed"
target_fps = 60
multiplier = 1
```

`target_fps` controls Fixed output cadence; it does not directly cap the
application's source FPS. See [Configuration](docs/Configuration.md) for the
complete semantics and hot-reload limitations.

Fixed Target also keeps a stable physical-device identity for diagnostics and
explicit device-pair experiments. Dual-GPU transport is not automatically
selected for production frame generation.

### Adaptive 1x bypass

Adaptive mode with `multiplier = 1` bypasses frame generation while keeping the
profile and layer active. This is a generation bypass, not a complete unload of
the Vulkan layer or backend.

## Cross-device development probes

The CLI exposes diagnostic commands. They are not production runtime features
and do not change the default same-GPU path:

```bash
lsfg-vk-cli devices
lsfg-vk-cli interop
lsfg-vk-cli interop-buffer-probe --allocator PATH --device-a INDEX --device-b INDEX
lsfg-vk-cli interop-sync-fd-probe --allocator PATH --device-a INDEX --device-b INDEX
```

The probes cover, in isolated steps:

- external buffer capability discovery;
- cross-device DMA-BUF `VkBuffer` import/use;
- binary semaphore `SYNC_FD` synchronization between Vulkan devices.

The DMA-BUF buffer probe and the `SYNC_FD` bridge were validated on the
development hardware combination **AMD Ryzen 9 7900 / RADV** and **NVIDIA RTX
3060**. This validates that specific diagnostic path and pair only. It is not a
claim of PCIe peer-to-peer access, physical zero-copy, performance, universal
compatibility, shared `VkImage` support, or modifier compatibility.

Cross-device transport is not integrated into LSFG frame generation, the
swapchain, the production backend, or the configuration UI. Same-GPU remains
the default safe runtime behavior.

## Current limitations

- Fixed Target does not limit the application's source FPS.
- Adaptive 1x does not completely unload the layer/backend.
- Dual-GPU operation is not integrated into the production runtime or UI.
- The validated probes do not establish PCIe P2P, physical zero-copy, or
  performance characteristics.
- Results are not a universal compatibility claim for other GPUs, drivers, or
  allocator paths.

### Benchmarking Mode
You can run a frame generation benchmark using `lsfg-vk-cli`:
```bash
~/.local/bin/lsfg-vk-cli benchmark
```

By default, the benchmark will run for 10 seconds. Use `--help` to see all available CLI options. The short option `-h` is reserved for `--height` in `benchmark` and `debug`.

## Support and Troubleshooting
If you encounter any issues or have questions regarding lsfg-vk, read through the [Troubleshooting](docs/Troubleshooting.md) documentation page or join the [Discord server](https://discord.gg/losslessscaling) for assistance.
