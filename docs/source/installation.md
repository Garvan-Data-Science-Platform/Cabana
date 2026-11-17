# Installation

## System Requirements

1. **Processor (CPU):** Intel Core i7 or AMD Ryzen 7, or higher for better performance
2. **Memory (RAM):** Minimum 16 GB, with 32 GB preferred
3. **Graphics Processing Unit (GPU):** NVIDIA GeForce GTX/RTX series GPU with at least 10 GB VRAM

The segmentation network runs on the NVIDIA GPU when CUDA is available. On Apple silicon Macs it runs on the Apple GPU through PyTorch's MPS backend automatically (about 3 to 4 times faster than the CPU), falling back to the CPU if an operation is unsupported. Set the environment variable `CABANA_DEVICE` to `cpu`, `cuda` or `mps` to override the choice.

## Installation Instructions

1. Install miniconda. The default installation location is `C:\\ProgramData\\miniconda3` on Windows. Verify the installation by running `conda env list` in the command line. If the installation was successful, you should be able to see a `base` env and the path to the env.

2. Create a conda env named `cabana` (you can change it to any name you want) by running `conda create -n cabana python=3.12` in the command line. If the env is created correctly, activate the created conda env by running `conda activate cabana`.

3. On Windows with an NVIDIA graphics card, `pip install cabana` on its own installs the CPU-only PyTorch from PyPI, and segmentation then runs on the CPU. The simplest fix is to finish the installation and then choose **Help > Install GPU Support** in Cabana (see [Checking that the graphics card is used](#checking-that-the-graphics-card-is-used)). To do it by hand instead, install the CUDA build of PyTorch before Cabana: choose *Windows*, *Pip* and a CUDA version at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and run the command shown there, for example `pip install torch --index-url https://download.pytorch.org/whl/cu128`. Cabana only needs `torch`. Skip this step on macOS, on Linux and on computers without an NVIDIA card.

4. Install Cabana by running `pip install -U cabana`. If the installation is successful, you can start Cabana GUI by running `python -m cabana`. The same install provides the `cabana-tma` command (also `python -m cabana tma`) for [TMA preprocessing](tma.md) without the GUI. Alternatively, you can import Cabana in your code for more customized analysis (see examples in <a href="_static/tutorial.ipynb" target="_blank" rel="noopener">tutorial.ipynb</a>).

The steps above are written for Windows. On macOS and Linux install miniconda for your platform, run the same `conda create` and `conda activate` commands in a terminal, then `pip install -U cabana`. The install also provides two console commands, `cabana-gui` (the GUI) and `cabana-tma` (TMA preprocessing without the GUI). On Apple silicon Macs the segmentation network runs on the Apple GPU automatically.

### Checking that the graphics card is used

Run this in the `cabana` env:

```
python -c "from cabana.segmenter import select_device; print(select_device())"
```

It prints `cuda` (NVIDIA), `mps` (Apple silicon) or `cpu`. Every run also writes the device to its log (`Segmentation device: cuda`).

If it prints `cpu` on a computer with an NVIDIA card, PyTorch is the CPU-only build (`python -c "import torch; print(torch.__version__)"` ends in `+cpu`). Cabana says so once when it starts, with an **Install GPU Support** button, and in the log; the same command is under **Help > Install GPU Support**. It closes Cabana and opens a small window that looks up the newest PyTorch build the NVIDIA driver supports, downloads it (about 2.5 GB) and replaces the CPU-only build in Cabana's environment; start Cabana again when the window reports that it is done. Cabana has to close because Windows does not allow PyTorch's files to be replaced while they are in use. If the installation fails, the previous build stays in place and the window shows pip's output. An account that may not change the environment is told so; an administrator then runs the command. A PyTorch that was installed with `conda install` is left alone, because replacing it with pip would leave conda's records wrong: switch builds with conda, or remove the conda package and install with pip as in step 3. Updating Cabana afterwards keeps the CUDA build. To do it by hand, close Cabana and run `pip uninstall torch` in the `cabana` env, then the install command from step 3. If PyTorch is a CUDA build and the device is still `cpu`, update the NVIDIA driver.

The environment is shared by everyone who starts Cabana from it, so one reinstall is enough for an environment in a shared folder such as `C:\\ProgramData\\miniconda3`; run it from a command line opened with *Run as administrator*.

If you want to create a shortcut on the Windows desktop, follow the steps below:

1) Make sure that Git is installed. If not, download it from this link and install it. Verify the installation by running `git --version` in the command line. 

2) Run `git clone https://github.com/Garvan-Data-Science-Platform/Cabana.git` in the command line. You will be prompted to log in using your GitHub account. If authentication is successful, a `Cabana` folder will be downloaded to your current working directory. You can move (copy-and-paste) the folder to any installation location and change the current working direction to `Cabana` folder.

3) Modify the content in cabana.bat. Make sure CONDAPATH is set to the folder where miniconda was installed (the `base` row of `conda env list`; it must contain `Scripts\activate.bat`), ENVNAME is set to `cabana` and ENVPATH to the `cabana` environment's folder as shown by `conda env list` (edit it when the environments are not under the miniconda folder, for example `C:\GCondaEnv\envs\cabana`), and the last command before `call conda deactivate` is `python -m cabana`. Keep the `cd /d "%USERPROFILE%"` line before it: `python -m cabana` loads a `cabana` folder in the current folder before the installed package, so started from the downloaded `Cabana` folder it would run that copy instead of the one `pip` installs and updates.

4) Create a `Cabana` shortcut on `C:\\Users\\Public\\Desktop` for `cabana.bat`. Change the icon of Cabana to any icon you want, or you can use the `cabana-logo.ico` in the downloaded `Cabana` folder from Github. You are now ready to use Cabana by double-clicking the shortcut.

## Updating Cabana

Choose **Help > Check for Updates** (on macOS: **Cabana > Check for Updates**). Cabana asks PyPI for the latest release and, when a newer one exists, offers to install it into the environment Cabana is running in. Close Cabana and start it again to use the new version. If the installation fails, the message shows pip's output; Cabana keeps working.

Cabana also checks by itself once a day, a few seconds after it starts, and only speaks up when a newer version is available. Choose **Skip This Version** to stop being reminded of that version; a later version is offered again, and **Check for Updates** always offers it.

The update always goes to the copy of Cabana that is running, and Cabana checks afterwards that it arrived there:

- **Shared installations** (for example an environment under `C:\ProgramData` used by several Windows accounts). An account that may not change that folder is told so and is not offered the update; an administrator updates it with `python -m pip install --upgrade cabana` in that environment. Earlier versions let pip install a second copy into the account's own folder (`AppData\Roaming\Python`) instead, which then hid the shared one for that account.
- **A copy in your own folder.** If such a per-account copy already exists, the update message says so and updates that copy for your account only. To return to the shared copy, run `python -m pip uninstall cabana` once from your account in the `cabana` env: it removes your own copy first.
- **Leftover install records.** When an interrupted update has left more than one install record, **Check for Updates** offers to repair it (also when Cabana is up to date): it removes the leftovers and reinstalls Cabana. Nothing has to be deleted by hand.

Versions 0.3.10 and earlier have no update check: update them once by hand with `pip install -U cabana` in the `cabana` env (run `conda activate cabana` first).

When Cabana is not running from the installed package, **Check for Updates** shows the new version but does not install it, and the daily check stays silent. This happens when a `cabana` folder in the folder Cabana was started from is loaded instead (see step 3 of the shortcut instructions above), or when Cabana runs from a git checkout or an editable install (`pip install -e .`). Update such a copy with `git pull` or `pip`.

If an earlier upgrade was interrupted, the environment can hold more than one Cabana install record (`cabana-<version>.dist-info` folders in `site-packages`). Cabana then shows a wrong version number and an update seems not to take effect. **Check for Updates** lists the folders: close Cabana, delete them, and run `pip install --force-reinstall --no-deps cabana` in the `cabana` env. Do not use `pip uninstall` for this, as it may remove files of the current installation.
