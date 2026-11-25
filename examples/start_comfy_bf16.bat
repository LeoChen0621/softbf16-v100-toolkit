@echo off
rem ComfyUI with bf16 weights on V100. Keep the usual flags; only the entry point changes.
rem Set these to your paths if they differ.
set COMFY=D:\ComfyUI
set PYTHON=%COMFY%\python\python.exe
set TOOLKIT=%~dp0..

set SOFTBF16_COMFY=%COMFY%
cd /d "%COMFY%"
"%PYTHON%" "%TOOLKIT%\shim\run_patched.py" --listen --preview-method auto --disable-cuda-malloc --bf16-unet
