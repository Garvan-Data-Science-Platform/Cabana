@echo OFF
rem How to run a Python script in a given conda environment from a batch file.

rem It doesn't require:
rem - conda to be in the PATH
rem - cmd.exe to be initialized with conda init

rem Define here the path to your conda installation
set CONDAPATH=C:\ProgramData\miniconda3\
rem Define here the name of the environment
set ENVNAME=cabana
set ENVPATH=%CONDAPATH%\envs\%ENVNAME%

rem Activate the conda environment
rem Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

rem Start in the home folder: `python -m cabana` would load a cabana folder in
rem the current folder (e.g. a downloaded copy of Cabana) instead of the installed package
cd /d "%USERPROFILE%"

rem Run a python script in that environment
python -m cabana

rem Deactivate the environment
call conda deactivate

pause
