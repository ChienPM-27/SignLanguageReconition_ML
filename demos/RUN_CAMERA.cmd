@echo off
cd /d "%~dp0.."
".venv\Scripts\python.exe" -X utf8 "demos\camera_v3.py" %*
if errorlevel 1 pause
