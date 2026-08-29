@echo off
title VeriTrust AI Launcher
echo ==================================================
echo         VERITRUST AI: FORENSIC PLATFORM          
echo ==================================================
echo Launching full-stack environment...
powershell -ExecutionPolicy Bypass -File "%~dp0start_veritrust.ps1"
pause
