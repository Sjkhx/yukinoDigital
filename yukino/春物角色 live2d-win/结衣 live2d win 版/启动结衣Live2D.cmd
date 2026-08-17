@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 由比滨结衣 Live2D

if not exist "%~dp0程序文件\server.ps1" (
  echo [错误] 程序文件不完整。
  echo 请重新解压完整的文件夹后再试。
  pause
  exit /b 1
)

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0程序文件\server.ps1"

if errorlevel 1 (
  echo.
  echo 启动失败，请查看“使用指南.txt”中的常见问题。
  pause
)
