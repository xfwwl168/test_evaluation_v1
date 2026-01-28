@echo off
chcp 65001 >nul
color 0A
title 量化交易引擎 v1.0

REM ============================================
REM 量化交易引擎启动脚本
REM ============================================

echo.
echo ======================================================================
echo                     量化交易引擎 v1.0
echo ======================================================================
echo.
echo 正在启动...
echo.

REM 切换到脚本所在目录
cd /d %~dp0

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.9+
    echo.
    echo 下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM 检查 menu.py 是否存在
if not exist "menu.py" (
    echo [错误] 找不到 menu.py 文件
    echo.
    echo 请确保 menu.py 在当前目录:
    echo %cd%
    echo.
    pause
    exit /b 1
)

REM 运行菜单程序
python menu.py

REM 退出时显示信息
echo.
echo ======================================================================
echo 感谢使用量化交易引擎！
echo ======================================================================
echo.
pause
