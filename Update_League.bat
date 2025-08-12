@echo off
setlocal enableextensions enabledelayedexpansion

rem === Define log path and ensure log folder exists ===
set LOGDIR=logs
set LOGFILE=%LOGDIR%\bat_log.txt
if not exist %LOGDIR% (
    mkdir %LOGDIR%
)

echo === START %DATE% %TIME% === > %LOGFILE%

rem === Change to project directory ===
cd /d C:\Projekt\Poisson || (
    echo [ERROR] Could not find folder: C:\Projekt\Poisson >> %LOGFILE%
    echo Failed to change directory. >> %LOGFILE%
    type %LOGFILE%
    pause
    exit /b 1
)

rem === Select Python interpreter ===
if exist .venv\Scripts\python.exe (
    set "PY=.venv\Scripts\python.exe"
) else if exist venv\Scripts\python.exe (
    set "PY=venv\Scripts\python.exe"
) else (
    set "PY=python"
)

rem === Check for uncommitted changes before pull ===
echo [INFO] Checking for uncommitted changes... >> %LOGFILE%
git diff --quiet || (
    echo [ERROR] You have unstaged changes. Please commit or stash them manually. >> %LOGFILE%
    git status >> %LOGFILE%
    type %LOGFILE%
    pause
    exit /b 1
)

rem === Git stash and pull ===
echo [INFO] Stashing local changes... >> %LOGFILE%
git stash push -m "auto-stash before pull" >> %LOGFILE% 2>&1

echo [INFO] Pulling from remote... >> %LOGFILE%
git pull --rebase >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git pull failed. >> %LOGFILE%
    type %LOGFILE%
    pause
    exit /b 1
)

echo [INFO] Re-applying stash... >> %LOGFILE%
git stash pop >> %LOGFILE% 2>&1

rem === Run data cleaning script ===
echo [STEP 1/3] Running clean_existing_csvs.py... >> %LOGFILE%
%PY% Data\clean_existing_csvs.py >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CSV cleaning failed. >> %LOGFILE%
    type %LOGFILE%
    pause
    exit /b 1
)

rem === Commit and push changes ===
echo [STEP 2/3] Running commit_and_push.py... >> %LOGFILE%
%PY% commit_and_push.py >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Commit & push failed. >> %LOGFILE%
    type %LOGFILE%
    pause
    exit /b 1
)

echo [STEP 3/3] All tasks completed successfully. >> %LOGFILE%
type %LOGFILE%
pause
