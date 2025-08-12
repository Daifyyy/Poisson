@echo off
setlocal enableextensions enabledelayedexpansion

set LOGFILE=bat_log.txt
echo === START %DATE% %TIME% === > %LOGFILE%

cd /d C:\Projekt\Poisson || (
    echo [ERROR] Could not find folder: C:\Projekt\Poisson >> %LOGFILE%
    echo Failed to change directory. >> %LOGFILE%
    type %LOGFILE%
    pause
    exit /b 1
)

if exist .venv\Scripts\python.exe (
    set "PY=.venv\Scripts\python.exe"
) else if exist venv\Scripts\python.exe (
    set "PY=venv\Scripts\python.exe"
) else (
    set "PY=python"
)

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

echo [STEP 1/3] Running clean_existing_csvs.py... >> %LOGFILE%
%PY% clean_existing_csvs.py >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CSV cleaning failed. >> %LOGFILE%
    type %LOGFILE%
    pause
    exit /b 1
)

echo [STEP 2/3] Running commit_and_push.py... >> %LOGFILE%
%PY% commit_and_push.py >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Commit & push failed. >> %LOGFILE%
    type %LOGFILE%
    pause
    exit /b 1
)

echo [DONE] All tasks completed successfully. >> %LOGFILE%
type %LOGFILE%
pause
