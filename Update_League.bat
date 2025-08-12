@echo off
setlocal enableextensions enabledelayedexpansion

rem === Step 0: Change to project folder ===
cd /d C:\Projekt\Poisson || (
    echo [ERROR] Could not find folder: C:\Projekt\Poisson
    pause
    exit /b 1
)

rem === Step 1: Select Python interpreter ===
if exist .venv\Scripts\python.exe (
    set "PY=.venv\Scripts\python.exe"
) else if exist venv\Scripts\python.exe (
    set "PY=venv\Scripts\python.exe"
) else (
    set "PY=python"
)

rem === Step 2: Git stash and pull ===
echo [INFO] Stashing local changes...
git stash push -m "auto-stash before pull"
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] Git stash failed.
)

echo [INFO] Pulling changes from remote...
git pull --rebase
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git pull failed.
    pause
    exit /b 1
)

echo [INFO] Applying stashed changes...
git stash pop

rem === Step 3: Update CSV files ===
echo [STEP 1/3] Running update_league_data.py...
%PY% update_league_data.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Data update failed.
    pause
    exit /b 1
)

rem === Step 4: Commit & push changes ===
echo [STEP 2/3] Committing and pushing changes...
%PY% commit_and_push.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Commit and push failed.
    pause
    exit /b 1
)

echo [DONE] All operations completed successfully.
pause
