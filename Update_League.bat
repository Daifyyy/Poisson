@echo off
setlocal enableextensions enabledelayedexpansion

rem === Změna složky ===
cd /d C:\Projekt\Poisson || (
    echo [ERROR] Nelze najít složku: C:\Projekt\Poisson
    pause
    exit /b 1
)

rem === Výběr interpretu Python ===
if exist .venv\Scripts\python.exe (
    set "PY=.venv\Scripts\python.exe"
) else if exist venv\Scripts\python.exe (
    set "PY=venv\Scripts\python.exe"
) else (
    set "PY=python"
)

rem === Kontrola necommitnutých změn ===
git diff --quiet || (
    echo [ERROR] Máš neuložené změny. Commitni nebo stashni ručně.
    git status
    pause
    exit /b 1
)

rem === Git stash, pull a pop ===
echo [INFO] Ukládám změny...
git stash push -m "auto-stash před pull"

echo [INFO] Stahuji nové změny...
git pull --rebase
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git pull selhal.
    pause
    exit /b 1
)

echo [INFO] Obnovuji stash...
git stash pop

rem === Spuštění čištění CSV ===
echo [KROK 1/3] Čistím CSV...
%PY% clean_existing_csvs.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Čištění CSV selhalo.
    pause
    exit /b 1
)

rem === Commit a push ===
echo [KROK 2/3] Commituji a pushuji změny...
%PY% commit_and_push.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Commit nebo push selhal.
    pause
    exit /b 1
)

echo [KROK 3/3] ✅ Hotovo! Vše úspěšně dokončeno.
pause
