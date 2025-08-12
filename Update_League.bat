@echo off
setlocal enableextensions enabledelayedexpansion

rem === 0) Přechod do složky projektu ===
cd /d C:\Projekt\Poisson || (
    echo ❌ Nelze najít složku C:\Projekt\Poisson
    pause
    exit /b 1
)

rem === 1) Zvol Python ===
if exist .venv\Scripts\python.exe (
  set "PY=.venv\Scripts\python.exe"
) else if exist venv\Scripts\python.exe (
  set "PY=venv\Scripts\python.exe"
) else (
  set "PY=python"
)

rem === 2) Git stash a pull ===
echo 🔄 Ukládám necommitnuté změny...
git stash push -m "auto-stash před pull"
IF %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Git stash selhal.
)

echo 🔄 Stahuji změny z GitHubu...
git pull --rebase
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Git pull selhal.
    pause
    exit /b 1
)

echo 🔄 Obnovuji změny...
git stash pop

rem === 3) Aktualizace CSV ===
echo === 1/3: Spouštím update_league_data.py ===
%PY% update_league_data.py
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Selhala aktualizace dat.
    pause
    exit /b 1
)

rem === 4) Commit & Push ===
echo === 2/3: Commit & Push ===
%PY% commit_and_push.py
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Commit & push selhal.
    pause
    exit /b 1
)

echo ✅ Všechno hotovo. Zavřete okno nebo stiskněte klávesu.
pause
