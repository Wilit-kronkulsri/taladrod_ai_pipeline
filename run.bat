@echo off
rem One-click start on Windows: create .venv on first run, install, open the dashboard.
cd /d "%~dp0"

if not exist "pyproject.toml" (
    echo Please extract the whole zip first, then run run.bat from the extracted folder.
    pause
    exit /b 1
)

rem Find a real Python 3.10+. "where python" is not enough: Windows ships a Store stub named python.exe.
set "PY="
python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
    py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    echo Python 3.10 or newer was not found.
    echo Install it from https://www.python.org and tick "Add python.exe to PATH", then run this file again.
    pause
    exit /b 1
)

rem .installed is written only after pip finishes, so a closed window or a failed install is retried next time.
if not exist ".venv\.installed" (
    echo First run: creating .venv and installing packages. This takes several minutes.
    if not exist ".venv\Scripts\python.exe" %PY% -m venv ".venv"
    if errorlevel 1 goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto :fail
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 goto :fail
    type nul > ".venv\.installed"
)

echo Opening the dashboard at http://localhost:8501  (close this window to stop it)
rem headless skips Streamlit's first-run e-mail prompt, so open the browser ourselves
start "" /b cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo Something went wrong. See the message above.
echo If pip failed with "No such file or directory", move this folder to a short path such as D:\taladrod and try again.
pause
exit /b 1
