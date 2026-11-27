@echo off
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat goto noenv
call venv\Scripts\activate.bat
python -m streamlit run app.py
pause
exit /b

:noenv
echo The environment is not ready. Open START_HERE.bat and choose option 1 first.
pause
