@echo off
setlocal
cd /d "%~dp0"
title ISL to Text and Voice

:menu
cls
echo ================================================================
echo   Indian Sign Language to Text and Voice  -  Python 3.11, XGBoost
echo ================================================================
echo.
echo   1  First-time setup   - install the libraries, only once
echo   2  Check my setup     - libraries, camera, voice
echo   3  Record gesture data
echo   4  Train the model
echo   5  Run the app
echo   6  Run the tests
echo   7  Add or re-record ONE sign   - asks for its name
echo   8  Delete the recorded data of ONE sign
echo   9  Exit
echo.
set "choice="
set /p choice=Type 1-9 and press Enter: 
if "%choice%"=="1" goto setup
if "%choice%"=="2" goto check
if "%choice%"=="3" goto record
if "%choice%"=="4" goto train
if "%choice%"=="5" goto app
if "%choice%"=="6" goto tests
if "%choice%"=="7" goto onesign
if "%choice%"=="8" goto delsign
if "%choice%"=="9" goto end
goto menu

:setup
echo.
if exist venv\Scripts\python.exe goto install
echo Creating the Python environment ...
py -3.11 -m venv venv
if not exist venv\Scripts\python.exe python -m venv venv
if not exist venv\Scripts\python.exe goto nopython
:install
call venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup finished. Now choose option 2 to check everything.
pause
goto menu

:nopython
echo.
echo Python 3.11 was not found. Install Python 3.11 from python.org and tick
echo "Add python.exe to PATH" in the installer, then run this option again.
pause
goto menu

:needenv
echo.
echo The environment is not ready yet. Choose option 1 first.
pause
goto menu

:check
if not exist venv\Scripts\activate.bat goto needenv
call venv\Scripts\activate.bat
python check_setup.py
pause
goto menu

:record
if not exist venv\Scripts\activate.bat goto needenv
call venv\Scripts\activate.bat
echo In the camera window: SPACE = start, S = skip, Q = quit.
python collect_data.py
pause
goto menu

:train
if not exist venv\Scripts\activate.bat goto needenv
call venv\Scripts\activate.bat
python train_model.py
python train_model.py --split-by-session --cv 0
pause
goto menu

:app
if not exist venv\Scripts\activate.bat goto needenv
call venv\Scripts\activate.bat
python -m streamlit run app.py
pause
goto menu

:tests
if not exist venv\Scripts\activate.bat goto needenv
call venv\Scripts\activate.bat
python -m unittest discover -s tests -t .
pause
goto menu

:onesign
if not exist venv\Scripts\activate.bat goto needenv
call venv\Scripts\activate.bat
echo.
echo Type the sign name exactly as in assets\sign_guide.csv. A new name is added to the sheet.
echo Old recordings of this sign are replaced. Simple names without special characters work best.
set "gname="
set /p gname=Sign name, for example Thank You: 
if "%gname%"=="" goto menu
python collect_data.py --redo "%gname%"
echo.
echo Now choose option 4 to train the model again.
pause
goto menu

:delsign
if not exist venv\Scripts\activate.bat goto needenv
call venv\Scripts\activate.bat
echo.
echo This removes only the recorded samples. To remove the sign itself, delete its row in assets\sign_guide.csv.
set "gname="
set /p gname=Sign name to delete, for example Thank You: 
if "%gname%"=="" goto menu
python collect_data.py --delete "%gname%"
echo.
echo Now choose option 4 to train the model again.
pause
goto menu

:end
endlocal
