REM Install Python dependencies
python -m pip install -r requirements.txt

ECHO Starting Flask backend server on http://127.0.0.1:5000 in a new window...
start "Flask Backend" python backend/server.py

ECHO Starting frontend HTTP server on http://localhost:8000 in a new window...
start "Frontend Server" python -m http.server 8000 --directory frontend

ECHO Waiting 5 seconds for servers to start...
timeout /t 5 > NUL

ECHO Opening http://localhost:8000 in your default web browser...
start http://localhost:8000

ECHO.
ECHO IMPORTANT: If the site still cannot be reached, please check the two new terminal windows that opened.
ECHO Look for any error messages in those windows and copy them here.
ECHO Press any key to close this window . . .
pause
