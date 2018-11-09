"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install . -vvv
if errorlevel 1 exit 1