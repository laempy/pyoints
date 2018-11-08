"%PYTHON%" -m pip install cylinder-fitting plyfile laspy -vvv
"%PYTHON%" -m pip install . --no-deps --ignore-installed -vvv
if errorlevel 1 exit 1