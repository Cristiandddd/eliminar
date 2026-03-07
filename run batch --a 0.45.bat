@echo off
echo Ejecutando 30 simulaciones del Minority Game...
echo.

for /l %%i in (1,1,30) do (
    echo [%%i/30] Ejecutando...
    minority.exe -n 1137 -m 9
    echo.
    timeout /t 1 /nobreak >nul
)

echo Completado!
echo Archivos generados:
dir betting_history*.json
pause