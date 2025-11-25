@echo off
echo ========================================
echo    CALCULADORA COMERCIAL - MVP
echo ========================================
echo.
echo Configurando ambiente para evitar problemas de firewall...
echo.

REM Verifica se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado!
    echo Por favor, instale o Python primeiro.
    pause
    exit /b 1
)

REM Instala dependências se necessário
echo Verificando dependencias...
pip install -r requirements.txt

echo.
echo Iniciando aplicacao...
echo A aplicacao abrira em: http://localhost:8501
echo Para parar, pressione Ctrl+C
echo.

REM Executa a aplicação
python run_app.py

pause
