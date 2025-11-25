#!/usr/bin/env python3
"""
Script para executar a Calculadora Comercial sem problemas de firewall.
Este script configura o Streamlit para rodar em localhost com configurações seguras.
"""

import subprocess
import sys
import os

def main():
    print("🚀 Iniciando Calculadora Comercial...")
    print("📋 Configurações de segurança aplicadas para evitar problemas de firewall")
    print("=" * 60)
    
    # Configurações para evitar problemas de firewall
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.address", "localhost",
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    try:
        print("🌐 Aplicação rodando em: http://localhost:8501")
        print("⚠️  Se o navegador não abrir automaticamente, copie o link acima")
        print("🛑 Para parar a aplicação, pressione Ctrl+C")
        print("=" * 60)
        
        # Executa o Streamlit
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Aplicação encerrada pelo usuário")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar a aplicação: {e}")
        print("💡 Verifique se o Streamlit está instalado: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main()
