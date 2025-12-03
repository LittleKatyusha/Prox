@echo off
echo ============================================================
echo Kiru Proxy - Production Startup
echo ============================================================
echo.
echo Starting Python server with integrated Next.js landing page...
echo.
echo Services:
echo - Python API Server: http://localhost:8741
echo - Landing Page: http://localhost:8741/
echo - Admin Dashboard: http://localhost:8741/admin/usage
echo - API Endpoint: http://localhost:8741/v1/chat/completions
echo.
echo ============================================================
echo.

cd /d "%~dp0"
python server.py

pause