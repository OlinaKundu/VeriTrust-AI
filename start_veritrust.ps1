# ==============================================================================
#  VERITRUST AI: ALL-IN-ONE LOCAL RUNTIME LAUNCHER
#  Starts FastAPI Backend (Port 8000) & Next.js Frontend (Port 3000) simultaneously
# ==============================================================================

$Host.UI.RawUI.WindowTitle = "VeriTrust AI: Multi-Modal Forensic Engine"
$RootPath = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "         VERITRUST AI: FORENSIC PLATFORM          " -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Target Workspace: $RootPath" -ForegroundColor DarkGray

# 1. Check Python and Node.js Prerequisites
Write-Host "`n[1/3] Checking Environment Prerequisites..." -ForegroundColor Green
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not found in PATH. Please install Python 3.10+." -ForegroundColor Red
    Pause
    Exit
}
if (!(Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Node.js/npm is not found in PATH. Please install Node.js 18+." -ForegroundColor Red
    Pause
    Exit
}
Write-Host "  -> Python & Node.js environment detected." -ForegroundColor DarkGray

# 2. Launch FastAPI Backend in a dedicated process
Write-Host "`n[2/3] Booting FastAPI Backend on http://localhost:8000..." -ForegroundColor Green
$BackendCmd = "cd '$RootPath\backend'; uvicorn app.main:app --reload --port 8000 --host 127.0.0.1"
$BackendProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '--- VERITRUST AI: BACKEND LOGS ---' -ForegroundColor Cyan; $BackendCmd" -PassThru

# Give backend a moment to initialize GPU kernels
Start-Sleep -Seconds 2

# 3. Launch Next.js Frontend
Write-Host "`n[3/3] Booting Next.js Cyberpunk Dashboard on http://localhost:3000..." -ForegroundColor Green
$FrontendCmd = "cd '$RootPath\frontend'; npm run dev"
$FrontendProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '--- VERITRUST AI: FRONTEND LOGS ---' -ForegroundColor Green; $FrontendCmd" -PassThru

# 4. Open Default Browser
Start-Sleep -Seconds 3
Write-Host "`n[SUCCESS] Both Microservices Active!" -ForegroundColor Yellow
Write-Host "  -> Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "  -> Dashboard:   http://localhost:3000" -ForegroundColor Green
Write-Host "`nOpening browser at http://localhost:3000..." -ForegroundColor White
Start-Process "http://localhost:3000"

Write-Host "`n[INFO] Close this window or press Ctrl+C to terminate both servers." -ForegroundColor DarkYellow

# Keep monitoring
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "`nTerminating VeriTrust microservices..." -ForegroundColor Red
    if ($BackendProcess -and !$BackendProcess.HasExited) { Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue }
    if ($FrontendProcess -and !$FrontendProcess.HasExited) { Stop-Process -Id $FrontendProcess.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "Shutdown complete." -ForegroundColor Green
}
