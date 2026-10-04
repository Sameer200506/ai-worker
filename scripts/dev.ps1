$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PWD'; .\.venv\Scripts\python.exe -m uvicorn apps.sandbox.main:app --host 127.0.0.1 --port 3001"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PWD'; .\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload"
if (Test-Path "apps\web\package.json") {
  Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PWD\apps\web'; if (-not (Test-Path node_modules)) { npm install }; npm run dev"
}
Write-Host "Sandbox http://127.0.0.1:3001"
Write-Host "API      http://127.0.0.1:8000"
Write-Host "Web      http://127.0.0.1:3000"
