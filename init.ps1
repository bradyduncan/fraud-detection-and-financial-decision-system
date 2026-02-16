Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$venvDir = ".venv"

function Get-PythonCmd {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "Python not found. Install Python 3 and ensure it's on PATH."
}

$pythonCmd = Get-PythonCmd

if (-not (Test-Path $venvDir)) {
    & $pythonCmd -m venv $venvDir
}

$pip = Join-Path $venvDir "Scripts\pip.exe"
if (-not (Test-Path $pip)) {
    throw "pip not found at $pip. Virtual environment creation may have failed."
}

& $pip install --upgrade pip
& $pip install -r requirements.txt
