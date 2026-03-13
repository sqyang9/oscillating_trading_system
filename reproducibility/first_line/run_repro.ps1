$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
python -m pip install -r (Join-Path $root 'requirements.txt')
python (Join-Path $root 'run_repro.py')
