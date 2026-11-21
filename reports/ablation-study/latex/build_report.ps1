param([string[]]$AttemptPaths)
$ErrorActionPreference = 'Stop'
$reportRoot = $PSScriptRoot
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $reportRoot '../../..')).Path
$pythonExe = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { $pythonExe = 'python' }
$summaryTool = Join-Path $repoRoot 'reports/ablation-study/tools/summarize_results.py'
if (-not $AttemptPaths) { $AttemptPaths = @(Join-Path $repoRoot 'artifacts/ablation-study') }
& $pythonExe -X utf8 $summaryTool @AttemptPaths --output (Join-Path $repoRoot 'reports/ablation-study/RESULTS.md')
if ($LASTEXITCODE -ne 0) { throw 'Artifact validation/summary failed.' }
& $pythonExe -X utf8 (Join-Path $reportRoot 'refresh_tables.py')
if ($LASTEXITCODE -ne 0) { throw 'LaTeX table refresh failed.' }
$buildRoot = Join-Path $repoRoot 'tmp/pdfs/ablation-study-build'
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
Push-Location -LiteralPath $reportRoot
try {
    foreach ($pass in 1..2) {
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory $buildRoot main.tex
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed on pass $pass. See $buildRoot/main.log" }
    }
    Copy-Item -LiteralPath (Join-Path $buildRoot 'main.pdf') -Destination (Join-Path $reportRoot 'ablation_study.pdf') -Force
} finally { Pop-Location }
Write-Output (Join-Path $reportRoot 'ablation_study.pdf')
