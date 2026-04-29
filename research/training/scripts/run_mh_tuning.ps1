param(
    [string[]]$MhValues = @(),
    [string[]]$Detectors = @(),
    [string[]]$Granularities = @(),
    [int]$Attempts = 5,
    [int]$MaxWorkers = 1,
    [string]$LogLevel = "INFO",
    [switch]$Resume,
    [switch]$RebuildCache
)

# Sequential statistical sweep wrapper.
# Runs one discovered `mhX` subset at a time with the sweep-style tuner.

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$startTime = Get-Date
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$repoRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent | Split-Path -Parent
$dataSubsetsRoot = Join-Path $repoRoot "data-subsets"
$cacheRoot = Join-Path $repoRoot "artifacts\cache\statistical_tuning"
$resultsRoot = Join-Path $repoRoot "results\tuning\statistical"
$logsDir = Join-Path $repoRoot "tuning_logs"

if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$allMhDirs = Get-ChildItem -Path $dataSubsetsRoot -Directory | Where-Object { $_.Name -match '^mh\d+$' } |
    Sort-Object { [int]($_.Name -replace '^mh', '') }
$defaultResearchMh = @("mh5", "mh10", "mh15", "mh20", "mh25", "mh30")

if (-not $allMhDirs) {
    throw "No mhX directories found under $dataSubsetsRoot"
}

if ($MhValues.Count -gt 0) {
    $normalizedMhValues = $MhValues | ForEach-Object {
        if ($_ -match '^mh\d+$') {
            $_
        } elseif ($_ -match '^\d+$') {
            "mh$_"
        } else {
            throw "Unsupported mh value '$_'. Use values like mh28 or 28."
        }
    }
    $mhDirs = $allMhDirs | Where-Object { $_.Name -in $normalizedMhValues }
    $missingMhValues = $normalizedMhValues | Where-Object { $_ -notin $mhDirs.Name }
    if ($missingMhValues) {
        throw "Requested mh directories were not found: $($missingMhValues -join ', ')"
    }
} else {
    $mhDirs = $allMhDirs | Where-Object { $_.Name -in $defaultResearchMh }
    if (-not $mhDirs) {
        throw "None of the default research mh directories were found under $dataSubsetsRoot: $($defaultResearchMh -join ', ')"
    }
}

$results = @()

foreach ($mhDir in $mhDirs) {
    $mhName = $mhDir.Name
    $runLabel = "${mhName}_$timestamp"
    $outputRoot = Join-Path $resultsRoot $runLabel
    $logFile = Join-Path $logsDir "statistical_sweep_${runLabel}.log"

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Starting statistical sweep for $mhName" -ForegroundColor Cyan
    if ($Detectors.Count -gt 0) {
        Write-Host "Detectors: $($Detectors -join ', ')" -ForegroundColor Cyan
    }
    if ($Granularities.Count -gt 0) {
        Write-Host "Granularities: $($Granularities -join ', ')" -ForegroundColor Cyan
    }
    Write-Host "Output: $outputRoot" -ForegroundColor Cyan
    Write-Host "Log:    $logFile" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    $command = @(
        "python",
        (Join-Path $repoRoot "research\training\scripts\tune_statistical.py"),
        "--data-subsets-root", $dataSubsetsRoot,
        "--cache-root", $cacheRoot,
        "--output-root", $outputRoot,
        "--mh-values", $mhName,
        "--attempts", $Attempts.ToString(),
        "--max-workers", $MaxWorkers.ToString(),
        "--log-level", $LogLevel
    )

    if ($Detectors.Count -gt 0) {
        $command += "--detectors"
        $command += $Detectors
    }

    if ($Granularities.Count -gt 0) {
        $command += "--granularities"
        $command += $Granularities
    }

    if ($Resume.IsPresent) {
        $command += "--resume"
    }

    if ($RebuildCache.IsPresent) {
        $command += "--rebuild-cache"
    }

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $command[0] @($command[1..($command.Length - 1)]) 2>&1 | Tee-Object -FilePath $logFile
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $results += [PSCustomObject]@{
        MhLevel    = $mhName
        ExitCode   = $exitCode
        OutputRoot = $outputRoot
        LogFile    = $logFile
    }

    if ($exitCode -ne 0) {
        Write-Host "Sweep failed for $mhName" -ForegroundColor Red
    } else {
        Write-Host "Sweep complete for $mhName" -ForegroundColor Green
    }
}

$duration = (Get-Date) - $startTime
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "Sequential sweep complete in $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
$results | Format-Table -AutoSize
