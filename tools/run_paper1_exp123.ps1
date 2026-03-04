param(
    [string]$OutDir = "output/paper1_exp123",
    [string]$Seeds = "20260304,20260305,20260306,20260307,20260308",
    [string]$ConvergenceDepths = "7,8,9",
    [string]$ConvergenceGrids = "64,128,256,512",
    [int]$PointsPerModel = 20000,
    [int]$StartDepth = 1,
    [int]$BaseDepth = 8,
    [int]$BaseGrid = 64,
    [double]$BaseTermination = 1e-3,
    [double]$AblationTermination = 1e-2,
    [int]$NumThreads = 1,
    [switch]$Force,
    [switch]$Verbose,
    [switch]$SkipConvergence,
    [switch]$SkipHardcase,
    [switch]$SkipAblation
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$pyScript = Join-Path $repoRoot "pytools/paper1_exp123_pipeline.py"

if (!(Test-Path $pyScript)) {
    Write-Error "Python script not found: $pyScript"
    exit 1
}

$argsList = @(
    $pyScript,
    "--out_dir", $OutDir,
    "--seeds", $Seeds,
    "--convergence_depths", $ConvergenceDepths,
    "--convergence_grids", $ConvergenceGrids,
    "--points_per_model", "$PointsPerModel",
    "--start_depth", "$StartDepth",
    "--base_depth", "$BaseDepth",
    "--base_grid", "$BaseGrid",
    "--base_termination", "$BaseTermination",
    "--ablation_termination", "$AblationTermination",
    "--num_threads", "$NumThreads"
)

if ($Force) { $argsList += "--force" }
if ($Verbose) { $argsList += "--verbose" }
if ($SkipConvergence) { $argsList += "--skip_convergence" }
if ($SkipHardcase) { $argsList += "--skip_hardcase" }
if ($SkipAblation) { $argsList += "--skip_ablation" }

Write-Host "[exp123] repo: $repoRoot"
Write-Host "[exp123] run : python $($argsList -join ' ')"

Push-Location $repoRoot
try {
    python @argsList
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
