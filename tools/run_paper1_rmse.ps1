param(
    [string[]]$Models = @(
        "models/nsm/Gear_I.nsm",
        "models/nsm/Gear_II.nsm"
    ),
    [string[]]$Methods = @(
        "planar",
        "subdiv_planar",
        "nagata",
        "tess_patch",
        "ours"
    ),
    [string]$OutDir = "output/paper1_rmse",
    [int]$Depth = 8,
    [int]$StartDepth = 1,
    [double]$Termination = 1e-3,
    [int]$Grid = 128,
    [int]$NumThreads = 1,
    [switch]$Force,
    [switch]$NoPlot,
    [double]$Narrowband = 0.0,
    [switch]$AutoBenchmarks,
    [ValidateSet("quick", "full", "hard")]
    [string]$BenchmarkProfile = "quick",
    [string]$BenchmarkModelsDir = "output/benchmarks/models",
    [switch]$RenderModelImages,
    [string]$ModelImagesDir = "",
    [string]$TexModelsSnippet = "",
    [string]$ModelsGalleryPng = "",
    [int]$ModelsTexCols = 3,
    [switch]$AnalyticPointEval,
    [int]$AnalyticPointsPerModel = 20000,
    [int]$AnalyticSeed = 20260304,
    [string]$AnalyticSeeds = "",
    [string]$AnalyticMetadataCsv = "",
    [double]$AnalyticGradStep = 0.0
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$pyScript = Join-Path $repoRoot "pytools/paper1_rmse_pipeline.py"

if (!(Test-Path $pyScript)) {
    Write-Error "Python script not found: $pyScript"
    exit 1
}

$effectiveModels = $Models
if ($AutoBenchmarks -and -not $PSBoundParameters.ContainsKey("Models")) {
    # When auto benchmarks are enabled and user did not provide explicit models,
    # run only on generated benchmark suite.
    $effectiveModels = @()
}

$argsList = @($pyScript)
if ($effectiveModels.Count -gt 0) {
    $argsList += "--models"
    $argsList += $effectiveModels
}
$argsList += @(
    "--out_dir", $OutDir,
    "--methods"
)
$argsList += $Methods
$argsList += @(
    "--depth", "$Depth",
    "--start_depth", "$StartDepth",
    "--termination", "$Termination",
    "--grid", "$Grid",
    "--num_threads", "$NumThreads"
)

if ($Force) { $argsList += "--force" }
if ($NoPlot) { $argsList += "--no_plot" }
if ($Narrowband -gt 0) {
    $argsList += @("--narrowband", "$Narrowband")
}
if ($AutoBenchmarks) {
    $argsList += @(
        "--auto_benchmarks",
        "--benchmark_profile", $BenchmarkProfile,
        "--benchmark_models_dir", $BenchmarkModelsDir
    )
}
if ($AnalyticPointEval) {
    $argsList += @(
        "--analytic_point_eval",
        "--analytic_points_per_model", "$AnalyticPointsPerModel",
        "--analytic_seed", "$AnalyticSeed"
    )
    $parsedSeeds = @()
    if ($AnalyticSeeds -ne "") {
        $tokens = ($AnalyticSeeds -split "[,; ]+") | Where-Object { $_ -ne "" }
        foreach ($tok in $tokens) {
            $tmp = 0
            if ([int]::TryParse($tok, [ref]$tmp)) {
                $parsedSeeds += "$tmp"
            } else {
                Write-Error "Invalid AnalyticSeeds token: $tok"
                exit 2
            }
        }
    }
    if ($parsedSeeds.Count -gt 0) {
        $argsList += "--analytic_seeds"
        foreach ($s in $parsedSeeds) {
            $argsList += "$s"
        }
    }
    if ($AnalyticMetadataCsv -ne "") {
        $argsList += @("--analytic_metadata_csv", $AnalyticMetadataCsv)
    }
    if ($AnalyticGradStep -gt 0) {
        $argsList += @("--analytic_grad_step", "$AnalyticGradStep")
    }
}
if ($RenderModelImages) {
    $argsList += @(
        "--render_model_images",
        "--models_tex_cols", "$ModelsTexCols"
    )
    if ($ModelImagesDir -ne "") {
        $argsList += @("--model_images_dir", $ModelImagesDir)
    }
    if ($TexModelsSnippet -ne "") {
        $argsList += @("--tex_models_snippet", $TexModelsSnippet)
    }
    if ($ModelsGalleryPng -ne "") {
        $argsList += @("--models_gallery_png", $ModelsGalleryPng)
    }
}

Write-Host "[paper1] repo: $repoRoot"
Write-Host "[paper1] run : python $($argsList -join ' ')"

Push-Location $repoRoot
try {
    python @argsList
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
