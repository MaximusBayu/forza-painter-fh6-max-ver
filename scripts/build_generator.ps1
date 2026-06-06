# Phase 6 — reproducible OpenCL-only build of the bundled geometry generator.
#
# Verified recipe (this machine: NVIDIA RTX 3050 Ti, OpenCL.dll present, Go 1.26.4
# installed to .toolchain\go). The generator is a GPU program that uses cgo, so a
# C compiler is required. We build OpenCL-ONLY to avoid the Vulkan SDK + glslc +
# shader chain (the upstream Vulkan backend is excluded via a //go:build vulkan
# tag added to internal/gpu/vulkan.go, with a !vulkan stub in vulkan_stub.go).
#
# REQUIREMENTS that must already exist (the one piece this machine still lacks):
#   - A cgo-compatible C compiler: mingw-w64 **gcc** (or gnu-style clang).
#     MSVC cl.exe does NOT work with cgo. Point -Gcc at gcc.exe, or put it on PATH.
#   - Go (auto-detected from .toolchain\go or PATH).
#   - OpenCL runtime (C:\Windows\System32\OpenCL.dll) — used to synthesize the
#     import library; the go-opencl binding vendors its own cl.h so no OpenCL SDK
#     headers are needed.
#
# Run from the repo root once a compiler is available:
#   powershell -File scripts/build_generator.ps1 -Gcc C:\path\to\mingw64\bin\gcc.exe

[CmdletBinding()]
param(
    [string]$Repo = "https://github.com/zjl88858/forza-painter-geometrize-gpu.git",
    [string]$Ref = "main",
    [string]$Src = "$PSScriptRoot\..\.build\geometrize-gpu",
    [string]$Gcc = "",
    [string]$GoExe = "$PSScriptRoot\..\.toolchain\go\bin\go.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$binTarget = Join-Path $repoRoot "bin\forza-painter-geometrize-go.exe"

# --- Resolve toolchain --------------------------------------------------------
if (-not (Test-Path $GoExe)) {
    $g = Get-Command go -ErrorAction SilentlyContinue
    if (-not $g) { throw "Go not found (looked at $GoExe and PATH)." }
    $GoExe = $g.Source
}
if (-not $Gcc) {
    $g = Get-Command gcc -ErrorAction SilentlyContinue
    if ($g) { $Gcc = $g.Source }
}
if (-not $Gcc -or -not (Test-Path $Gcc)) {
    throw "No cgo C compiler. Install mingw-w64 gcc and pass -Gcc <gcc.exe>. (MSVC cl.exe is NOT cgo-compatible.)"
}
$gccBin = Split-Path $Gcc
$dlltool = Join-Path $gccBin "dlltool.exe"
$gendef = Join-Path $gccBin "gendef.exe"

# --- Fetch upstream -----------------------------------------------------------
if (-not (Test-Path $Src)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Src) | Out-Null
    git clone $Repo $Src
}

# --- Exclude the Vulkan backend (OpenCL-only fork) ----------------------------
$vk = Join-Path $Src "internal\gpu\vulkan.go"
$firstLine = (Get-Content $vk -TotalCount 1)
if ($firstLine -notmatch "go:build vulkan") {
    $body = Get-Content $vk -Raw
    "//go:build vulkan`r`n// +build vulkan`r`n`r`n$body" | Set-Content -Path $vk -Encoding utf8
}
$stub = Join-Path $Src "internal\gpu\vulkan_stub.go"
if (-not (Test-Path $stub)) {
@'
//go:build !vulkan
// +build !vulkan

package gpu

import "fmt"

func newVulkanBackend(target, current []float32, maskData []uint8, width, height, maxCandidates, gridSize int) (Backend, error) {
	return nil, fmt.Errorf("vulkan backend not compiled in (build with -tags vulkan); use the opencl backend")
}
'@ | Set-Content -Path $stub -Encoding utf8
}

# --- Synthesize an OpenCL import library from the system DLL -------------------
$libDir = Join-Path $Src ".opencllib"
New-Item -ItemType Directory -Force -Path $libDir | Out-Null
$libOpenCL = Join-Path $libDir "libOpenCL.a"
if (-not (Test-Path $libOpenCL)) {
    $sysDll = "C:\Windows\System32\OpenCL.dll"
    if (-not (Test-Path $sysDll)) { throw "OpenCL.dll not found in System32 (no OpenCL runtime)." }
    Push-Location $libDir
    try {
        & $gendef $sysDll                     # -> OpenCL.def
        & $dlltool -d "OpenCL.def" -l "libOpenCL.a" -D "OpenCL.dll"
    } finally { Pop-Location }
    if (-not (Test-Path $libOpenCL)) { throw "Failed to synthesize libOpenCL.a" }
}

# --- Build (OpenCL-only; vulkan tag intentionally NOT set) --------------------
$env:CGO_ENABLED = "1"
$env:CC = $Gcc
$env:CGO_LDFLAGS = "-L$libDir"
$outExe = Join-Path $Src "forza-painter-geometrize-go.exe"
Push-Location $Src
try {
    & $GoExe build -o $outExe ./cmd/forza-painter-geometrize
    if ($LASTEXITCODE -ne 0) { throw "go build failed ($LASTEXITCODE)" }
} finally { Pop-Location }
if (-not (Test-Path $outExe)) { throw "build produced no binary" }

# --- Install (atomic, with backup) -------------------------------------------
if (Test-Path $binTarget) { Copy-Item $binTarget "$binTarget.bak" -Force }
Copy-Item $outExe $binTarget -Force
Write-Host "Installed -> $binTarget"
Write-Host "Verify offline (no game needed):"
Write-Host "  $binTarget <image.png> -settings <ini> -output out -preview prev.png"
