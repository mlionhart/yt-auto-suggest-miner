# Copies results\*.jsonl and *.csv to Server 2 in a new dated folder. Never overwrites old backups.
# Run from the project folder in PowerShell:  .\backup.ps1
$ErrorActionPreference = "Stop"

$Remote = "devops@devops-worker"
$Base = "/mnt/storage/backups/yt-auto-suggest-miner"
$Stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$Dest = "$Base/$Stamp"

ssh $Remote "mkdir -p '$Dest'"
if ($LASTEXITCODE -ne 0) { throw "Could not create $Dest on $Remote" }

scp (Get-ChildItem results\* -Include *.jsonl, *.csv).FullName "${Remote}:$Dest/"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

Write-Host "Copied to ${Remote}:$Dest"
ssh $Remote "ls -l '$Dest'"
