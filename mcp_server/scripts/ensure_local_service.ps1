[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Neo4jHome = '',
    [int]$McpPort = 8005,
    [string]$RemoteTarget = '',
    [int]$RemotePort = 18004,
    [string]$PublicSseUrl = '',
    [string]$TunnelTaskName = '',
    [int]$StartupGraceSeconds = 20
)

$ErrorActionPreference = 'Stop'
$bootstrapLog = Join-Path $PSScriptRoot '..\..\logs\local-service-watchdog.log'

trap {
    try {
        "$(Get-Date -Format o) ERROR $($_.Exception.Message)" |
            Out-File -FilePath $bootstrapLog -Append -Encoding utf8
    }
    catch {
        # The task scheduler will still record the non-zero exit code.
    }
    exit 1
}

Add-Type -AssemblyName System.Net.Http

if (-not $Neo4jHome) {
    $Neo4jHome = Join-Path (Split-Path $RepoRoot -Parent) 'neo4j'
}

$logDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$watchdogLog = Join-Path $logDir 'local-service-watchdog.log'

function Test-TcpPort {
    param(
        [string]$HostName,
        [int]$Port,
        [int]$TimeoutMilliseconds = 1500
    )

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connect = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $connect.AsyncWaitHandle.WaitOne($TimeoutMilliseconds)) {
            return $false
        }
        $client.EndConnect($connect)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Test-PublicSse {
    param([string]$Url)

    if (-not $Url) {
        return $false
    }

    $handler = [System.Net.Http.HttpClientHandler]::new()
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(10)
    try {
        $response = $client.GetAsync($Url).GetAwaiter().GetResult()
        return [int]$response.StatusCode -in 200, 401
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
        $handler.Dispose()
    }
}

function Test-RemoteTunnel {
    param(
        [string]$SshPath,
        [string]$Target,
        [int]$Port
    )

    if (-not $Target -or -not (Test-Path $SshPath)) {
        return $false
    }

    try {
        $listeners = & $SshPath `
            '-T' `
            '-o' 'BatchMode=yes' `
            '-o' 'ConnectTimeout=10' `
            $Target `
            'ss -ltn' 2>$null
        return $LASTEXITCODE -eq 0 -and $listeners -match "127\.0\.0\.1:$Port"
    }
    catch {
        return $false
    }
}

function Start-LoggedProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory
    )

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdout = Join-Path $logDir "$Name-out-$stamp.log"
    $stderr = Join-Path $logDir "$Name-err-$stamp.log"

    Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden | Out-Null
}

$neo4jHealthy = Test-TcpPort -HostName '127.0.0.1' -Port 7687
if (-not $neo4jHealthy) {
    $neo4jCommand = Join-Path $Neo4jHome 'bin\neo4j.bat'
    if (-not (Test-Path $neo4jCommand)) {
        throw "Neo4j executable not found: $neo4jCommand"
    }

    Start-LoggedProcess `
        -Name 'neo4j-watchdog' `
        -FilePath $neo4jCommand `
        -ArgumentList @('console') `
        -WorkingDirectory $Neo4jHome

    Start-Sleep -Seconds $StartupGraceSeconds
    $neo4jHealthy = Test-TcpPort -HostName '127.0.0.1' -Port 7687
}

if (-not $neo4jHealthy) {
    throw 'Neo4j is not listening on 127.0.0.1:7687; Graphiti was not started.'
}

$mcpHealthy = Test-TcpPort -HostName '127.0.0.1' -Port $McpPort
if (-not $mcpHealthy) {
    $python = Join-Path $RepoRoot 'mcp_server\.venv\Scripts\python.exe'
    $mcpRoot = Join-Path $RepoRoot 'mcp_server'
    if (-not (Test-Path $python)) {
        throw "Graphiti Python executable not found: $python"
    }

    Start-LoggedProcess `
        -Name 'mcp-watchdog' `
        -FilePath $python `
        -ArgumentList @('main.py', '--transport', 'sse', '--host', '127.0.0.1', '--port', "$McpPort") `
        -WorkingDirectory $mcpRoot

    Start-Sleep -Seconds $StartupGraceSeconds
    $mcpHealthy = Test-TcpPort -HostName '127.0.0.1' -Port $McpPort
}

if (-not $mcpHealthy) {
    throw "Graphiti MCP is not listening on 127.0.0.1:$McpPort; the tunnel was not started."
}

$publicHealthy = Test-PublicSse -Url $PublicSseUrl
$ssh = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
$remoteTunnelHealthy = Test-RemoteTunnel -SshPath $ssh -Target $RemoteTarget -Port $RemotePort
if (-not $publicHealthy -and -not $remoteTunnelHealthy -and $RemoteTarget) {
    if (-not (Test-Path $ssh)) {
        throw "OpenSSH executable not found: $ssh"
    }

    $tunnelTask = if ($TunnelTaskName) {
        Get-ScheduledTask -TaskName $TunnelTaskName -ErrorAction SilentlyContinue
    }

    if ($tunnelTask) {
        Stop-ScheduledTask -TaskName $TunnelTaskName -ErrorAction SilentlyContinue
        Start-ScheduledTask -TaskName $TunnelTaskName
    }
    else {
        Start-LoggedProcess `
            -Name 'ssh-tunnel-watchdog' `
            -FilePath $ssh `
            -ArgumentList @(
                '-N',
                '-T',
                '-o', 'BatchMode=yes',
                '-o', 'TCPKeepAlive=yes',
                '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=2',
                '-o', 'ExitOnForwardFailure=yes',
                '-R', "127.0.0.1:${RemotePort}:127.0.0.1:${McpPort}",
                $RemoteTarget
            ) `
            -WorkingDirectory $RepoRoot
    }

    Start-Sleep -Seconds 8
    $publicHealthy = Test-PublicSse -Url $PublicSseUrl
    $remoteTunnelHealthy = Test-RemoteTunnel -SshPath $ssh -Target $RemoteTarget -Port $RemotePort
}

if ($RemoteTarget -and -not $publicHealthy -and -not $remoteTunnelHealthy) {
    throw "The public SSE endpoint and remote tunnel are both unavailable after recovery."
}

$status = [pscustomobject]@{
    Neo4jHealthy = $neo4jHealthy
    McpHealthy = $mcpHealthy
    PublicSseHealthy = $publicHealthy
    RemoteTunnelHealthy = $remoteTunnelHealthy
    CheckedAt = Get-Date
}

"$($status.CheckedAt.ToString('o')) neo4j=$neo4jHealthy mcp=$mcpHealthy public_sse=$publicHealthy remote_tunnel=$remoteTunnelHealthy" |
    Out-File -FilePath $watchdogLog -Append -Encoding utf8
$status
exit 0
