$ErrorActionPreference = "Stop"

$packageRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$siteRoot = Get-ChildItem -LiteralPath $packageRoot -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName "index.html") -PathType Leaf } |
    Select-Object -First 1 -ExpandProperty FullName

if ([string]::IsNullOrWhiteSpace($siteRoot)) {
    Write-Host "[ERROR] Website files are missing. Please extract the complete package again." -ForegroundColor Red
    exit 1
}

$mimeTypes = @{
    ".html" = "text/html; charset=utf-8"
    ".js" = "text/javascript; charset=utf-8"
    ".css" = "text/css; charset=utf-8"
    ".json" = "application/json; charset=utf-8"
    ".txt" = "text/plain; charset=utf-8"
    ".png" = "image/png"
    ".jpg" = "image/jpeg"
    ".jpeg" = "image/jpeg"
    ".gif" = "image/gif"
    ".svg" = "image/svg+xml"
    ".ico" = "image/x-icon"
    ".wasm" = "application/wasm"
    ".moc3" = "application/octet-stream"
}

$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)

try {
    $listener.Start()
    $port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    $address = "http://127.0.0.1:$port/"

    Clear-Host
    Write-Host "===============================================" -ForegroundColor Cyan
    Write-Host "  Yukino Live2D is running" -ForegroundColor Cyan
    Write-Host "===============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Your browser will open automatically: $address"
    Write-Host "Keep this window open. Close it to stop the website." -ForegroundColor Yellow
    Write-Host ""

    Start-Process $address

    while ($true) {
        $client = $listener.AcceptTcpClient()
        try {
            $client.ReceiveTimeout = 5000
            $client.SendTimeout = 5000
            $stream = $client.GetStream()
            $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::ASCII, $false, 1024, $true)
            $requestLine = $reader.ReadLine()

            if ([string]::IsNullOrWhiteSpace($requestLine)) {
                continue
            }

            do {
                $headerLine = $reader.ReadLine()
            } while (-not [string]::IsNullOrEmpty($headerLine))

            $parts = $requestLine.Split(" ")
            if ($parts.Length -lt 2 -or ($parts[0] -ne "GET" -and $parts[0] -ne "HEAD")) {
                $body = [System.Text.Encoding]::UTF8.GetBytes("Method not allowed")
                $response = "HTTP/1.1 405 Method Not Allowed`r`nContent-Type: text/plain; charset=utf-8`r`nContent-Length: $($body.Length)`r`nConnection: close`r`n`r`n"
                $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($response)
                $stream.Write($headerBytes, 0, $headerBytes.Length)
                if ($parts[0] -ne "HEAD") { $stream.Write($body, 0, $body.Length) }
                continue
            }

            $urlPath = ($parts[1] -split "\?", 2)[0]
            try {
                $decodedPath = [System.Uri]::UnescapeDataString($urlPath).TrimStart([char]"/").Replace([char]"/", [System.IO.Path]::DirectorySeparatorChar)
            } catch {
                $decodedPath = ""
            }

            if ([string]::IsNullOrWhiteSpace($decodedPath)) {
                $decodedPath = "index.html"
            }

            $filePath = [System.IO.Path]::GetFullPath((Join-Path $siteRoot $decodedPath))
            $insideSite = $filePath.StartsWith($siteRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or $filePath.Equals($siteRoot, [System.StringComparison]::OrdinalIgnoreCase)

            if ($insideSite -and (Test-Path $filePath -PathType Container)) {
                $filePath = Join-Path $filePath "index.html"
            }

            if (-not $insideSite -or -not (Test-Path $filePath -PathType Leaf)) {
                $body = [System.Text.Encoding]::UTF8.GetBytes("File not found")
                $response = "HTTP/1.1 404 Not Found`r`nContent-Type: text/plain; charset=utf-8`r`nContent-Length: $($body.Length)`r`nConnection: close`r`n`r`n"
                $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($response)
                $stream.Write($headerBytes, 0, $headerBytes.Length)
                if ($parts[0] -ne "HEAD") { $stream.Write($body, 0, $body.Length) }
                continue
            }

            $body = [System.IO.File]::ReadAllBytes($filePath)
            $extension = [System.IO.Path]::GetExtension($filePath).ToLowerInvariant()
            $contentType = if ($mimeTypes.ContainsKey($extension)) { $mimeTypes[$extension] } else { "application/octet-stream" }
            $response = "HTTP/1.1 200 OK`r`nContent-Type: $contentType`r`nContent-Length: $($body.Length)`r`nCache-Control: no-cache`r`nX-Content-Type-Options: nosniff`r`nConnection: close`r`n`r`n"
            $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($response)
            $stream.Write($headerBytes, 0, $headerBytes.Length)
            if ($parts[0] -ne "HEAD") { $stream.Write($body, 0, $body.Length) }
        } catch {
            # A failed browser request must not stop the local server.
        } finally {
            if ($null -ne $client) { $client.Close() }
        }
    }
} catch {
    Write-Host "[ERROR] Could not start the local website: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    $listener.Stop()
}
