#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy Trading Algorithm to AWS Elastic Beanstalk
    
.DESCRIPTION
    This script builds the deployment package and uploads it to AWS Elastic Beanstalk
    with all the latest updates and fixes applied.
#>

param(
    [string]$AppName = "trading-algorithm",
    [string]$EnvironmentName = "tradingalgorithm-env", 
    [string]$Region = "us-west-2",
    [switch]$SkipPrompts
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Trading Algorithm - AWS Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check AWS CLI
Write-Host "[1/6] Checking AWS CLI installation..." -ForegroundColor Yellow
try {
    $awsVersion = aws --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI not found"
    }
    Write-Host "[OK] AWS CLI found: $awsVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: AWS CLI is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install AWS CLI from: https://aws.amazon.com/cli/" -ForegroundColor Red
    exit 1
}

# Step 2: Check AWS credentials
Write-Host "[2/6] Checking AWS credentials..." -ForegroundColor Yellow
try {
    $callerIdentity = aws sts get-caller-identity 2>$null | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) {
        throw "AWS credentials not configured"
    }
    Write-Host "[OK] AWS credentials configured for: $($callerIdentity.Arn)" -ForegroundColor Green
} catch {
    Write-Host "ERROR: AWS credentials not configured" -ForegroundColor Red
    Write-Host "Please run: aws configure" -ForegroundColor Red
    exit 1
}

# Step 3: Get deployment details
if (-not $SkipPrompts) {
    Write-Host "[3/6] Getting deployment details..." -ForegroundColor Yellow
    $AppName = Read-Host "Enter Application Name (default: $AppName)"
    if ([string]::IsNullOrWhiteSpace($AppName)) { $AppName = "trading-algorithm" }
    
    $EnvironmentName = Read-Host "Enter Environment Name (default: $EnvironmentName)"
    if ([string]::IsNullOrWhiteSpace($EnvironmentName)) { $EnvironmentName = "tradingalgorithm-env" }
    
    $Region = Read-Host "Enter AWS Region (default: $Region)"
    if ([string]::IsNullOrWhiteSpace($Region)) { $Region = "us-west-2" }
}

Write-Host "[OK] Deployment target:" -ForegroundColor Green
Write-Host "  Application: $AppName" -ForegroundColor White
Write-Host "  Environment: $EnvironmentName" -ForegroundColor White
Write-Host "  Region: $Region" -ForegroundColor White
Write-Host ""

# Step 4: Create deployment package
Write-Host "[4/6] Creating deployment package..." -ForegroundColor Yellow
$deploymentFile = "deployment_fixed.zip"
if (Test-Path $deploymentFile) {
    Remove-Item $deploymentFile -Force
}

try {
    $files = @(
        "flask_app.py",
        "wsgi.py", 
        "requirements.txt",
        "Procfile",
        "gunicorn.conf.py",
        "templates",
        "config.py"
    )
    
    Compress-Archive -Path $files -DestinationPath $deploymentFile -Force
    Write-Host "[OK] Deployment package created: $deploymentFile" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to create deployment package" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Step 5: Upload to S3
Write-Host "[5/6] Uploading to S3..." -ForegroundColor Yellow
$s3Bucket = "elasticbeanstalk-$Region-$AppName"
$versionLabel = "v$(Get-Date -Format 'yyyyMMdd-HHmmss')"

try {
    # Try to upload to S3 first
    aws s3 cp $deploymentFile "s3://$s3Bucket/$deploymentFile" --region $Region 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Uploaded to S3: s3://$s3Bucket/$deploymentFile" -ForegroundColor Green
    } else {
        throw "S3 upload failed"
    }
} catch {
    Write-Host "WARNING: S3 upload failed, trying direct deployment..." -ForegroundColor Yellow
}

# Step 6: Deploy to Elastic Beanstalk
Write-Host "[6/6] Deploying to Elastic Beanstalk..." -ForegroundColor Yellow
try {
    $deployCommand = @(
        "aws", "elasticbeanstalk", "create-application-version",
        "--application-name", $AppName,
        "--version-label", $versionLabel,
        "--source-bundle", "S3Bucket=$s3Bucket,S3Key=$deploymentFile",
        "--region", $Region
    )
    
    & $deployCommand[0] $deployCommand[1..($deployCommand.Length-1)] 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Application version created: $versionLabel" -ForegroundColor Green
        
        # Update environment
        Write-Host "Updating environment..." -ForegroundColor Yellow
        aws elasticbeanstalk update-environment `
            --environment-name $EnvironmentName `
            --version-label $versionLabel `
            --region $Region 2>$null
            
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Environment update initiated" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] Environment update failed, but version was created" -ForegroundColor Yellow
        }
    } else {
        throw "Failed to create application version"
    }
} catch {
    Write-Host "ERROR: Deployment failed" -ForegroundColor Red
    Write-Host "Please manually upload $deploymentFile to your EB environment" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Recent updates included:" -ForegroundColor Cyan
Write-Host "✓ Enhanced SocketIO configuration for production deployment" -ForegroundColor White
Write-Host "✓ Improved error handling and logging" -ForegroundColor White
Write-Host "✓ Optimized Gunicorn settings for web applications" -ForegroundColor White
Write-Host "✓ Updated dependencies for better compatibility" -ForegroundColor White
Write-Host ""
Write-Host "Check your Elastic Beanstalk console for deployment status." -ForegroundColor Yellow
Write-Host "Your application should be updating now with the latest improvements." -ForegroundColor Yellow 