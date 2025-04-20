Write-Host "Setting up FarmaBot..." -ForegroundColor Green

# Check if .env file exists
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.template") {
        Write-Host "Creating .env file from template..." -ForegroundColor Yellow
        Copy-Item ".env.template" ".env"
        Write-Host "Please edit the .env file to add your API keys" -ForegroundColor Yellow
    } else {
        Write-Host "Error: .env.template not found" -ForegroundColor Red
        exit 1
    }
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ./venv/Scripts/Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Verify database connection
Write-Host "Checking database connection..." -ForegroundColor Yellow
$dbCheck = Read-Host "Do you want to check the database connection? (y/n)"
if ($dbCheck -eq 'y') {
    if (Test-Path "scripts/test_db.py") {
        python scripts/test_db.py
    } else {
        Write-Host "Database test script not found" -ForegroundColor Red
    }
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "To start FarmaBot, run: python main.py" -ForegroundColor Cyan
Write-Host "Remember to edit your .env file with your API keys and database connection string" -ForegroundColor Yellow 