Write-Host "Starting FarmaBot cleanup for GitHub repository..." -ForegroundColor Green

# Remove JSON order files
Write-Host "Removing order JSON files..." -ForegroundColor Yellow
Remove-Item -Path "orders_*.json" -Force -ErrorAction SilentlyContinue

# Remove unnecessary files
$filesToRemove = @(
    "FarmaBot_corregido.py",
    "SQL.ipynb",
    "farmabot.txt",
    "chatbot_logic.py",
    "interface.py",
    "database_operations.py",
    "logging_utils.py",
    "vector_store.py",
    "test_medicine_search.py",
    "0.7.1"
)

foreach ($file in $filesToRemove) {
    if (Test-Path $file) {
        Write-Host "Removing $file..." -ForegroundColor Yellow
        Remove-Item -Path $file -Force
    }
}

# Handle .env file safely
if (Test-Path ".env") {
    Write-Host "Found .env file with sensitive data" -ForegroundColor Yellow
    
    # Check if .env.template exists, create if not
    if (-not (Test-Path ".env.template")) {
        Write-Host "Creating .env.template file..." -ForegroundColor Yellow
        Copy-Item ".env" ".env.template"
        
        # Replace actual API keys with placeholders
        (Get-Content ".env.template") | 
            ForEach-Object { $_ -replace "OPENAI_API_KEY=.+", "OPENAI_API_KEY=your_openai_api_key_here" } |
            ForEach-Object { $_ -replace "GOOGLE_API_KEY=.+", "GOOGLE_API_KEY=your_google_api_key_here" } |
            Set-Content ".env.template"
    }
    
    Write-Host "Warning: .env contains sensitive API keys and should not be uploaded to GitHub" -ForegroundColor Red
    $response = Read-Host "Do you want to rename .env to .env.local to prevent accidental upload? (y/n)"
    if ($response -eq 'y') {
        Rename-Item -Path ".env" -NewName ".env.local" -Force
        Write-Host ".env renamed to .env.local" -ForegroundColor Green
    }
}

# Remove unnecessary directories
$dirsToRemove = @(
    ".ipynb_checkpoints",
    ".conda",
    ".gradio",
    ".venv",
    "venv",
    "__pycache__"
)

foreach ($dir in $dirsToRemove) {
    if (Test-Path $dir) {
        Write-Host "Removing $dir directory..." -ForegroundColor Yellow
        Remove-Item -Path $dir -Recurse -Force
    }
}

# Remove all __pycache__ directories recursively
Write-Host "Removing all __pycache__ directories..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | 
    ForEach-Object {
        Remove-Item -Path $_.FullName -Recurse -Force
    }

# Check if vector database directories should be deleted
$vectorDBs = @(
    "medicines_vectordb",
    "chroma_db_medicines"
)

foreach ($db in $vectorDBs) {
    if (Test-Path $db) {
        $response = Read-Host "Do you want to remove the $db directory? It can be regenerated later. (y/n)"
        if ($response -eq 'y') {
            Write-Host "Removing $db directory..." -ForegroundColor Yellow
            Remove-Item -Path $db -Recurse -Force
        }
    }
}

Write-Host "Cleanup complete! The project is ready for GitHub upload." -ForegroundColor Green 