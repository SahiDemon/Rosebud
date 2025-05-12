@echo off
echo Starting FindFlix Film Recommendation App...

REM Check if Docker is running
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo Docker doesn't seem to be running. Please start Docker Desktop first.
    echo.
    pause
    exit /b
)

REM Check if .env file exists
if not exist .env (
    echo .env file not found. Creating from example...
    copy .env.example .env
    echo Please edit the .env file with your API keys before continuing.
    echo.
    pause
    exit /b
)

REM Run docker-compose up
echo Building and starting containers...
docker-compose up -d

REM Check if it started successfully
if %errorlevel% neq 0 (
    echo.
    echo Error starting Docker containers. Please check the error messages above.
    echo.
    pause
    exit /b
)

echo.
echo FindFlix is now running!
echo Open your browser and go to: http://localhost:8000
echo.
echo To stop FindFlix, run: docker-compose down
echo.
pause