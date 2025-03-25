#!/bin/bash

echo "Starting Rosebud Film Recommendation App..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker doesn't seem to be running. Please start Docker first."
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo ".env file not found. Creating from example..."
    cp .env.example .env
    echo "Please edit the .env file with your API keys before continuing."
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Run docker-compose up
echo "Building and starting containers..."
docker-compose up -d

# Check if it started successfully
if [ $? -ne 0 ]; then
    echo
    echo "Error starting Docker containers. Please check the error messages above."
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

echo
echo "Rosebud is now running!"
echo "Open your browser and go to: http://localhost:8000"
echo
echo "To stop Rosebud, run: docker-compose down"
echo
read -p "Press Enter to continue..."