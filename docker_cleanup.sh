#!/bin/bash

echo "Starting Docker cleanup..."

echo "Changing to R2R docker directory..."
cd /home/e4user/R2R/docker

echo "Stopping all running containers..."
docker stop $(docker ps -aq)

echo "Removing all containers..."
docker rm $(docker ps -aq)

echo "Removing all images..."
docker rmi $(docker images -q) -f

echo "Removing all volumes..."
docker volume rm $(docker volume ls -q)

echo "Removing all networks (except default ones)..."
docker network prune -f

echo "Removing everything (containers, images, volumes, networks) in one command..."
docker system prune -a --volumes -f

echo "Docker cleanup complete!"

echo "Building everything up..."
docker compose -f compose.full.yaml --profile postgres up -d

echo "Build process complete!" 