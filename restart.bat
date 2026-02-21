@echo off
cd /d D:\ds-project
echo [1/2] Stopping all containers...
docker compose -f docker-compose.dev.yml down
echo [2/2] Building and starting all containers...
docker compose -f docker-compose.dev.yml up -d --build
echo Done.
docker compose -f docker-compose.dev.yml ps
