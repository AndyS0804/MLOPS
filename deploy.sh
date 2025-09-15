#!/bin/bash
set -e

IMAGE="andys0804/tp1:latest"
CONTAINER_NAME="my_app"

echo "Pulling latest image..."
docker pull $IMAGE

echo "Stopping old container..."
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

echo "Starting new container..."
docker run -d --name $CONTAINER_NAME -p 5242:5242 $IMAGE

echo "Deployment complete!"
