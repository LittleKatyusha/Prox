#!/bin/bash

# Kiru Proxy VPS Setup Script
# Run this script on your VPS after uploading the project

set -e  # Exit on error

echo "======================================"
echo "Kiru Proxy VPS Setup"
echo "======================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get current directory
PROJECT_DIR=$(pwd)

echo -e "${GREEN}Installing system dependencies...${NC}"
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nodejs npm nginx

echo -e "${GREEN}Installing PM2...${NC}"
sudo npm install -g pm2

echo -e "${GREEN}Setting up Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

echo -e "${GREEN}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install fastapi uvicorn aiohttp python-dotenv prometheus-client

echo -e "${GREEN}Building Next.js landing page...${NC}"
cd model-landing
npm install
npm run build
cd ..

echo -e "${GREEN}Creating logs directory...${NC}"
mkdir -p logs

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found!${NC}"
    echo -e "${YELLOW}Please create .env file with your configuration${NC}"
    echo -e "${YELLOW}You can copy .env.example if available${NC}"
fi

# Update ecosystem.config.js with current directory
echo -e "${GREEN}Updating ecosystem.config.js...${NC}"
sed -i "s|cwd: '/var/www/kiru-proxy'|cwd: '$PROJECT_DIR'|g" ecosystem.config.js

echo -e "${GREEN}Starting application with PM2...${NC}"
pm2 start ecosystem.config.js

echo -e "${GREEN}Saving PM2 configuration...${NC}"
pm2 save

echo -e "${GREEN}Setting up PM2 startup script...${NC}"
echo -e "${YELLOW}Please run the command that PM2 will show below:${NC}"
pm2 startup systemd

echo ""
echo -e "${GREEN}======================================"
echo -e "Setup Complete!"
echo -e "======================================${NC}"
echo ""
echo -e "Next steps:"
echo -e "1. Configure your .env file if not done yet"
echo -e "2. Run the PM2 startup command shown above"
echo -e "3. Setup Nginx reverse proxy (see DEPLOYMENT_GUIDE.md)"
echo -e "4. Setup SSL with Let's Encrypt (optional)"
echo ""
echo -e "Useful commands:"
echo -e "  pm2 status          - Check application status"
echo -e "  pm2 logs kiru-proxy - View logs"
echo -e "  pm2 restart kiru-proxy - Restart application"
echo -e "  pm2 stop kiru-proxy    - Stop application"
echo ""
echo -e "Application should be running on: http://localhost:8741"
echo -e "Access it via: http://your-vps-ip:8741"
echo ""