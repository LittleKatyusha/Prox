# Cloudflare Tunnel Setup Guide

## Arsitektur
Proyek ini menggunakan 2 service terpisah:
- **Python Server** (Port 8741) - Backend API untuk AI models
- **Next.js** (Port 3000) - Frontend landing page

## Option 1: Dual Domain Setup (Recommended)

Menggunakan 2 subdomain terpisah untuk frontend dan backend.

### Configuration
File: `cloudflared-config.yml`

```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: /path/to/credentials.json

ingress:
  - hostname: models.yourdomain.com
    service: http://localhost:3000
  - hostname: api.yourdomain.com
    service: http://localhost:8741
  - service: http_status:404
```

### DNS Setup
1. `models.yourdomain.com` → CNAME → `YOUR_TUNNEL_ID.cfargotunnel.com`
2. `api.yourdomain.com` → CNAME → `YOUR_TUNNEL_ID.cfargotunnel.com`

### Update Next.js API Route
Edit `model-landing/app/api/models/route.ts`:
```typescript
const response = await fetch('https://api.yourdomain.com/api/models/grouped', {
  cache: 'no-store',
});
```

---

## Option 2: Single Domain with Path-Based Routing

Menggunakan satu domain dengan routing berbasis path.

### Configuration
```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: /path/to/credentials.json

ingress:
  - hostname: yourdomain.com
    path: /api/*
    service: http://localhost:8741
  - hostname: yourdomain.com
    service: http://localhost:3000
  - service: http_status:404
```

### DNS Setup
1. `yourdomain.com` → CNAME → `YOUR_TUNNEL_ID.cfargotunnel.com`

### Update Next.js API Route
Edit `model-landing/app/api/models/route.ts`:
```typescript
const response = await fetch('https://yourdomain.com/api/models/grouped', {
  cache: 'no-store',
});
```

---

## Option 3: Single Port Setup (Advanced)

Integrasikan Next.js ke dalam Python server sehingga hanya butuh 1 port.

### Steps:
1. Build Next.js:
```bash
cd model-landing
npm run build
npm run export  # or next export
```

2. Update Python server untuk serve static files Next.js

3. Cloudflared config:
```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: /path/to/credentials.json

ingress:
  - hostname: yourdomain.com
    service: http://localhost:8741
  - service: http_status:404
```

---

## Installation Steps

### 1. Install Cloudflared
**Windows:**
```powershell
winget install cloudflare.cloudflared
```

**Linux/Mac:**
```bash
# Download from https://github.com/cloudflare/cloudflared/releases
```

### 2. Login to Cloudflare
```bash
cloudflared tunnel login
```

### 3. Create Tunnel
```bash
cloudflared tunnel create kiru-proxy
```
Ini akan generate `YOUR_TUNNEL_ID` dan credentials file.

### 4. Update Config File
Edit `cloudflared-config.yml` dan ganti:
- `YOUR_TUNNEL_ID_HERE` dengan tunnel ID Anda
- `/path/to/YOUR_TUNNEL_CREDENTIALS.json` dengan path credentials file

### 5. Configure DNS
Sesuaikan dengan option yang dipilih (lihat di atas)

### 6. Start Tunnel
```bash
cloudflared tunnel --config cloudflared-config.yml run
```

---

## Running Services

### Start All Services:

**Terminal 1 - Python Server:**
```bash
python server.py
```

**Terminal 2 - Next.js:**
```bash
cd model-landing
npm run dev
```

**Terminal 3 - Cloudflared:**
```bash
cloudflared tunnel --config cloudflared-config.yml run
```

---

## Production Deployment

### Build Next.js for Production:
```bash
cd model-landing
npm run build
npm start  # Port 3000
```

### Run as Service (Windows):
Create `start-all.bat`:
```batch
@echo off
start "Python Server" cmd /k python server.py
start "Next.js" cmd /k "cd model-landing && npm start"
start "Cloudflared" cmd /k cloudflared tunnel --config cloudflared-config.yml run
```

### Run as Service (Linux/Mac):
Create systemd services or use PM2:
```bash
# Install PM2
npm install -g pm2

# Start Python server
pm2 start server.py --name api-server --interpreter python3

# Start Next.js
cd model-landing
pm2 start npm --name landing-page -- start

# Start Cloudflared
pm2 start cloudflared -- tunnel --config ../cloudflared-config.yml run
```

---

## Troubleshooting

### Issue: Next.js can't reach Python server
- Make sure Python server is running on port 8741
- Check firewall settings
- Verify API route URL in `model-landing/app/api/models/route.ts`

### Issue: Cloudflared tunnel not connecting
- Verify tunnel credentials file path
- Check tunnel ID is correct
- Ensure DNS records are properly configured
- Wait a few minutes for DNS propagation

### Issue: 502 Bad Gateway
- Ensure both services are running
- Check port numbers match configuration
- Verify service URLs in cloudflared config

---

## Environment Variables

Create `.env` file for Python server (already configured):
```
V1_URL=your_v1_url
V1_TOKEN=your_v1_token
...
MASTER_KEY=your_master_key
```

---

## Security Recommendations

1. Use HTTPS only in production
2. Keep credentials file secure
3. Set up proper CORS if using different domains
4. Use environment variables for sensitive data
5. Enable Cloudflare Access for admin routes
6. Rate limit public endpoints