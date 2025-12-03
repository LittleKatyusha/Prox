# Panduan Deployment ke VPS

## Prasyarat

1. VPS dengan Ubuntu/Debian (atau OS Linux lainnya)
2. Python 3.8+ sudah terinstall
3. Node.js 18+ dan npm sudah terinstall
4. Access SSH ke VPS
5. Domain (opsional, bisa pakai IP)

## Langkah 1: Persiapan VPS

### 1.1 Update sistem
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Install dependencies
```bash
# Install Python dan pip
sudo apt install python3 python3-pip python3-venv -y

# Install Node.js (jika belum ada)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs -y

# Install PM2 untuk process management
sudo npm install -g pm2

# Install nginx (opsional, untuk reverse proxy)
sudo apt install nginx -y
```

## Langkah 2: Upload Project ke VPS

### Opsi A: Menggunakan Git (Recommended)
```bash
# Di VPS
cd /var/www  # atau directory pilihan Anda
git clone <repository-url> kiru-proxy
cd kiru-proxy
```

### Opsi B: Menggunakan SCP/SFTP
```bash
# Di local machine
scp -r /path/to/kiru-proxy user@vps-ip:/var/www/
```

### Opsi C: Menggunakan rsync (Exclude node_modules)
```bash
# Di local machine
rsync -avz --exclude 'node_modules' --exclude '.next' --exclude 'logs' \
  /path/to/kiru-proxy/ user@vps-ip:/var/www/kiru-proxy/
```

## Langkah 3: Setup Project

### 3.1 Masuk ke directory project
```bash
cd /var/www/kiru-proxy
```

### 3.2 Setup Python environment
```bash
# Buat virtual environment
python3 -m venv venv

# Aktivasi virtual environment
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn aiohttp python-dotenv prometheus-client
```

### 3.3 Build Next.js landing page
```bash
cd model-landing
npm install
npm run build
cd ..
```

### 3.4 Setup file .env
```bash
# Copy dan edit .env file
nano .env
```

Pastikan `.env` berisi konfigurasi yang benar:
```env
V1_URL=http://localhost:8081
V1_TOKEN=sk-your-v1-token

V2_URL=https://apic1.ohmycdn.com/v1
V2_TOKEN=sk-key1,sk-key2

V3_URL=https://apic1.ohmycdn.com/v1
V3_TOKEN=sk-key1,sk-key2

# ... dst untuk V4-V10

MASTER_KEY=your-secure-master-key
```

## Langkah 4: Setup PM2 untuk Auto-Start

### 4.1 Buat ecosystem file
File `ecosystem.config.js` sudah ada, pastikan konfigurasinya benar:

```javascript
module.exports = {
  apps: [{
    name: 'kiru-proxy',
    script: 'venv/bin/python',
    args: 'server.py',
    cwd: '/var/www/kiru-proxy',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production'
    }
  }]
}
```

### 4.2 Start aplikasi dengan PM2
```bash
# Start server
pm2 start ecosystem.config.js

# Save PM2 list
pm2 save

# Setup PM2 untuk auto-start saat VPS reboot
pm2 startup systemd
# Jalankan command yang diberikan PM2 (biasanya dimulai dengan sudo)
```

### 4.3 Monitor aplikasi
```bash
# Lihat status
pm2 status

# Lihat logs
pm2 logs kiru-proxy

# Restart jika perlu
pm2 restart kiru-proxy
```

## Langkah 5: Setup Nginx Reverse Proxy (Opsional tapi Recommended)

### 5.1 Buat konfigurasi Nginx
```bash
sudo nano /etc/nginx/sites-available/kiru-proxy
```

Isi dengan:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Ganti dengan domain Anda atau IP VPS

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8741;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Untuk streaming
        proxy_buffering off;
        proxy_cache off;
    }
}
```

### 5.2 Enable site
```bash
# Buat symlink
sudo ln -s /etc/nginx/sites-available/kiru-proxy /etc/nginx/sites-enabled/

# Test konfigurasi
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

### 5.3 Setup SSL dengan Let's Encrypt (Recommended)
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Dapatkan SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renew sudah setup otomatis oleh certbot
```

## Langkah 6: Setup Firewall

```bash
# Allow HTTP, HTTPS, dan SSH
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable
```

## Langkah 7: Monitoring & Maintenance

### Monitoring
```bash
# PM2 monitoring
pm2 monit

# Check logs
pm2 logs kiru-proxy --lines 100

# Check error logs
pm2 logs kiru-proxy --err --lines 50
```

### Update aplikasi
```bash
# Pull latest code
cd /var/www/kiru-proxy
git pull

# Rebuild Next.js (jika ada perubahan)
cd model-landing
npm run build
cd ..

# Restart dengan PM2
pm2 restart kiru-proxy
```

## Troubleshooting

### Server tidak jalan
```bash
# Check PM2 status
pm2 status

# Check logs
pm2 logs kiru-proxy

# Restart
pm2 restart kiru-proxy
```

### Port sudah digunakan
```bash
# Check port yang digunakan
sudo lsof -i :8741

# Kill process jika perlu
sudo kill -9 <PID>
```

### Nginx error
```bash
# Check nginx status
sudo systemctl status nginx

# Check error logs
sudo tail -f /var/log/nginx/error.log

# Test konfigurasi
sudo nginx -t
```

## URL Akses Setelah Deploy

- **Landing Page**: `http://your-domain.com/` atau `http://vps-ip/`
- **Admin Manager**: `http://your-domain.com/admin/manager`
- **Admin Usage**: `http://your-domain.com/admin/usage`
- **User Usage**: `http://your-domain.com/usage`
- **API Endpoint**: `http://your-domain.com/v1/chat/completions`
- **Models List**: `http://your-domain.com/v1/models`

## Keamanan Tambahan

1. **Ganti MASTER_KEY** di `.env` dengan key yang kuat
2. **Gunakan HTTPS** dengan SSL certificate
3. **Restrict admin access** dengan IP whitelist di Nginx jika perlu:
   ```nginx
   location /admin {
       allow your-office-ip;
       deny all;
       proxy_pass http://127.0.0.1:8741;
   }
   ```
4. **Backup regular** untuk `api_keys.json` dan `logs/`
5. **Monitor logs** secara berkala

## Backup & Restore

### Backup
```bash
# Backup data penting
tar -czf kiru-proxy-backup-$(date +%Y%m%d).tar.gz \
  api_keys.json \
  allowed_models.txt \
  .env \
  logs/
```

### Restore
```bash
# Extract backup
tar -xzf kiru-proxy-backup-YYYYMMDD.tar.gz
```

## Support

Jika ada masalah, check:
1. PM2 logs: `pm2 logs kiru-proxy`
2. Nginx logs: `sudo tail -f /var/log/nginx/error.log`
3. Application logs: `cat logs/server.log`