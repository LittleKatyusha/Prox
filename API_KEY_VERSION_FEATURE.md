# Fitur Pembatasan Versi API Key

## Deskripsi
Fitur ini memungkinkan admin untuk membatasi akses API key ke versi model tertentu. Setiap API key dapat dikonfigurasi untuk hanya mengakses model dari versi tertentu (v1, v2, v3, dst) atau semua versi.

## Cara Menggunakan

### 1. Membuat API Key Baru dengan Versi
1. Buka halaman Admin Key Manager di `/admin/manager`
2. Isi form "Add New API Key":
   - **Key Name**: Nama untuk API key (contoh: "Production Key")
   - **API Key**: Masukkan key atau klik "Generate" untuk membuat otomatis
   - **Version Access**: Pilih versi yang diizinkan:
     - `All Versions`: Dapat mengakses semua model (v1, v2, v3, dst)
     - `V1 Only`: Hanya dapat mengakses model yang berakhiran `-v1`
     - `V2 Only`: Hanya dapat mengakses model yang berakhiran `-v2`
     - `V3 Only`: Hanya dapat mengakses model yang berakhiran `-v3`
     - Dan seterusnya hingga V10
3. Klik "Add Key"

### 2. Mengedit Versi API Key yang Ada
1. Pada tabel "Existing API Keys", klik tombol "Edit" pada key yang ingin diubah
2. Ubah nama dan/atau pilih versi baru dari dropdown
3. Klik "Save" untuk menyimpan perubahan

### 3. Cara Kerja Pembatasan
- Jika API key dibatasi ke versi tertentu (misalnya V2), maka:
  - ✅ Dapat mengakses: `claude-opus-4-5-20251101-v2`, `claude-sonnet-4-5-20250929-v2`
  - ❌ Tidak dapat mengakses: `claude-opus-4-5-20251101-v1`, `claude-opus-4-5-20251101-v3`
  
- Jika API key diset ke "All Versions":
  - ✅ Dapat mengakses semua model dari semua versi

### 4. Filter Daftar Model
Endpoint untuk mendapatkan daftar model sekarang otomatis difilter berdasarkan versi API key:

**GET `/v1/models`** (OpenAI Compatible)
- Memerlukan Authorization header dengan API key
- Hanya menampilkan model yang sesuai dengan versi API key
- Contoh: Key v2 hanya akan melihat model `-v2`

**GET `/api/models`** (Landing Page)
- Opsional Authorization header
- Tanpa auth: menampilkan semua model
- Dengan auth: hanya menampilkan model sesuai versi key

### 5. Error Response
Jika user mencoba mengakses model yang tidak diizinkan, akan menerima error:
```json
{
  "detail": "API key is restricted to version 'v2' models only. Cannot access 'claude-opus-4-5-20251101-v1'."
}
```

## Struktur Data

### api_keys.json
```json
{
  "keys": {
    "sk-xxxxx": {
      "name": "Production Key",
      "version": "v2"
    },
    "sk-yyyyy": {
      "name": "Development Key",
      "version": "all"
    }
  }
}
```

## Migrasi dari Format Lama
Sistem secara otomatis akan migrasi format lama (`key_names`) ke format baru (`keys`) dengan versi default "all".

## Contoh Penggunaan

### Skenario 1: Tim Development
- Buat API key dengan version "v1" untuk tim development
- Mereka hanya bisa test dengan model v1 yang lebih murah

### Skenario 2: Tim Production
- Buat API key dengan version "v2" atau "v3" untuk production
- Mereka hanya bisa akses model versi premium

### Skenario 3: Admin/Testing
- Buat API key dengan version "all"
- Dapat mengakses semua model untuk testing dan monitoring

## Endpoint yang Terpengaruh

### 1. `/v1/chat/completions` (POST)
- ✅ Validasi versi sebelum memproses request
- ❌ Reject jika model tidak sesuai versi key

### 2. `/v1/models` (GET)
- ✅ Filter daftar model berdasarkan versi key
- Memerlukan Authorization header

### 3. `/api/models` (GET)
- ✅ Filter daftar model jika Authorization header disediakan
- Tanpa auth: tampilkan semua model (untuk landing page publik)

### 4. `/api/models/grouped` (GET)
- ✅ Mendukung parameter `version_filter`
- Digunakan internal oleh `/api/models`

## Contoh Request

### Mendapatkan Daftar Model dengan Key V2
```bash
curl -H "Authorization: Bearer sk-your-v2-key" \
     http://localhost:8741/v1/models
```

Response hanya akan berisi model `-v2`:
```json
{
  "object": "list",
  "data": [
    {
      "id": "claude-opus-4-5-20251101-v2",
      "object": "model",
      "created": 1234567890,
      "owned_by": "norenaboi",
      "type": "chat"
    },
    {
      "id": "claude-sonnet-4-5-20250929-v2",
      "object": "model",
      "created": 1234567890,
      "owned_by": "norenaboi",
      "type": "chat"
    }
  ]
}
```

### Menggunakan Model dengan Key V2
```bash
curl -X POST http://localhost:8741/v1/chat/completions \
  -H "Authorization: Bearer sk-your-v2-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-5-20251101-v2",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```
✅ Berhasil - model sesuai dengan versi key

```bash
curl -X POST http://localhost:8741/v1/chat/completions \
  -H "Authorization: Bearer sk-your-v2-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-5-20251101-v1",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```
❌ Error 403 - model tidak sesuai dengan versi key