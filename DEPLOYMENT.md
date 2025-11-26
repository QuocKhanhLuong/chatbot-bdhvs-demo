# 🚀 Vercel Deployment Guide

## Frontend Deployment (Vercel)

### Quick Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/QuocKhanhLuong/chatbot-bdhvs-demo&root-directory=frontend)

### Manual Deployment

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel:**
   ```bash
   vercel login
   ```

3. **Deploy from frontend directory:**
   ```bash
   cd frontend
   vercel
   ```

4. **Set Environment Variables in Vercel Dashboard:**
   - Go to your project settings → Environment Variables
   - Add:
     | Variable | Value | Environment |
     |----------|-------|-------------|
     | `PYTHON_BACKEND_URL` | `https://your-backend-url.com` | Production |

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PYTHON_BACKEND_URL` | URL of your Python FastAPI backend | ✅ Yes |
| `NEXT_PUBLIC_APP_NAME` | App display name | ❌ No |

---

## Backend Deployment Options

The Python backend can be deployed on:

### Option 1: Railway (Recommended)

1. Create account at [railway.app](https://railway.app)
2. Create new project → Deploy from GitHub
3. Select the `backend` directory
4. Add environment variables:
   ```
   LLM_PROVIDER=megallm
   MEGALLM_API_KEY=your_key
   MEGALLM_BASE_URL=https://api.mega.ai/v1
   TAVILY_API_KEY=your_key
   ```
5. Railway will auto-detect the Dockerfile

### Option 2: Render

1. Create account at [render.com](https://render.com)
2. New → Web Service → Connect GitHub
3. Settings:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.server:app --host 0.0.0.0 --port $PORT`

### Option 3: Google Cloud Run

```bash
cd backend
gcloud run deploy ai-research-backend \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars "LLM_PROVIDER=megallm,MEGALLM_API_KEY=xxx"
```

### Option 4: AWS Lambda + API Gateway

Use [Mangum](https://mangum.io/) adapter for FastAPI:
```python
from mangum import Mangum
handler = Mangum(app)
```

---

## Post-Deployment Checklist

- [ ] Backend is deployed and accessible
- [ ] `PYTHON_BACKEND_URL` is set in Vercel
- [ ] Test chat functionality
- [ ] Test deep research feature
- [ ] Check CORS settings if cross-origin issues

---

## Troubleshooting

### CORS Errors
Make sure your backend has correct CORS settings:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### API Timeout on Vercel
Vercel serverless functions have a 10s timeout on hobby plan.
For long-running requests (deep research), consider:
1. Upgrade to Pro plan (60s timeout)
2. Use streaming responses
3. Implement background jobs

### Cold Start Issues
If the backend takes too long to start:
1. Use a smaller Docker image
2. Implement health checks
3. Consider using a always-on service (Railway, Render)
