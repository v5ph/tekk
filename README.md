# An AI Copilot for Scientific and Financial Research

Tekk is a production-ready AutoML web platform that enables users to upload datasets, describe analysis goals in natural language, and automatically train machine learning models with AI-generated insights.

## 🚀 Features

- **Authentication**: Secure email/password auth with JWT sessions
- **Data Upload**: Drag-and-drop support for CSV, XLSX, JSON, TXT files
- **Natural Language Interface**: Describe ML tasks in plain English
- **AutoML Pipeline**: Automated classification, regression, clustering, forecasting
- **AI Reports**: GPT-4 powered summaries and insights
- **Subscription Billing**: 3-tier Stripe billing system
- **Modern UI**: Dark mode, responsive design, sci-fi minimalism

## 🏗️ Architecture

### Frontend (Next.js)
- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS + shadcn/ui components
- **State**: Zustand for auth and global state
- **Charts**: Plotly.js for visualizations
- **File Upload**: react-dropzone

### Backend (FastAPI)
- **API**: FastAPI with async/await
- **ML**: PyCaret for AutoML + scikit-learn
- **AI**: OpenAI GPT-4 for prompt analysis and reports
- **Database**: Supabase (PostgreSQL)
- **Auth**: JWT with bcrypt password hashing
- **Billing**: Stripe Checkout + Webhooks
- **Tasks**: Celery + Redis for async ML processing

### Infrastructure
- **Frontend**: Vercel deployment
- **Backend**: Fly.io/Railway (Docker)
- **Database**: Supabase (managed PostgreSQL)
- **Storage**: Supabase Storage for file uploads
- **Domain**: Cloudflare DNS (tekk.systems)

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- Redis (for Celery)
- Supabase account
- OpenAI API key
- Stripe account

## 🛠️ Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd tekk
```

### 2. Backend Setup
```bash
cd tekk-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp env.example .env

# Edit .env with your credentials
# - Supabase URL and keys
# - OpenAI API key
# - Stripe keys
# - Redis URL
```

### 3. Frontend Setup
```bash
cd tekk-frontend

# Install dependencies
npm install

# Copy environment file
cp env.example .env.local

# Edit .env.local with your credentials
```

### 4. Database Setup

Create these tables in Supabase:

```sql
-- Users table
CREATE TABLE users (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  email VARCHAR UNIQUE NOT NULL,
  password_hash VARCHAR NOT NULL,
  full_name VARCHAR,
  subscription_plan VARCHAR DEFAULT 'free',
  projects_used INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Projects table
CREATE TABLE projects (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  name VARCHAR NOT NULL,
  description TEXT,
  status VARCHAR DEFAULT 'uploading',
  task_type VARCHAR,
  target_column VARCHAR,
  dataset_rows INTEGER,
  dataset_columns INTEGER,
  model_metrics JSONB,
  ai_summary TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view own data" ON users FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update own data" ON users FOR UPDATE USING (auth.uid() = id);
CREATE POLICY "Users can view own projects" ON projects FOR ALL USING (auth.uid() = user_id);
```

### 5. Start Services

**Backend:**
```bash
cd tekk-backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd tekk-frontend
npm run dev
```

**Redis (for Celery):**
```bash
redis-server
```

**Celery Worker:**
```bash
cd tekk-backend
celery -A app.celery worker --loglevel=info
```

## 🔧 Configuration

### Subscription Plans

| Plan | Price | Projects/Month | Row Limit | Features |
|------|-------|----------------|-----------|----------|
| Free | $0 | 3 | 1,000 | Basic ML, CSV export |
| Plus | $20 | 25 | 25,000 | Advanced ML, PDF reports |
| Pro | $60 | Unlimited | 100,000+ | Team features, API access |

### Environment Variables

**Backend (.env):**
```env
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
OPENAI_API_KEY=your_openai_api_key
STRIPE_SECRET_KEY=your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=your_webhook_secret
JWT_SECRET_KEY=your_jwt_secret
REDIS_URL=redis://localhost:6379
```

**Frontend (.env.local):**
```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=your_stripe_publishable_key
```

## 🚀 Deployment

### Backend (Fly.io)
```bash
cd tekk-backend
fly launch
fly deploy
```

### Frontend (Vercel)
```bash
cd tekk-frontend
vercel --prod
```

### Environment Setup
1. Set production environment variables in deployment platforms
2. Update CORS_ORIGINS in backend config
3. Configure Stripe webhooks to point to production API
4. Set up domain with Cloudflare DNS

## 📊 Usage Flow

1. **Sign Up**: User creates account (free plan)
2. **Create Project**: User creates a new analysis project
3. **Upload Data**: Drag-and-drop CSV/XLSX file
4. **Natural Language Prompt**: "Predict customer churn from this data"
5. **AI Analysis**: GPT-4 determines task type and target column
6. **AutoML Pipeline**: PyCaret trains multiple models automatically
7. **Results**: User gets AI-generated report with insights
8. **Download**: Export predictions, reports, visualizations

## 🔒 Security

- JWT authentication with secure token storage
- Row Level Security (RLS) in Supabase
- Input validation and sanitization
- Rate limiting on API endpoints
- Secure file upload with type validation
- Environment variable protection

## 🧪 Testing

```bash
# Backend tests
cd tekk-backend
pytest

# Frontend tests
cd tekk-frontend
npm test
```

## 📈 Monitoring

- FastAPI automatic OpenAPI docs at `/docs`
- Health check endpoint at `/api/v1/health`
- Stripe webhook monitoring
- Supabase dashboard for database monitoring

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Documentation: [docs.tekk.systems](https://docs.tekk.systems)
- Email: support@tekk.systems
- Discord: [Tekk Community](https://discord.gg/tekk)

---

Built with ❤️ for the research community 
