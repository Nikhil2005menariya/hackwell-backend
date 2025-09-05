from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import diabetes_routes,obesity_routes

# =====================================================
# Create FastAPI App
# =====================================================
app = FastAPI(
    title="Chronic Disease Prediction API",
    description="API for predicting chronic disease deterioration (starting with Diabetes).",
    version="1.0.0",
)

# =====================================================
# CORS Middleware (allow frontend to connect)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 allow all origins (you can restrict to your frontend URL later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Register Routes
# =====================================================
app.include_router(diabetes_routes.router)
app.include_router(obesity_routes.router)  # <— THIS IS REQUIRED

# =====================================================
# Root Route (Health Check)
# =====================================================
@app.get("/")
def root():
    return {"message": "✅ Chronic Disease Prediction API is running!"}
