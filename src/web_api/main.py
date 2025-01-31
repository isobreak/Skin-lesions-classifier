import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

from src.web_api.router import router as router_classification

app = FastAPI()
app.mount("/static", StaticFiles(directory="src/web_api/static"), name="static")

app.include_router(router_classification)

@app.get('/')
async def index():
    return RedirectResponse('/home/analysis')

origins = [
    "http://localhost:8080",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PATCH", "PUT"],
    allow_headers=["*"],
)