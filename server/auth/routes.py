from fastapi import APIRouter,HTTPException,Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .models import SignUpRequest
from .hash_utils import hash_password, verify_password
from config.db import users_collection

router = APIRouter()
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    user = users_collection.find_one({"username": credentials.username})
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": user["username"], "role": user["role"]}


@router.post("/signup")
def signup(signup_request: SignUpRequest):
    if users_collection.find_one({"username": signup_request.username}):
        raise HTTPException(status_code=400, detail="User already exists")
    hashed_password = hash_password(signup_request.password)
    users_collection.insert_one({"username": signup_request.username, "password": hashed_password, "role": signup_request.role})
    return {"message": "User created successfully"}


@router.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}
    