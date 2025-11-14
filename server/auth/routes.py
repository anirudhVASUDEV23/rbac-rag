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



##Yes — it behaves very similarly to middleware, but with an important difference.
##✅ Depends(authenticate) is NOT global middleware — it is per-endpoint middleware.
# ✔️ Similar to middleware because:

# It runs before your endpoint function

# It can block the request (return 401, 403, etc.)

# It can inject data into the endpoint (like user)

# ❌ But it is not global middleware because:

# It only runs for endpoints that use it

# It does not apply to all routes automatically

# Middleware sees every request → Depends only sees selected ones
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
    