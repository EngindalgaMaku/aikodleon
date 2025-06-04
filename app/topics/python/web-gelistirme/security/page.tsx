import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Security Best Practices | Python Web Development | Kodleon',
  description: 'Learn security best practices for Python web applications including authentication, authorization, data protection, and secure coding.',
};

const content = `
# Security Best Practices

Learn security best practices and secure coding techniques for Python web applications.

## Authentication and JWT

\`\`\`python
# security/auth.py
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Security configuration
SECRET_KEY = "your-secret-key"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Example usage in FastAPI endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
\`\`\`

## Authorization and RBAC

\`\`\`python
# security/rbac.py
from enum import Enum
from typing import List, Optional
from fastapi import Depends, HTTPException, status
from pydantic import BaseModel

class Role(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"

class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class RolePermissions:
    PERMISSIONS = {
        Role.ADMIN: [
            Permission.READ,
            Permission.WRITE,
            Permission.DELETE,
            Permission.ADMIN
        ],
        Role.MANAGER: [
            Permission.READ,
            Permission.WRITE
        ],
        Role.USER: [
            Permission.READ
        ]
    }

class UserInDB(BaseModel):
    username: str
    email: str
    role: Role
    disabled: Optional[bool] = None

def check_permission(required_permission: Permission):
    async def permission_dependency(
        current_user: UserInDB = Depends(get_current_user)
    ):
        if current_user.disabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user"
            )
        
        user_permissions = RolePermissions.PERMISSIONS[current_user.role]
        if required_permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    
    return permission_dependency

# Example usage in FastAPI endpoints
@app.get("/items/{item_id}")
async def read_item(
    item_id: int,
    user: UserInDB = Depends(check_permission(Permission.READ))
):
    return {"item_id": item_id, "owner": user.username}

@app.post("/items/")
async def create_item(
    item: dict,
    user: UserInDB = Depends(check_permission(Permission.WRITE))
):
    return {"item": item, "owner": user.username}

@app.delete("/items/{item_id}")
async def delete_item(
    item_id: int,
    user: UserInDB = Depends(check_permission(Permission.DELETE))
):
    return {"deleted": item_id, "by": user.username}
\`\`\`

## Input Validation and Sanitization

\`\`\`python
# security/validation.py
from typing import List
import re
from pydantic import BaseModel, EmailStr, validator, constr
from fastapi import HTTPException
import bleach
from email_validator import validate_email, EmailNotValidError

class UserInput(BaseModel):
    username: constr(min_length=3, max_length=50)
    email: EmailStr
    password: str
    bio: Optional[str] = None
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not re.match("^[a-zA-Z0-9_-]+$", v):
            raise ValueError('Username must be alphanumeric')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search("[A-Z]", v):
            raise ValueError('Password must contain an uppercase letter')
        if not re.search("[a-z]", v):
            raise ValueError('Password must contain a lowercase letter')
        if not re.search("[0-9]", v):
            raise ValueError('Password must contain a number')
        return v
    
    @validator('bio')
    def sanitize_bio(cls, v):
        if v:
            # Remove potentially malicious HTML
            return bleach.clean(
                v,
                tags=['p', 'b', 'i', 'u'],
                attributes={},
                strip=True
            )
        return v

def validate_email_address(email: str) -> bool:
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def sanitize_sql_input(value: str) -> str:
    # Remove SQL injection patterns
    patterns = [
        r'--',
        r';',
        r'\/\*',
        r'\*\/',
        r'xp_',
        r'sp_'
    ]
    sanitized = value
    for pattern in patterns:
        sanitized = re.sub(pattern, '', sanitized)
    return sanitized

# Example usage in FastAPI endpoints
@app.post("/users/")
async def create_user(user: UserInput):
    # Input is automatically validated by Pydantic
    sanitized_username = sanitize_sql_input(user.username)
    
    # Additional validation
    if not validate_email_address(user.email):
        raise HTTPException(
            status_code=400,
            detail="Invalid email address"
        )
    
    # Process validated and sanitized input
    return {"username": sanitized_username, "email": user.email}
\`\`\`

## Security Headers and CORS

\`\`\`python
# security/middleware.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://example.com",
        "https://api.example.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Content-Length"],
    max_age=600
)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'"
        )
        
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting middleware
from fastapi import Request
import time
from collections import defaultdict

class RateLimitExceeded(Exception):
    pass

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        requests_limit: int = 100,
        time_window: int = 60
    ):
        super().__init__(app)
        self.requests_limit = requests_limit
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < self.time_window
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_limit:
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )
        
        # Add new request
        self.requests[client_ip].append(current_time)
        
        response = await call_next(request)
        return response

app.add_middleware(
    RateLimitMiddleware,
    requests_limit=100,
    time_window=60
)
\`\`\`

## Exercises

1. **Authentication**
   - Implement JWT authentication
   - Add password hashing
   - Integrate OAuth2
   - Set up 2FA

2. **Authorization**
   - Create RBAC system
   - Add permission checks
   - Build role hierarchy
   - Implement access control lists

3. **Input Security**
   - Add input validation
   - Prevent SQL injection
   - Add XSS protection
   - Implement CSRF protection

## Next Steps

1. [Performance Optimization](/topics/python/web-gelistirme/performance)
2. [Cloud Deployment](/topics/python/web-gelistirme/deployment)
3. [Testing and Quality Assurance](/topics/python/web-gelistirme/testing)

## Useful Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Python Security Best Practices](https://snyk.io/blog/python-security-best-practices/)
- [Web Security Cheat Sheet](https://cheatsheetseries.owasp.org/)
`;

const learningPath = [
  {
    title: '1. Authentication',
    description: 'Learn authentication and identity management.',
    topics: [
      'JWT tokens',
      'Password hashing',
      'OAuth2 integration',
      'Two-factor auth',
      'Session management',
    ],
    icon: '🔐',
    href: '/topics/python/web-gelistirme/security/auth'
  },
  {
    title: '2. Authorization',
    description: 'Learn authorization and access control.',
    topics: [
      'Role-based access',
      'Permission system',
      'Access control lists',
      'Policy enforcement',
      'Audit logging',
    ],
    icon: '🛡️',
    href: '/topics/python/web-gelistirme/security/authz'
  },
  {
    title: '3. Data Protection',
    description: 'Learn data security and encryption.',
    topics: [
      'Encryption',
      'Key management',
      'Secure storage',
      'Data masking',
      'Backup security',
    ],
    icon: '🔒',
    href: '/topics/python/web-gelistirme/security/data'
  },
  {
    title: '4. Application Security',
    description: 'Learn application security practices.',
    topics: [
      'Input validation',
      'XSS prevention',
      'CSRF protection',
      'Security headers',
      'Rate limiting',
    ],
    icon: '🛡️',
    href: '/topics/python/web-gelistirme/security/appsec'
  }
];

export default function SecurityPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/web-gelistirme" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Web Development
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert">
          <MarkdownContent content={content} />
        </div>

        <h2 className="text-2xl font-bold mb-6">Learning Path</h2>
        
        <div className="grid gap-6 md:grid-cols-2">
          {learningPath.map((topic, index) => (
            <Card key={index} className="p-6 hover:bg-accent transition-colors cursor-pointer">
              <Link href={topic.href}>
                <div className="flex items-start space-x-4">
                  <div className="text-4xl">{topic.icon}</div>
                  <div className="space-y-2">
                    <h3 className="font-bold">{topic.title}</h3>
                    <p className="text-sm text-muted-foreground">{topic.description}</p>
                    <ul className="text-sm space-y-1 list-disc list-inside text-muted-foreground">
                      {topic.topics.map((t, i) => (
                        <li key={i}>{t}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Link>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Education Platform</p>
        </div>
      </div>
    </div>
  );
} 