# backend.py
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
import uvicorn
import secrets
from fastapi.security import HTTPBearer
from datetime import datetime, timedelta

app = FastAPI(title="TicketFlow API")
security = HTTPBearer()


# Data models
class Ticket(BaseModel):
    id: int
    description: str
    user: str
    status: str = "open"


class TicketCreate(BaseModel):
    description: str


class TicketUpdate(BaseModel):
    description: Optional[str] = None
    status: Optional[str] = None


class Token(BaseModel):
    token: str
    user: str


class UserCredentials(BaseModel):
    username: str
    password: str


# In-memory storage
tickets_db: Dict[int, Ticket] = {}
tokens_db: Dict[str, str] = {}  # token -> username mapping
ticket_id_counter = 1

# User database
users = [
    (1, "MartyMcFly", "Password1"),
    (2, "DocBrown", "flux-capacitor-123"),
    (3, "BiffTannen", "Password3"),
    (4, "GeorgeMcFly", "Password4")
]


def verify_credentials(username: str, password: str) -> bool:
    """Verify user credentials against the user database"""
    return any(user[1] == username and user[2] == password for user in users)


def get_current_user(authorization: str = Header(...)) -> str:
    """Validate token and return associated username"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = authorization.split("Bearer ")[1]
    username = tokens_db.get(token)

    if not username:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return username


@app.post("/auth/token")
def create_token(credentials: UserCredentials):
    """Create a new authentication token"""
    if not verify_credentials(credentials.username, credentials.password):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    # Generate new token
    token = secrets.token_urlsafe(32)
    tokens_db[token] = credentials.username

    return Token(token=token, user=credentials.username)


@app.get("/health")
def health():
    return "OK"


@app.get("/tickets")
def list_tickets(current_user: str = Depends(get_current_user)) -> List[Ticket]:
    """List all tickets for the authenticated user"""
    return [ticket for ticket in tickets_db.values() if ticket.user == current_user]


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int, current_user: str = Depends(get_current_user)):
    """Get a specific ticket if it belongs to the authenticated user"""
    if ticket_id not in tickets_db:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket = tickets_db[ticket_id]
    if ticket.user != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    return ticket


@app.post("/tickets")
def create_ticket(ticket: TicketCreate, current_user: str = Depends(get_current_user)):
    """Create a new ticket for the authenticated user"""
    global ticket_id_counter
    new_ticket = Ticket(
        id=ticket_id_counter,
        description=ticket.description,
        user=current_user
    )
    tickets_db[ticket_id_counter] = new_ticket
    ticket_id_counter += 1
    return new_ticket


@app.put("/tickets/{ticket_id}")
def update_ticket(
        ticket_id: int,
        ticket_update: TicketUpdate,
        current_user: str = Depends(get_current_user)
):
    """Update a ticket if it belongs to the authenticated user"""
    if ticket_id not in tickets_db:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket = tickets_db[ticket_id]
    if ticket.user != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    if ticket_update.description is not None:
        ticket.description = ticket_update.description
    if ticket_update.status is not None:
        ticket.status = ticket_update.status

    tickets_db[ticket_id] = ticket
    return ticket


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)