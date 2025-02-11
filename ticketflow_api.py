from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Optional, List
import secrets
import uvicorn
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

# --- Database Setup ---
DATABASE_URL = "sqlite:///./ticketflow.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}  # Needed for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    tickets = relationship("TicketModel", back_populates="owner")


class TicketModel(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, default="open")
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("UserModel", back_populates="tickets")


# Create the database tables
Base.metadata.create_all(bind=engine)


# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting up...")
    db: Session = SessionLocal()
    try:
        # Bootstrap users if needed
        if db.query(UserModel).count() == 0:
            print("Bootstrapping initial users...")
            initial_users = [
                {"first_name": "Marty", "last_name": "McFly", "username": "martymcfly", "password": "Password1"},
                {"first_name": "Doc", "last_name": "Brown", "username": "docbrown", "password": "flux-capacitor-123"},
                {"first_name": "Biff", "last_name": "Tannen", "username": "bifftannen", "password": "Password3"},
                {"first_name": "George", "last_name": "McFly", "username": "georgemcfly", "password": "Password4"}
            ]
            for user in initial_users:
                db_user = UserModel(first_name=user["first_name"], last_name=user["last_name"],
                                    username=user["username"], password=user["password"])
                db.add(db_user)
            db.commit()

            # After creating users, create their initial tickets
            print("Bootstrapping initial tickets...")
            tickets_data = {
                "martymcfly": [
                    {
                        "title": "Time Machine Maintenance",
                        "description": "Need to check the flux capacitor and refuel with plutonium"
                    },
                    {
                        "title": "Guitar Amplifier Issue",
                        "description": "The mega-amplifier is making strange noises when I play Johnny B. Goode"
                    }
                ],
                "docbrown": [
                    {
                        "title": "DeLorean Upgrade Request",
                        "description": "Planning to install hover conversion and Mr. Fusion energy reactor"
                    },
                    {
                        "title": "Laboratory Equipment",
                        "description": "Need new equipment for temporal experiments and lightning rod repairs"
                    }
                ],
                "bifftannen": [
                    {
                        "title": "Car Wash Service",
                        "description": "Schedule regular cleaning for my 1946 Ford Super De Luxe"
                    },
                    {
                        "title": "Sports Almanac Missing",
                        "description": "Can't find my sports statistics book from the future"
                    }
                ],
                "georgemcfly": [
                    {
                        "title": "Writing Assistance",
                        "description": "Need help with science fiction story submissions"
                    },
                    {
                        "title": "Car Insurance Claim",
                        "description": "Filing a claim for the damage from hitting the pine tree"
                    }
                ]
            }

            for username, tickets in tickets_data.items():
                user = db.query(UserModel).filter(UserModel.username == username).first()
                if user:
                    for ticket_data in tickets:
                        ticket = TicketModel(
                            title=ticket_data["title"],
                            description=ticket_data["description"],
                            status="open",
                            user_id=user.id
                        )
                        db.add(ticket)
            db.commit()
            print("Bootstrap complete!")
        else:
            print("System already bootstrapped, skipping initialization.")
    finally:
        db.close()

    yield  # Server is now running

    # Shutdown logic (if needed)
    print("Shutting down...")


# --- FastAPI Application Setup ---
app = FastAPI(title="TicketFlow API", lifespan=lifespan)
security = HTTPBearer()

# In-memory tokens database (token -> username)
tokens_db: dict[str, str] = {}


# --- Pydantic Models (API Contract) ---
class Ticket(BaseModel):
    id: int
    description: str
    user: str
    status: str = "open"


class TicketCreate(BaseModel):
    description: str
    title: str


class TicketUpdate(BaseModel):
    description: Optional[str] = None
    status: Optional[str] = None


class Token(BaseModel):
    token: str
    user: str


class UserCredentials(BaseModel):
    username: str
    password: str


class UserInfo(BaseModel):
    id: int
    first_name: str
    last_name: str
    username: str


# --- Utility Functions ---
def ticket_model_to_ticket(ticket: TicketModel) -> Ticket:
    """Convert a TicketModel (SQLAlchemy) instance to a Ticket (Pydantic) instance."""
    return Ticket(
        id=ticket.id,
        description=ticket.description,
        status=ticket.status,
        user=ticket.owner.username if ticket.owner else "Unknown"
    )


# --- Authentication Dependencies & Helpers ---
def get_current_user(authorization: str = Header(...)) -> str:
    """
    Extract and validate the token from the Authorization header.
    Returns the username associated with the token.
    """
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


def get_current_user_obj(
        current_username: str = Depends(get_current_user), db: Session = Depends(get_db)
) -> UserModel:
    """
    Dependency that returns the UserModel for the currently authenticated user.
    """
    user = db.query(UserModel).filter(UserModel.username == current_username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def verify_credentials(username: str, password: str, db: Session) -> bool:
    """
    Verify that the given username and password match an entry in the user database.
    """
    user = db.query(UserModel).filter(UserModel.username == username, UserModel.password == password).first()
    return user is not None


# --- Endpoints ---

@app.post("/auth/token")
def create_token(credentials: UserCredentials, db: Session = Depends(get_db)):
    """
    Create a new authentication token.
    """
    if not verify_credentials(credentials.username, credentials.password, db):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )
    # Generate new token and store it in memory.
    token = secrets.token_urlsafe(32)
    tokens_db[token] = credentials.username
    return Token(token=token, user=credentials.username)


@app.get("/health")
def health():
    return "OK"


@app.get("/tickets")
def list_tickets(
        current_user: str = Depends(get_current_user),
        db: Session = Depends(get_db)
) -> List[Ticket]:
    """
    List all tickets for the authenticated user.
    """
    user = db.query(UserModel).filter(UserModel.username == current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    tickets = db.query(TicketModel).filter(TicketModel.user_id == user.id).all()
    return [ticket_model_to_ticket(t) for t in tickets]


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int, current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Get a specific ticket if it belongs to the authenticated user.
    """
    ticket = db.query(TicketModel).filter(TicketModel.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if ticket.owner.username != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    return ticket_model_to_ticket(ticket)


@app.post("/tickets")
def create_ticket(ticket: TicketCreate, current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Create a new ticket for the authenticated user.
    """
    user = db.query(UserModel).filter(UserModel.username == current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    new_ticket = TicketModel(
        description=ticket.description,
        title=ticket.title,
        status="open",
        user_id=user.id
    )
    db.add(new_ticket)
    db.commit()
    db.refresh(new_ticket)
    return ticket_model_to_ticket(new_ticket)


@app.put("/tickets/{ticket_id}")
def update_ticket(
        ticket_id: int,
        ticket_update: TicketUpdate,
        current_user: str = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Update a ticket if it belongs to the authenticated user.
    """
    ticket = db.query(TicketModel).filter(TicketModel.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if ticket_update.description is not None:
        ticket.description = ticket_update.description
    if ticket_update.status is not None:
        ticket.status = ticket_update.status
    db.commit()
    db.refresh(ticket)
    return ticket_model_to_ticket(ticket)


@app.get("/users/me")
def read_current_user(user: UserModel = Depends(get_current_user_obj)):
    """
    Return information about the current authenticated user.
    """
    return UserInfo(id=user.id, first_name=user.first_name, last_name=user.last_name, username=user.username)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)