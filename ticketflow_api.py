# ticketflow_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn

app = FastAPI(title="TicketFlow API")

# Data models for tickets
class Ticket(BaseModel):
    id: int
    description: str
    user: str
    status: str = "open"

class TicketCreate(BaseModel):
    description: str
    user: str

class TicketUpdate(BaseModel):
    description: Optional[str] = None
    status: Optional[str] = None

# In-memory ticket store
tickets_db: Dict[int, Ticket] = {}
ticket_id_counter = 1

@app.post("/tickets")
def create_ticket(ticket: TicketCreate):
    global ticket_id_counter
    new_ticket = Ticket(id=ticket_id_counter, description=ticket.description, user=ticket.user)
    tickets_db[ticket_id_counter] = new_ticket
    ticket_id_counter += 1
    return new_ticket

@app.get("/tickets")
def list_tickets():
    return list(tickets_db.values())

@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int):
    if ticket_id in tickets_db:
        return tickets_db[ticket_id]
    raise HTTPException(status_code=404, detail="Ticket not found")

@app.put("/tickets/{ticket_id}")
def update_ticket(ticket_id: int, ticket_update: TicketUpdate):
    if ticket_id not in tickets_db:
        raise HTTPException(status_code=404, detail="Ticket not found")
    ticket = tickets_db[ticket_id]
    if ticket_update.description is not None:
        ticket.description = ticket_update.description
    if ticket_update.status is not None:
        ticket.status = ticket_update.status
    tickets_db[ticket_id] = ticket
    return ticket

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
