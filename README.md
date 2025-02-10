# Description
Launches a demo support bot (`support_agent.py`) that runs on chatgpt-4o-latest and is connected to the demo ticketing API (`ticketflow_api.py`).

Supported functions:
- Create a ticket for a user.
- List all tickets for a user.
- Update a ticket by ID.
- Query a ticket by ID.

The bot uses hardcoded users:
```python
                {"first_name": "Marty", "last_name": "McFly", "username": "martymcfly", "password": "Password1"},
                {"first_name": "Doc", "last_name": "Brown", "username": "docbrown", "password": "flux-capacitor-123"},
                {"first_name": "Biff", "last_name": "Tannen", "username": "bifftannen", "password": "Password3"},
                {"first_name": "George", "last_name": "McFly", "username": "georgemcfly", "password": "Password4"}
```

# Running
To run the agent, first specify OpenAI API key as an environment variable:
```shell
OPENAI_API_KEY=<your-openai-key>
```

The next step is to launch the agent by running:
```shell
docker-compose up --build
```

This command will run the API server on port 8000 and the agent on port 8001.

# Usage
1. Get an auth token:
```shell
curl --location 'http://localhost:8000/auth/token' \
--header 'Content-Type: application/json' \
--data '{"username": "martymcfly", "password": "Password1"}'
```

2. Ask the agent:
```shell
curl --location 'http://localhost:8001/agent' \
--header 'Authorization: Bearer KNKNoqU1F9faCud9cq4m1ZO1ZCaGCcbfwFBnpCp-Seo' \
--header 'Content-Type: application/json' \
--data '{
    "query": "Can you create a ticket for me please?"
}'
```