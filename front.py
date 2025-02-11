import streamlit as st
import os
import requests
from typing import Dict


def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'auth_token' not in st.session_state:
        st.session_state.auth_token = None
    if 'user_display_name' not in st.session_state:
        st.session_state.user_display_name = None
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    if 'should_rerun' not in st.session_state:
        st.session_state.should_rerun = False


def try_login(username: str, password: str) -> tuple[bool, str]:
    try:
        response = requests.post(
            f"{os.environ.get('TICKETFLOW_API_URL', 'http://ticketflow_api:8000')}/auth/token",
            json={"username": username, "password": password},
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            data = response.json()
            return True, data
        else:
            return False, "Invalid credentials. Please try again."
    except Exception as e:
        return False, f"Error during login: {str(e)}"


def handle_login():
    if st.session_state.username and st.session_state.password:
        success, result = try_login(st.session_state.username, st.session_state.password)
        if success:
            # Store authentication data in session state
            st.session_state.auth_token = result['token']
            st.session_state.user_display_name = result['user']
            st.session_state.authenticated = True
            st.session_state.should_rerun = True
        else:
            st.error(result)
    else:
        st.warning("Please enter both username and password.")


def display_login_page():
    st.title("Vulnerable Support Agent Login")

    with st.container():
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=handle_login, use_container_width=True)

        st.markdown("</div></div>", unsafe_allow_html=True)


def display_chat_message(message: Dict[str, str], is_user: bool):
    if is_user:
        st.write(f'<div style="display: flex; justify-content: flex-end; margin: 10px 0;">'
                 f'<div style="background-color: #007AFF; color: white; padding: 10px; '
                 f'border-radius: 15px; max-width: 70%;">'
                 f'{message["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.write(f'<div style="display: flex; justify-content: flex-start; margin: 10px 0;">'
                 f'<div style="background-color: #E9ECEF; color: black; padding: 10px; '
                 f'border-radius: 15px; max-width: 70%;">'
                 f'{message["content"]}</div></div>', unsafe_allow_html=True)


def handle_logout():
    # Clear the state on logout
    st.session_state.clear()
    st.session_state.should_rerun = True


def process_message(message: str):
    if message:
        user_message = {"role": "user", "content": message}
        st.session_state.messages.append(user_message)

        try:
            response = requests.post(
                f"{os.environ.get('SUPPORT_AGENT_URL', 'http://support_agent:8001')}/agent",
                json={"query": message},
                headers={"Authorization": f"Bearer {st.session_state.auth_token}"}
            )

            if response.status_code == 200:
                assistant_message = {
                    "role": "assistant",
                    "content": response.json()["response"]
                }
                st.session_state.messages.append(assistant_message)
            else:
                st.error("Failed to get response from agent")
        except Exception as e:
            st.error(f"Error communicating with agent: {str(e)}")

        # Increment input key to create a new input field
        st.session_state.input_key += 1
        st.session_state.should_rerun = True


def handle_submit():
    input_key = f"text_input_{st.session_state.input_key}"
    if input_key in st.session_state and st.session_state[input_key]:
        message = st.session_state[input_key]
        process_message(message)


def display_chat_interface():
    st.title(f"Support Agent Chat - {st.session_state.user_display_name}")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(message, message["role"] == "user")

    # Input for new message
    with st.container():
        st.write('<div style="height: 100px;"></div>', unsafe_allow_html=True)  # Spacer

        # Create the text input with a unique key each time
        st.text_input(
            "Type your message:",
            key=f"text_input_{st.session_state.input_key}",
            placeholder="Ask about tickets or request support...",
            on_change=handle_submit
        )


def display_sidebar():
    """Display sidebar with instructions and additional info."""
    with st.sidebar:
        st.title("Support Agent")
        st.markdown("""
        I can help you with any queries related to your support ticket.
        On your behalf I can:
        - Create a ticket by describing your issue
        - View your existing tickets
        - Update ticket status
        - Query specific ticket details

        Type your message in the chat to get started!
        """)

        if st.button("Logout", use_container_width=True, on_click=handle_logout):
            pass


def main():
    st.set_page_config(
        page_title="Support Agent Chat",
        page_icon="💬",
        layout="wide"
    )

    init_session_state()

    if st.session_state.should_rerun:
        st.session_state.should_rerun = False
        st.rerun()

    if not st.session_state.authenticated:
        display_login_page()
    else:
        display_sidebar()
        display_chat_interface()


if __name__ == "__main__":
    main()