import streamlit as st
import os
import requests

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'auth_token' not in st.session_state:
        st.session_state.auth_token = None
    if 'user_display_name' not in st.session_state:
        st.session_state.user_display_name = None
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
    with st.container():
        _, col2, _ = st.columns(3)

        with col2:
            col2.title("Support Agent Login")
            col2.text_input("Username", key="username")
            col2.text_input("Password", type="password", key="password")
            col2.button("Login", on_click=handle_login, use_container_width=True)

            col2.markdown("</div></div>", unsafe_allow_html=True)

def handle_logout():
    st.session_state.clear()
    st.session_state.should_rerun = True

def process_message(message: str):
    if message:
        st.session_state.messages.append({"role": "user", "content": message})

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
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I apologize, but I'm having trouble processing your request right now."
                })
        except Exception as e:
            st.error(f"Error communicating with agent: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I apologize, but I'm having trouble processing your request right now."
            })

def display_chat_interface():
    st.title(f"Support Agent Chat - {st.session_state.user_display_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Type your message..."):
        process_message(prompt)
        st.rerun()

def display_sidebar():
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