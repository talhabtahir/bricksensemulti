import streamlit as st

def authenticate():
    """Handle authentication with a password check."""
    st.title("Authentication")
    password = st.text_input("Enter the password to access this page", type="password")
    
    if st.button("Submit"):
        if password == "1234":
            st.session_state.authenticated = True
            st.session_state.authenticating = False  # Ensure form does not display again
            st.success("Authenticated successfully!")
        else:
            st.error("Incorrect password. Please try again.")

def check_authentication():
    """Check if the user is authenticated. If not, prompt for authentication."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'authenticating' not in st.session_state:
        st.session_state.authenticating = True
    if not st.session_state.authenticated:
        st.session_state.authenticating = True
    if st.session_state.authenticating:
        authenticate()
        if st.session_state.authenticated:
            st.session_state.authenticating = False
            st.experimental_rerun()  # Rerun to clear authentication form after successful login
