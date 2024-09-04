import streamlit as st

# Define the pages
PAGES = {
    "Page 1": "page1",
    "Page 2": "page2",
    "Page 3": "page3"
}

def main():
    st.set_page_config(
        page_title="Brick Detection App",
        page_icon="static/brickicon8.png",
        layout="centered"
    )
    
    # Authentication (example: simple password protection)
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        password = st.text_input("Enter the password to access the app", type="5488")
        if password == "your_password_here":
            st.session_state.authenticated = True
            st.experimental_rerun()  # Rerun the app to update session state
        else:
            st.error("Incorrect password. Please try again.")
            return

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Load the appropriate page
    if page == "Page 1":
        if "access_page_1" in st.session_state and st.session_state.access_page_1:
            import page1
        else:
            st.warning("You do not have access to this page.")
    elif page == "Page 2":
        if "access_page_2" in st.session_state and st.session_state.access_page_2:
            import page2
        else:
            st.warning("You do not have access to this page.")
    elif page == "Page 3":
        if "access_page_3" in st.session_state and st.session_state.access_page_3:
            import page3
        else:
            st.warning("You do not have access to this page.")

if __name__ == "__main__":
    main()
