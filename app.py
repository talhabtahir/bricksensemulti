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
    
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    def initialize_access_permissions():
    st.session_state.access_page_1 = True  # Set access to True or False based on your requirements
    st.session_state.access_page_2 = False
    st.session_state.access_page_3 = False

    initialize_access_permissions()

    # Authentication
    if not st.session_state.authenticated:
        password = st.text_input("Enter the password to access the app", type="password")
        if st.button("Submit"):
            if password == "1234":
                st.session_state.authenticated = True
                # st.experimental_rerun()  # Optional: only if you need to force a full rerun
            else:
                st.error("Incorrect password. Please try again.")
        return

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Display content based on the page selection and access control
    if page == "Page 1":
        if st.session_state.authenticated:
            import page1
        else:
            st.warning("You do not have access to this page.")
    elif page == "Page 2":
        if st.session_state.authenticated:
            import page2
        else:
            st.warning("You do not have access to this page.")
    elif page == "Page 3":
        if st.session_state.authenticated:
            import page3
        else:
            st.warning("You do not have access to this page.")

if __name__ == "__main__":
    main()
