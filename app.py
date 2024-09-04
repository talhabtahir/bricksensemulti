import streamlit as st

# Define the pages
PAGES = {
    "Page 1": "page1",
    "Page 2": "page2",
    "Page 3": "page3"
}

def main():
    # Configure the main page
    st.set_page_config(
        page_title="Brick Detection App",
        page_icon="static/brickicon8.png",
        layout="centered"
    )
    
    # Initialize session state for authentication and submission tracking
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    # Display authentication form if not authenticated
    if not st.session_state.authenticated:
        st.title("Authentication")
        password = st.text_input("Enter the password to access the app", type="password")
        
        if st.button("Submit") or st.session_state.submitted:
            if password == "your_password_here":
                st.session_state.authenticated = True
                st.session_state.submitted = False  # Reset submission state after successful login
                st.experimental_rerun()  # Rerun to enter the authenticated state
            else:
                st.session_state.submitted = True
                st.error("Incorrect password. Please try again.")
        return

    # Display the main content if authenticated
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Import and run the selected page dynamically
    page_module = PAGES[page]
    try:
        page_app = __import__(page_module)
        page_app.run()  # Ensure each page module has a run() function
    except ModuleNotFoundError:
        st.error("Page not found. Please check the page configuration.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
