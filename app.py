import streamlit as st
import importlib

# Configure the main page (must be called first)
st.set_page_config(
    page_title="Brick Detection App",
    page_icon="static/brickicon8.png",
    layout="centered"
)

# Define the pages
PAGES = {
    "Page 1": "page1",
    "Page 2": "page2",
    "Page 3": "page3"
}

def authenticate():
    """Handle authentication with a password check."""
    st.title("Authentication")
    password = st.text_input("Enter the password to access the app", type="password")
    
    if st.button("Submit"):
        if password == "1234":
            st.session_state.authenticated = True
            st.session_state.selected_page = "Page 1"  # Automatically set to Page 1 after authentication
            st.success("Authenticated successfully! Redirecting to Page 1...")
        else:
            st.error("Incorrect password. Please try again.")

def main():
    # Initialize session state for authentication and page selection
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Page 1"  # Default to Page 1

    if st.session_state.authenticated:
        # Display the main content if authenticated
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", list(PAGES.keys()), index=list(PAGES.keys()).index(st.session_state.selected_page))
        st.session_state.selected_page = page

        # Dynamically import and run the selected page
        page_module = PAGES[page]
        try:
            page_app = importlib.import_module(page_module)
            page_app.run()  # Ensure each page module has a run() function
        except ModuleNotFoundError:
            st.error("Page not found. Please check the page configuration.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        # Display authentication form if not authenticated
        authenticate()

# Apply custom CSS for UI improvements
def add_custom_css():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }
        .stTextInput>input {
            border-radius: 8px;
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    add_custom_css()  # Add custom styles
    main()
