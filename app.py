import streamlit as st
import importlib

# Configure the main page
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

def main():
    # Initialize session state for page selection
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Page 1"  # Default to Page 1

    # Display the main content
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
