import streamlit as st
import importlib

# Configure the main page
st.set_page_config(
    page_title="Brick Detection App",
    page_icon="static/brickicon3.png",
    layout="centered"
)

# Define the pages with corresponding icons
PAGES = {
    "BrickSense Single ☝️": "page1",
    "BrickSense Multiple ✌️": "page2",
    "BrickSense Augmented ✋": "page3"
}

def main():
    # Initialize session state for page selection
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = list(PAGES.keys())[0]  # Default to the first page

    # Sidebar navigation with icons
    st.sidebar.image("static/crackedwall1.png", width=250)
    st.sidebar.title("Navigation")
    st.sidebar.markdown("### Select a Page:")
    
    # Use selectbox for better user experience
    page = st.sidebar.selectbox(
        "Choose a page to visit:",
        list(PAGES.keys()),
        index=list(PAGES.keys()).index(st.session_state.selected_page)
    )
    
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
        /* Customize the sidebar navigation */
        .css-1vbd788, .stSelectbox {
            font-size: 18px;
            color: #2E8B57;
        }
        .stSelectbox>div {
            background-color: #F0F8FF;
            border-radius: 5px;
            padding: 5px;
        }
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

