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
    "Single Picture ☝️": "page1",
    "Single & Multi ✌️": "page2",
    "Single with YOLOv5/Resnet50 ✋": "page3"
}

def about_section():
    """Displays the About section of the app."""
    st.title("About the App")
    st.markdown("""
        **Brick Detection App** is designed to help detect and classify cracks or defects in brick walls.
        
        ### Features:
        - **Home**: Overview of the app and quick navigation.
        - **Analysis**: Run analysis on images to detect cracks or defects.
        - **Reports**: View detailed reports of the analysis.
        
        ### Instructions:
        1. Use the sidebar or tabs to navigate between different sections.
        2. Upload brick wall images on the Analysis page for defect detection.
        3. View detailed reports and statistics on the Reports page.

        ### Technologies Used:
        - Python
        - Streamlit
        - TensorFlow (for detection model)
    """)

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("### Select a Page:")
    page = st.sidebar.selectbox(
        "Choose a page to visit:",
        list(PAGES.keys())
    )

    # If user selects "Home", show about the app section
    if page == "🏠 Home":
        about_section()
    else:
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
        /* Customize the about section title */
        .stMarkdown h1 {
            color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    add_custom_css()  # Add custom styles
    main()
