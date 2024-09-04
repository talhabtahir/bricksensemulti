import streamlit as st

# Define the pages
PAGES = {
    "Page 1": "page1",
    "Page 2": "page2",
    "Page 3": "page3"
}

def main():
    # Set the page configuration
    st.set_page_config(
        page_title="Brick Detection App",
        page_icon="static/brickicon8.png",
        layout="centered"
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Page access control
    if page == "Page 1":
        if "access_page_1" in st.session_state and st.session_state.access_page_1:
            # Import and run Page 1 code
            import page1
        else:
            st.warning("You do not have access to this page.")
    elif page == "Page 2":
        if "access_page_2" in st.session_state and st.session_state.access_page_2:
            # Import and run Page 2 code
            import page2
        else:
            st.warning("You do not have access to this page.")
    elif page == "Page 3":
        if "access_page_3" in st.session_state and st.session_state.access_page_3:
            # Import and run Page 3 code
            import page3
        else:
            st.warning("You do not have access to this page.")

if __name__ == "__main__":
    main()
