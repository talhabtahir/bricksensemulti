import streamlit as st

def run():
    st.title("Post-Authentication Navigation")
    st.write("You are now authenticated! Please choose a page to visit:")
    
    page = st.selectbox(
        "Select a page:",
        ["Page 1", "Page 2", "Page 3"],
        index=0
    )
    
    if st.button("Go to Page"):
        st.session_state.selected_page = page
        st.session_state.show_navigation = True  # Ensure navigation is visible
        st.experimental_rerun()  # Rerun to load the selected page

