"""Streamlit login/signup gate."""

from __future__ import annotations

from typing import Optional

from .user_store import UserStore


def require_login(store: Optional[UserStore] = None) -> Optional[str]:
    """Block rendering until the user is authenticated. Returns username or None."""
    import streamlit as st

    store = store or UserStore()

    if st.session_state.get("authenticated") and st.session_state.get("username"):
        return st.session_state.username

    st.title("\U0001f4c8 Trading Simulator \u2014 Sign In")
    tab_login, tab_signup = st.tabs(["\U0001f510 Login", "\U0001f195 Create Account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Log in", type="primary")
        if submit:
            if store.verify_password(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username.strip().lower()
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_name = st.text_input("Display name")
            new_email = st.text_input("Email (optional)")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submit_signup = st.form_submit_button("Create account", type="primary")
        if submit_signup:
            if not new_username or not new_password:
                st.error("Username and password are required.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif store.user_exists(new_username):
                st.error("That username is already taken.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                store.add_user(new_username, new_name, new_password, new_email)
                st.success("Account created! Please log in above.")

    return None


def logout_button(key: str = "logout_btn") -> None:
    """Render a logout button in the sidebar."""
    import streamlit as st

    if st.sidebar.button("\U0001f6aa Log out", key=key):
        for k in ("authenticated", "username"):
            st.session_state.pop(k, None)
        st.rerun()
