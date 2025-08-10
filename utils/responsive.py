import streamlit as st


def init_responsive_layout() -> None:
    """Inject CSS to make Streamlit columns stack on small screens."""
    st.markdown(
        """
        <style>
        @media (max-width: 600px) {
          div[data-testid="column"] { flex: 1 1 100% !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def responsive_columns(*args, **kwargs):
    """Wrapper for st.columns enabling future responsiveness tweaks."""
    return st.columns(*args, **kwargs)
