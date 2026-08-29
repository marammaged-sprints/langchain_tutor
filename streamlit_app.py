"""Streamlit Cloud entry point.

Running the app from the repository root keeps ``langchain_tutor`` importable as
a package while allowing the UI implementation to live inside the package.
"""

from langchain_tutor.app import main


main()
