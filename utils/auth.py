"""
utils/auth.py
--------------
Very simple username/password + role based authentication for demo purposes.

IMPORTANT: This is NOT production-grade security. Passwords are stored in
plain text here purely for demo/portfolio simplicity. For a real deployment,
replace this with a proper auth provider (e.g. hashed passwords in a DB,
OAuth, or a library such as streamlit-authenticator).
"""

# demo user store: username -> (password, role, display_name)
USERS = {
    "admin": {"password": "admin123", "role": "Admin", "name": "Administrator"},
    "faculty": {"password": "faculty123", "role": "Faculty", "name": "Faculty Member"},
    "student": {"password": "student123", "role": "Student", "name": "Demo Student", "student_id": "STU0001"},
}


def authenticate(username: str, password: str):
    """Returns the user dict if credentials are valid, else None."""
    user = USERS.get(username)
    if user and user["password"] == password:
        return {"username": username, **user}
    return None
