"""Add or update a user in users.json:  python scripts/hash_password.py <username>   (prompts for password)"""
import getpass
import json
import os
import sys

import bcrypt

path = os.getenv("AUTH_USERS_FILE", "users.json")
username = sys.argv[1]
password = getpass.getpass(f"Password for {username}: ")
users = json.load(open(path)) if os.path.exists(path) else {}
users[username] = bcrypt.hashpw(password.encode()[:72], bcrypt.gensalt(rounds=12)).decode()
json.dump(users, open(path, "w"), indent=2)
print(f"saved {username} -> {path}")