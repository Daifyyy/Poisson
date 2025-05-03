import subprocess
import datetime

# Přidat změny
subprocess.run(["git", "add", "data/"], check=True)

# Zkontroluj, jestli je co commitnout
status_output = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)

if not status_output.stdout.strip():
    print("ℹ️ Není žádná změna k commitnutí.")
else:
    # Vytvořit commit
    commit_message = f"Aktualizace CSV souborů – {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    subprocess.run(["git", "commit", "-m", commit_message], check=True)

    # Push na GitHub
    subprocess.run(["git", "push"], check=True)
    print("✅ Změny byly úspěšně commitnuty a odeslány.")
