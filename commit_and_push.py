import subprocess
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent  # adjust if this script is not in the repo root

def run(cmd, check=True):
    return subprocess.run(cmd, cwd=REPO_ROOT, check=check, text=True, capture_output=True)

def get_current_branch():
    out = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return out.stdout.strip()

def has_upstream():
    r = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], check=False)
    return r.returncode == 0

def upstream_state():
    """
    Returns one of: 'uptodate', 'ahead', 'behind', 'diverged', 'noupstream'
    """
    if not has_upstream():
        return "noupstream"
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    upstream = run(["git", "rev-parse", "@{u}"]).stdout.strip()
    base = run(["git", "merge-base", "HEAD", "@{u}"]).stdout.strip()

    if head == upstream:
        return "uptodate"
    elif head == base:
        return "behind"
    elif upstream == base:
        return "ahead"
    else:
        return "diverged"

def ensure_rebased():
    run(["git", "fetch", "--all", "--prune"])
    state = upstream_state()

    if state in ("behind", "diverged"):
        print("🔄 Remote branch has new commits – attempting rebase...")

        # Detekuj, zda jsou nějaké untracked soubory, které by mohly způsobit konflikt
        ls = run(["git", "ls-files", "--others", "--exclude-standard"])
        untracked_files = ls.stdout.strip().splitlines()

        # Pokud jsou untracked .csv soubory, přesuň je bokem
        moved = []
        for file in untracked_files:
            if file.endswith(".csv") and Path(file).exists():
                bak_path = Path(file).with_suffix(".csv.bak")
                Path(file).rename(bak_path)
                moved.append((file, str(bak_path)))
                print(f"⚠️ Přesunut untracked soubor: {file} → {bak_path}")

        pull = run(["git", "pull", "--rebase", "--autostash"], check=False)
        if pull.returncode != 0:
            print("❌ Rebase selhal:")
            print(pull.stderr or pull.stdout)
            # Obnov soubory zpět
            for original, backup in moved:
                Path(backup).rename(original)
                print(f"↩️ Obnoven: {backup} → {original}")
            sys.exit(pull.returncode)

        print("✅ Rebase proběhl úspěšně.")

        # Obnov soubory zpět po úspěšném rebase
        for original, backup in moved:
            Path(backup).rename(original)
            print(f"↩️ Obnoven: {backup} → {original}")

    elif state == "uptodate":
        print("✔️ Větev je aktuální s upstreamem.")
    elif state == "ahead":
        print("ℹ️ Lokální větev je napřed – pokračujeme.")
    else:
        print("ℹ️ Není nastaven upstream – pull přeskakuji.")

def main():
    # Step 1: Sync with remote (rebase if needed)
    ensure_rebased()

    # Step 2: Stage all changes (tracked and untracked)
    run(["git", "add", "-A"])

    # Step 3: Check if there is anything to commit
    status = run(["git", "status", "--porcelain"])
    if not status.stdout.strip():
        print("ℹ️ No changes to commit.")
        return

    # Step 4: Commit
    commit_message = f"Update – {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    run(["git", "commit", "-m", commit_message])

    # Step 5: Push
    branch = get_current_branch()
    if has_upstream():
        run(["git", "push"])
    else:
        print(f"ℹ️ Branch '{branch}' has no upstream – setting 'origin/{branch}'.")
        run(["git", "push", "--set-upstream", "origin", branch])

    print("✅ Changes committed and pushed successfully.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("❌ Git error:")
        print(e.stderr or e.stdout)
        sys.exit(e.returncode)
