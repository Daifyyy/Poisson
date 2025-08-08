import subprocess
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent  # uprav, pokud skript není v kořeni repa

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
    Vrátí jeden z: 'uptodate', 'ahead', 'behind', 'diverged', 'noupstream'
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
    # stáhni nové reference
    run(["git", "fetch", "--all", "--prune"])
    state = upstream_state()
    if state in ("behind", "diverged"):
        print("🔄 Na vzdálené větvi jsou novější commity – provádím rebase…")
        pull = run(["git", "pull", "--rebase", "--autostash"], check=False)
        if pull.returncode != 0:
            print("❌ Rebase/pull selhal:")
            print(pull.stderr or pull.stdout)
            sys.exit(pull.returncode)
        print("✅ Rebase hotový, pokračuji.")
    elif state == "uptodate":
        print("✔️ Větev je aktuální vůči upstreamu.")
    elif state == "ahead":
        print("ℹ️ Lokální větev je napřed před upstreamem (OK).")
    else:
        print("ℹ️ Upstream není nastaven – přeskočím pull.")

def main():
    # 1) srovnat lokální stav s remote (případný rebase)
    ensure_rebased()

    # 2) na-stage-ovat všechno (tracked i untracked)
    run(["git", "add", "-A"])

    # 3) pokud není co commitnout, skonči
    status = run(["git", "status", "--porcelain"])
    if not status.stdout.strip():
        print("ℹ️ Není žádná změna k commitnutí.")
        return

    # 4) commit
    commit_message = f"Aktualizace – {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    run(["git", "commit", "-m", commit_message])

    # 5) push
    branch = get_current_branch()
    if has_upstream():
        run(["git", "push"])
    else:
        print(f"ℹ️ Větev '{branch}' zatím nemá upstream – nastavím 'origin/{branch}'.")
        run(["git", "push", "--set-upstream", "origin", branch])

    print("✅ Změny byly úspěšně commitnuty a odeslány.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("❌ Git chyba:")
        print(e.stderr or e.stdout)
        sys.exit(e.returncode)
