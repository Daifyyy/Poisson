import streamlit as st
import pandas as pd
from io import BytesIO
from utils import bet_db


RESULT_OPTIONS = ["pending", "win", "loss", "push"]


def _calc_profit(result: str, stake: float, odds: float) -> float | None:
    """Return profit based on bet result."""
    if result == "win":
        return (odds - 1) * stake
    if result == "loss":
        return -stake
    if result == "push":
        return 0.0
    return None


def render_my_bets_section() -> None:
    """Display recorded bets with edit and export options."""
    st.title("📒 My Bets")

    stats = bet_db.compute_stats()
    cols = st.columns(2)
    cols[0].metric("ROI", f"{stats['roi']*100:.1f}%")
    cols[1].metric("Success Rate", f"{stats['win_rate']*100:.1f}%")

    bets = bet_db.fetch_bets()
    if not bets:
        st.info("No bets recorded.")
        return

    df = pd.DataFrame(bets)

    edited_df = st.data_editor(
        df,
        column_config={
            "result": st.column_config.SelectboxColumn("Result", options=RESULT_OPTIONS),
            "odds": st.column_config.NumberColumn("Odds", min_value=1.0, step=0.01),
            "stake": st.column_config.NumberColumn("Stake", min_value=0.0, step=0.1),
        },
        disabled=[
            "id",
            "league",
            "home_team",
            "away_team",
            "bet_type",
            "profit",
            "created_at",
        ],
        hide_index=True,
        width="stretch",
        key="bets_editor",
    )

    if st.button("💾 Update bets"):
        for row in edited_df.itertuples(index=False):
            original = df[df["id"] == row.id].iloc[0]
            updates = {}
            if row.result != original["result"]:
                updates["result"] = row.result
            if row.odds != original["odds"]:
                updates["odds"] = float(row.odds)
            if row.stake != original["stake"]:
                updates["stake"] = float(row.stake)
            if updates:
                profit = _calc_profit(row.result, float(row.stake), float(row.odds))
                updates["profit"] = profit
                bet_db.update_bet(row.id, **updates)
        st.success("Bets updated")
        st.rerun()

    st.markdown("### Delete bet")
    del_id = st.selectbox("Select bet ID", df["id"], key="delete_select")
    if st.button("🗑️ Delete selected bet"):
        bet_db.delete_bet(int(del_id))
        st.success("Bet deleted")
        st.rerun()

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name="bets.csv", mime="text/csv")

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        "⬇️ Download Excel",
        excel_buffer.getvalue(),
        file_name="bets.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
