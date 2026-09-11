"""
Inspect interviews.db with the stdlib sqlite3 module.

  python -m backend.inspect_db
  python -m backend.inspect_db --session <id>
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from backend.store import DEFAULT_DB_PATH, interview_store


def main() -> None:
    parser = argparse.ArgumentParser(description="List SQLite interview records")
    parser.add_argument("--session", help="Print turns for one session_id")
    parser.add_argument("--db", default=interview_store.db_path or str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    db = Path(args.db)
    print("Engine: SQLite (stdlib sqlite3). Postgres is not used.")
    print("Path:  ", db)
    print("Exists:", db.is_file())
    if not db.is_file():
        print("No database yet. Run an interview or: python -m backend.compare_interviews")
        return

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    print("\n--- interviews ---")
    rows = conn.execute(
        """
        SELECT i.session_id, i.candidate_name, i.target_role, i.track, i.stage,
               r.overall_score, r.recommendation, r.graded_by,
               (SELECT COUNT(*) FROM turns t WHERE t.session_id = i.session_id) AS n_turns
        FROM interviews i
        LEFT JOIN reports r ON r.session_id = i.session_id
        ORDER BY COALESCE(i.ended_at, i.created_at) DESC
        LIMIT 20
        """
    ).fetchall()
    if not rows:
        print("(empty)")
    for row in rows:
        print(dict(row))

    sid = args.session or (rows[0]["session_id"] if rows else None)
    if sid:
        print(f"\n--- turns for {sid} ---")
        turns = conn.execute(
            """
            SELECT turn_index, question_mode, question_id, competency,
                   substr(asked_question, 1, 100) AS asked,
                   substr(candidate_answer, 1, 80) AS answer
            FROM turns WHERE session_id = ? ORDER BY turn_index
            """,
            (sid,),
        ).fetchall()
        for t in turns:
            print(dict(t))

        report = conn.execute(
            "SELECT overall_score, recommendation, graded_by, substr(report_json,1,200) AS preview FROM reports WHERE session_id = ?",
            (sid,),
        ).fetchone()
        if report:
            print("\n--- report ---")
            print(dict(report))

    conn.close()


if __name__ == "__main__":
    main()
