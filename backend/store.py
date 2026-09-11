"""
Durable interview store (record of truth).

Redis holds hot session state and expires. This holds the permanent record:
transcript, per-turn rubric, and the graded scorecard, so a recruiter can still
read what a candidate said months after the call.

Backend is SQLite today. Everything goes through the small DAO below and uses
parameterised SQL, so moving to Postgres means reimplementing this one class,
not touching the app.

Enable/point it with INTERVIEW_DB_PATH (default: interviews.db next to app.py).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Interview.Store")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "interviews.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS interviews (
    session_id            TEXT PRIMARY KEY,
    candidate_name        TEXT,
    target_role           TEXT,
    track                 TEXT,
    stage                 TEXT,
    created_at            REAL,
    started_at            REAL,
    ended_at              REAL,
    duration_seconds      INTEGER,
    updated_at            REAL,
    candidate_json        TEXT,
    job_requirements_json TEXT,
    interview_state_json  TEXT,
    resume_text           TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    session_id             TEXT NOT NULL,
    turn_index             INTEGER NOT NULL,
    asked_question         TEXT,
    candidate_answer       TEXT,
    next_question          TEXT,
    competency             TEXT,
    topic                  TEXT,
    question_id            TEXT,
    question_difficulty    INTEGER,
    question_mode          TEXT,
    live_score             REAL,
    expected_concepts_json TEXT,
    asked_at               REAL,
    PRIMARY KEY (session_id, turn_index)
);

CREATE TABLE IF NOT EXISTS reports (
    session_id     TEXT PRIMARY KEY,
    overall_score  REAL,
    recommendation TEXT,
    graded_by      TEXT,
    turns_graded   INTEGER,
    graded_at      REAL,
    report_json    TEXT
);

CREATE INDEX IF NOT EXISTS idx_interviews_role    ON interviews(target_role);
CREATE INDEX IF NOT EXISTS idx_interviews_ended   ON interviews(ended_at);
CREATE INDEX IF NOT EXISTS idx_reports_rec        ON reports(recommendation);
"""

def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def iso_ts(value: Any) -> Optional[str]:
    """Unix seconds (or ms) → UTC ISO-8601. None if missing."""
    if value is None or value == "":
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    if ts > 1e12:
        ts /= 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def duration_between(started_at: Any, ended_at: Any) -> Optional[int]:
    if started_at is None or ended_at is None:
        return None
    try:
        seconds = int(float(ended_at) - float(started_at))
    except (TypeError, ValueError):
        return None
    return max(0, seconds)


def interview_timing(started_at: Any, ended_at: Any, duration_seconds: Any = None) -> Dict[str, Any]:
    started = float(started_at) if started_at not in (None, "") else None
    ended = float(ended_at) if ended_at not in (None, "") else None
    duration = duration_seconds
    if duration is None:
        duration = duration_between(started, ended)
    elif duration is not None:
        try:
            duration = max(0, int(duration))
        except (TypeError, ValueError):
            duration = duration_between(started, ended)
    return {
        "started_at": started,
        "ended_at": ended,
        "duration_seconds": duration,
        "started_at_iso": iso_ts(started),
        "ended_at_iso": iso_ts(ended),
    }


def _loads(raw: Optional[str], fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


class InterviewStore:
    """
    Durable store DAO.

    To move to Postgres: reimplement this class with the same method names.
    Callers only use save_interview / save_report / get_* / list_interviews.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or os.getenv("INTERVIEW_DB_PATH") or DEFAULT_DB_PATH)
        self._available = False
        try:
            with self._connect() as conn:
                conn.executescript(SCHEMA)
                self._migrate(conn)
            self._available = True
            logger.info(f"[InterviewStore] Ready at {self.db_path}")
        except Exception as exc:
            logger.error(f"[InterviewStore] Disabled, could not open {self.db_path}: {exc}")

    @property
    def is_available(self) -> bool:
        return self._available

    def _connect(self) -> sqlite3.Connection:
        # A fresh connection per operation keeps this safe from worker threads.
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add start/end timing columns to DBs created before those fields existed."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(interviews)")}
        if "started_at" not in cols:
            conn.execute("ALTER TABLE interviews ADD COLUMN started_at REAL")
        if "duration_seconds" not in cols:
            conn.execute("ALTER TABLE interviews ADD COLUMN duration_seconds INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_interviews_started ON interviews(started_at)"
        )
        conn.execute(
            """
            UPDATE interviews
            SET started_at = COALESCE(started_at, created_at)
            WHERE started_at IS NULL
            """
        )
        conn.execute(
            """
            UPDATE interviews
            SET ended_at = (
                SELECT MAX(t.asked_at) FROM turns t WHERE t.session_id = interviews.session_id
            )
            WHERE ended_at IS NULL
              AND (
                    stage = 'completed'
                    OR EXISTS (SELECT 1 FROM reports r WHERE r.session_id = interviews.session_id)
                  )
              AND EXISTS (
                    SELECT 1 FROM turns t
                    WHERE t.session_id = interviews.session_id AND t.asked_at IS NOT NULL
                  )
            """
        )
        conn.execute(
            """
            UPDATE interviews
            SET duration_seconds = CAST(
                ended_at - COALESCE(started_at, created_at) AS INTEGER
            )
            WHERE ended_at IS NOT NULL
              AND duration_seconds IS NULL
              AND COALESCE(started_at, created_at) IS NOT NULL
            """
        )

    # ─── Writes ───────────────────────────────────────────────────────────────

    def save_interview(self, session: Dict[str, Any]) -> bool:
        """Upsert the interview and its full transcript. Safe to call repeatedly."""
        if not self._available:
            return False
        session_id = session.get("session_id")
        if not session_id:
            return False

        candidate = session.get("candidate") or {}
        state = session.get("interview_state") or {}
        history = session.get("history") or []
        started_at = session.get("started_at") or session.get("created_at")
        ended_at = session.get("ended_at")
        duration_seconds = session.get("duration_seconds")
        if duration_seconds is None:
            duration_seconds = duration_between(started_at, ended_at)

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO interviews (
                        session_id, candidate_name, target_role, track, stage,
                        created_at, started_at, ended_at, duration_seconds, updated_at,
                        candidate_json, job_requirements_json, interview_state_json, resume_text
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        candidate_name        = excluded.candidate_name,
                        target_role           = excluded.target_role,
                        track                 = excluded.track,
                        stage                 = excluded.stage,
                        started_at            = COALESCE(interviews.started_at, excluded.started_at),
                        ended_at              = COALESCE(excluded.ended_at, interviews.ended_at),
                        duration_seconds      = CASE
                            WHEN COALESCE(excluded.ended_at, interviews.ended_at) IS NOT NULL
                            THEN MAX(0, CAST(
                                COALESCE(excluded.ended_at, interviews.ended_at)
                                - COALESCE(interviews.started_at, excluded.started_at, excluded.created_at)
                                AS INTEGER
                            ))
                            ELSE interviews.duration_seconds
                        END,
                        updated_at            = excluded.updated_at,
                        candidate_json        = excluded.candidate_json,
                        job_requirements_json = excluded.job_requirements_json,
                        interview_state_json  = excluded.interview_state_json,
                        resume_text           = excluded.resume_text
                    """,
                    (
                        session_id,
                        candidate.get("name"),
                        candidate.get("target_role"),
                        candidate.get("target_track") or candidate.get("bank_track"),
                        state.get("stage"),
                        session.get("created_at") or started_at,
                        started_at,
                        ended_at,
                        duration_seconds,
                        time.time(),
                        _dumps(candidate),
                        _dumps(session.get("job_requirements") or {}),
                        _dumps(state),
                        (session.get("resume_text") or "")[:20000],
                    ),
                )

                for idx, turn in enumerate(history, start=1):
                    conn.execute(
                        """
                        INSERT INTO turns (
                            session_id, turn_index, asked_question, candidate_answer, next_question,
                            competency, topic, question_id, question_difficulty, question_mode,
                            live_score, expected_concepts_json, asked_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(session_id, turn_index) DO UPDATE SET
                            asked_question         = excluded.asked_question,
                            candidate_answer       = excluded.candidate_answer,
                            next_question          = excluded.next_question,
                            competency             = excluded.competency,
                            topic                  = excluded.topic,
                            question_id            = excluded.question_id,
                            question_difficulty    = excluded.question_difficulty,
                            question_mode          = excluded.question_mode,
                            live_score             = excluded.live_score,
                            expected_concepts_json = excluded.expected_concepts_json,
                            asked_at               = excluded.asked_at
                        """,
                        (
                            session_id,
                            turn.get("turn") or idx,
                            turn.get("asked_question"),
                            turn.get("candidate_answer"),
                            turn.get("next_question"),
                            turn.get("competency"),
                            turn.get("topic"),
                            turn.get("question_id"),
                            turn.get("question_difficulty"),
                            turn.get("question_mode"),
                            turn.get("live_score"),
                            _dumps(turn.get("expected_concepts") or []),
                            turn.get("asked_at"),
                        ),
                    )
            return True
        except Exception as exc:
            logger.error(f"[InterviewStore] save_interview failed for {session_id}: {exc}")
            return False

    def save_report(self, session_id: str, report: Dict[str, Any]) -> bool:
        if not self._available or not session_id:
            return False
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO reports (
                        session_id, overall_score, recommendation, graded_by,
                        turns_graded, graded_at, report_json
                    ) VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        overall_score  = excluded.overall_score,
                        recommendation = excluded.recommendation,
                        graded_by      = excluded.graded_by,
                        turns_graded   = excluded.turns_graded,
                        graded_at      = excluded.graded_at,
                        report_json    = excluded.report_json
                    """,
                    (
                        session_id,
                        report.get("overall_score"),
                        report.get("recommendation"),
                        report.get("graded_by"),
                        report.get("turns_graded"),
                        report.get("graded_at") or time.time(),
                        _dumps(report),
                    ),
                )
            return True
        except Exception as exc:
            logger.error(f"[InterviewStore] save_report failed for {session_id}: {exc}")
            return False

    # ─── Reads ────────────────────────────────────────────────────────────────

    def get_interview(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Rebuild a session-shaped dict from durable storage (Redis may be long gone)."""
        if not self._available:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM interviews WHERE session_id = ?", (session_id,)
                ).fetchone()
                if not row:
                    return None
                turns = conn.execute(
                    "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_index", (session_id,)
                ).fetchall()

            timing = interview_timing(
                row["started_at"] if "started_at" in row.keys() else row["created_at"],
                row["ended_at"],
                row["duration_seconds"] if "duration_seconds" in row.keys() else None,
            )
            return {
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                **timing,
                "candidate": _loads(row["candidate_json"], {}),
                "job_requirements": _loads(row["job_requirements_json"], {}),
                "interview_state": _loads(row["interview_state_json"], {}),
                "resume_text": row["resume_text"] or "",
                "history": [
                    {
                        "turn": t["turn_index"],
                        "asked_question": t["asked_question"],
                        "candidate_answer": t["candidate_answer"],
                        "next_question": t["next_question"],
                        "competency": t["competency"],
                        "topic": t["topic"],
                        "question_id": t["question_id"],
                        "question_difficulty": t["question_difficulty"],
                        "question_mode": t["question_mode"],
                        "live_score": t["live_score"],
                        "expected_concepts": _loads(t["expected_concepts_json"], []),
                        "asked_at": t["asked_at"],
                    }
                    for t in turns
                ],
            }
        except Exception as exc:
            logger.error(f"[InterviewStore] get_interview failed for {session_id}: {exc}")
            return None

    def get_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self._available:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT report_json FROM reports WHERE session_id = ?", (session_id,)
                ).fetchone()
            return _loads(row["report_json"], None) if row else None
        except Exception as exc:
            logger.error(f"[InterviewStore] get_report failed for {session_id}: {exc}")
            return None

    def list_interviews(
        self,
        limit: int = 50,
        target_role: Optional[str] = None,
        recommendation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Recruiter listing: newest first, with score when grading has finished."""
        if not self._available:
            return []
        sql = """
            SELECT i.session_id, i.candidate_name, i.target_role, i.stage,
                   i.created_at, i.started_at, i.ended_at, i.duration_seconds,
                   r.overall_score, r.recommendation, r.graded_by,
                   (SELECT COUNT(*) FROM turns t WHERE t.session_id = i.session_id) AS turn_count
            FROM interviews i
            LEFT JOIN reports r ON r.session_id = i.session_id
        """
        clauses, params = [], []
        if target_role:
            clauses.append("i.target_role = ?")
            params.append(target_role)
        if recommendation:
            clauses.append("r.recommendation = ?")
            params.append(recommendation)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY COALESCE(i.ended_at, i.started_at, i.created_at) DESC LIMIT ?"
        params.append(int(limit))

        try:
            with self._connect() as conn:
                rows = []
                for raw in conn.execute(sql, params).fetchall():
                    row = dict(raw)
                    row.update(interview_timing(
                        row.get("started_at") or row.get("created_at"),
                        row.get("ended_at"),
                        row.get("duration_seconds"),
                    ))
                    rows.append(row)
                return rows
        except Exception as exc:
            logger.error(f"[InterviewStore] list_interviews failed: {exc}")
            return []


interview_store = InterviewStore()
