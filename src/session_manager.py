"""Session manager utilities.

This module provides the :class:`SessionManager` which encapsulates session
directory creation and CSV logging used by the RAG application. The class is
designed to be small and dependency-light so it can be unit-tested easily.
"""

from __future__ import annotations

import csv
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


__all__ = ["SessionManager"]


class SessionManager:
    """Encapsulate session directory and CSV logging state and helpers.

    The public API is intentionally small: create_session_directories,
    initialize_csv_logs, log_interaction_to_csv, export_session_summary, and a
    few inspection helpers.
    """

    def __init__(self, base_session_dir: Path):
        self.BASE_SESSION_DIR = Path(base_session_dir)
        self.BASE_SESSION_DIR.mkdir(exist_ok=True)

        # session-level state
        self.session_id: Optional[str] = None
        self.CURRENT_SESSION_DIR: Optional[Path] = None
        self.CURRENT_VECTOR_DIR: Optional[Path] = None
        self.CURRENT_DATA_DIR: Optional[Path] = None
        self.CURRENT_LOGS_DIR: Optional[Path] = None
        self.CSV_LOG_FILE: Optional[Path] = None

        # global CSV across all sessions
        self.GLOBAL_CSV_LOG_FILE: Path = self.BASE_SESSION_DIR / "interactions.csv"

    def create_session_directories(self, corpus_name: str):
        """Create session directories based on corpus name. Returns True if existing session is used."""
        # Use only corpus name as session identifier
        self.session_id = corpus_name
        self.CURRENT_SESSION_DIR = self.BASE_SESSION_DIR / f"session_{self.session_id}"

        # Check if session already exists
        if self.CURRENT_SESSION_DIR.exists():
            print(f"Session '{corpus_name}' already exists.")
            print(f"Session directory: {self.CURRENT_SESSION_DIR}")

            existing_chunks_file = (
                self.CURRENT_SESSION_DIR / "data" / "processed" / "chunks.pkl"
            )
            existing_faiss_index = (
                self.CURRENT_SESSION_DIR / "faiss_index" / "faiss.index"
            )

            if existing_chunks_file.exists() and existing_faiss_index.exists():
                print("Existing session contains processed data and FAISS index.")
                use_existing = (
                    input("Do you want to use the existing session? (y/n): ")
                    .lower()
                    .strip()
                )

                if use_existing == "y":
                    print(f"Using existing session: {corpus_name}")
                    # Set up directories for existing session
                    self.CURRENT_VECTOR_DIR = self.CURRENT_SESSION_DIR / "faiss_index"
                    self.CURRENT_DATA_DIR = self.CURRENT_SESSION_DIR / "data"
                    self.CURRENT_LOGS_DIR = self.CURRENT_SESSION_DIR / "logs"
                    self.CURRENT_LOGS_DIR.mkdir(exist_ok=True)
                    self.CSV_LOG_FILE = self.CURRENT_LOGS_DIR / "interactions.csv"
                    return True
                else:
                    print(
                        "Cannot proceed with existing session name. Please choose a different name."
                    )
                    exit()
            else:
                print(
                    "Existing session directory is incomplete. Creating new session structure."
                )

        # Create new session directory
        self.CURRENT_SESSION_DIR.mkdir(exist_ok=True)

        # Subdirectories within the session
        self.CURRENT_VECTOR_DIR = self.CURRENT_SESSION_DIR / "faiss_index"
        self.CURRENT_DATA_DIR = self.CURRENT_SESSION_DIR / "data"
        self.CURRENT_LOGS_DIR = self.CURRENT_SESSION_DIR / "logs"
        self.CURRENT_LOGS_DIR.mkdir(exist_ok=True)

        # CSV log file for tracking interactions
        self.CSV_LOG_FILE = self.CURRENT_LOGS_DIR / "interactions.csv"
        return False

    def initialize_csv_logs(self):
        headers = [
            "interaction_id",
            "session_id",
            "timestamp",
            "question",
            "response",
            "retrieved_docs_count",
            "response_time_seconds",
            "session_directory",
        ]

        # Initialize session-specific CSV
        if self.CSV_LOG_FILE and not self.CSV_LOG_FILE.exists():
            with open(self.CSV_LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        # Initialize global CSV
        if not self.GLOBAL_CSV_LOG_FILE.exists():
            with open(self.GLOBAL_CSV_LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_interaction_to_csv(
        self,
        question: str,
        response: str,
        retrieved_docs_count: int,
        response_time: float,
    ):
        interaction_id = str(uuid.uuid4())
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row_data = [
            interaction_id,
            self.session_id,
            current_timestamp,
            question.replace("\n", " ").replace("\r", ""),
            response.replace("\n", " ").replace("\r", ""),
            retrieved_docs_count,
            round(response_time, 3),
            str(self.CURRENT_SESSION_DIR),
        ]

        # Log to session-specific CSV
        if self.CSV_LOG_FILE:
            with open(self.CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

        # Log to global CSV
        with open(self.GLOBAL_CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        logging.info(f"Logged interaction {interaction_id} to CSV files")

    def get_interaction_stats(self):
        if not self.GLOBAL_CSV_LOG_FILE.exists():
            return "No interactions logged yet."

        try:
            import csv as _csv

            with open(self.GLOBAL_CSV_LOG_FILE, "r", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                interactions = list(reader)

            total_interactions = len(interactions)
            unique_sessions = len(set(row["session_id"] for row in interactions))
            avg_response_time = (
                sum(float(row["response_time_seconds"]) for row in interactions)
                / total_interactions
                if total_interactions > 0
                else 0
            )

            return f"Stats: {total_interactions} interactions, {unique_sessions} sessions, {avg_response_time:.2f}s avg response time"
        except Exception as e:
            return f"Error reading interaction stats: {e}"

    def view_recent_interactions(self, limit: int = 5):
        if not self.GLOBAL_CSV_LOG_FILE.exists():
            print("No interactions logged yet.")
            return

        try:
            import csv as _csv

            with open(self.GLOBAL_CSV_LOG_FILE, "r", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                interactions = list(reader)

            recent = (
                interactions[-limit:] if len(interactions) >= limit else interactions
            )

            print(f"\nRecent {len(recent)} interactions:")
            for i, interaction in enumerate(recent, 1):
                print(
                    f"\n{i}. [{interaction['timestamp']}] Session: {interaction['session_id']}"
                )
                print(f"   Q: {interaction['question'][:80]}...")
                print(f"   A: {interaction['response'][:80]}...")
                print(
                    f"   {interaction['retrieved_docs_count']} docs, {interaction['response_time_seconds']}s"
                )
        except Exception as e:
            print(f"Error reading recent interactions: {e}")

    def export_session_summary(self):
        if not self.CURRENT_LOGS_DIR:
            logging.error("No current logs directory set. Cannot export summary.")
            return

        summary_file = self.CURRENT_LOGS_DIR / "session_summary.txt"

        try:
            interactions = []
            if self.CSV_LOG_FILE and self.CSV_LOG_FILE.exists():
                import csv as _csv

                with open(self.CSV_LOG_FILE, "r", encoding="utf-8") as f:
                    reader = _csv.DictReader(f)
                    interactions = list(reader)

            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(f"RAG Session Summary\n")
                f.write(f"==================\n\n")
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Session Directory: {self.CURRENT_SESSION_DIR}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Interactions: {len(interactions)}\n\n")

                if interactions:
                    avg_time = sum(
                        float(row["response_time_seconds"]) for row in interactions
                    ) / len(interactions)
                    f.write(f"Average Response Time: {avg_time:.2f} seconds\n")
                    f.write(
                        f"Total Documents Retrieved: {sum(int(row['retrieved_docs_count']) for row in interactions)}\n\n"
                    )

                    f.write("Interactions:\n")
                    f.write("=============\n\n")
                    for i, interaction in enumerate(interactions, 1):
                        f.write(f"{i}. [{interaction['timestamp']}]\n")
                        f.write(f"   Question: {interaction['question']}\n")
                        f.write(f"   Answer: {interaction['response']}\n")
                        f.write(
                            f"   Retrieved: {interaction['retrieved_docs_count']} docs\n"
                        )
                        f.write(f"   Time: {interaction['response_time_seconds']}s\n\n")

            logging.info(f"Session summary exported to {summary_file}")
        except Exception as e:
            logging.error(f"Error exporting session summary: {e}")
