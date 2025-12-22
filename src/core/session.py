

import csv
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


__all__ = ["SessionManager"]


class SessionManager:

    def __init__(self, base_session_dir: Path):
        self.BASE_SESSION_DIR = Path(base_session_dir)
        self.BASE_SESSION_DIR.mkdir(exist_ok=True)
        
        self.session_id: Optional[str] = None
        self.CURRENT_SESSION_DIR: Optional[Path] = None
        self.CURRENT_VECTOR_DIR: Optional[Path] = None
        self.CURRENT_DATA_DIR: Optional[Path] = None
        self.CURRENT_LOGS_DIR: Optional[Path] = None
        self.CSV_LOG_FILE: Optional[Path] = None
        self.GLOBAL_CSV_LOG_FILE = self.BASE_SESSION_DIR / "interactions.csv"

    def create_session_directories(self, corpus_name: str):
        existing_sessions = sorted(
            [
                d
                for d in self.BASE_SESSION_DIR.iterdir()
                if d.is_dir() and d.name.startswith("session_")
            ]
        )

        for session_dir in existing_sessions:
            data_dir = session_dir / "data"
            raw_dir = data_dir / "raw"
            if raw_dir.exists():
                raw_files = list(raw_dir.glob("*.txt"))
                for raw_file in raw_files:
                    if raw_file.stem == corpus_name:
                        logging.info(
                            f"Found existing session for corpus '{corpus_name}': {session_dir.name}"
                        )
                        self.session_id = session_dir.name.replace("session_", "")
                        self.CURRENT_SESSION_DIR = session_dir
                        self.CURRENT_VECTOR_DIR = session_dir / "faiss_index"
                        self.CURRENT_DATA_DIR = data_dir
                        self.CURRENT_LOGS_DIR = session_dir / "logs"
                        self.CSV_LOG_FILE = self.CURRENT_LOGS_DIR / "interactions.csv"

                        logging.info(
                            f"Reusing session: {self.CURRENT_SESSION_DIR.name}"
                        )
                        return True

        self.session_id = str(uuid.uuid4())
        self.CURRENT_SESSION_DIR = self.BASE_SESSION_DIR / f"session_{self.session_id}"
        self.CURRENT_VECTOR_DIR = self.CURRENT_SESSION_DIR / "faiss_index"
        self.CURRENT_DATA_DIR = self.CURRENT_SESSION_DIR / "data"
        self.CURRENT_LOGS_DIR = self.CURRENT_SESSION_DIR / "logs"
        self.CSV_LOG_FILE = self.CURRENT_LOGS_DIR / "interactions.csv"

        self.CURRENT_SESSION_DIR.mkdir(exist_ok=True)
        self.CURRENT_VECTOR_DIR.mkdir(exist_ok=True)
        self.CURRENT_DATA_DIR.mkdir(exist_ok=True)
        self.CURRENT_LOGS_DIR.mkdir(exist_ok=True)

        logging.info(f"Created new session: {self.CURRENT_SESSION_DIR.name}")
        return False

    def initialize_csv_logs(self):
        if not self.CSV_LOG_FILE.exists():
            with open(self.CSV_LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "question",
                        "answer",
                        "retrieved_docs_count",
                        "response_time_s",
                    ]
                )
            logging.info(f"Initialized session CSV log: {self.CSV_LOG_FILE}")

        if not self.GLOBAL_CSV_LOG_FILE.exists():
            with open(
                self.GLOBAL_CSV_LOG_FILE, "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "session_id",
                        "timestamp",
                        "question",
                        "answer",
                        "retrieved_docs_count",
                        "response_time_s",
                    ]
                )
            logging.info(f"Initialized global CSV log: {self.GLOBAL_CSV_LOG_FILE}")

    def log_interaction_to_csv(
        self,
        question: str,
        response: str,
        retrieved_docs_count: int,
        response_time: float,
    ):
        timestamp = datetime.now().isoformat()

        with open(self.CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [timestamp, question, response, retrieved_docs_count, response_time]
            )

        with open(self.GLOBAL_CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.session_id,
                    timestamp,
                    question,
                    response,
                    retrieved_docs_count,
                    response_time,
                ]
            )

        logging.info(
            f"Logged interaction: Q='{question[:50]}...', Docs={retrieved_docs_count}, Time={response_time:.2f}s"
        )

    def get_interaction_stats(self):
        if not self.CSV_LOG_FILE.exists():
            logging.warning("No CSV log file found.")
            return None

        with open(self.CSV_LOG_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            logging.info("No interactions logged yet.")
            return None

        total_interactions = len(rows)
        total_time = sum(float(row["response_time_s"]) for row in rows)
        avg_time = total_time / total_interactions if total_interactions > 0 else 0
        avg_docs = (
            sum(int(row["retrieved_docs_count"]) for row in rows) / total_interactions
            if total_interactions > 0
            else 0
        )

        return {
            "total_interactions": total_interactions,
            "avg_response_time": avg_time,
            "avg_docs_retrieved": avg_docs,
        }

    def view_recent_interactions(self, limit: int = 5):
        if not self.CSV_LOG_FILE.exists():
            logging.warning("No CSV log file found.")
            return []

        with open(self.CSV_LOG_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        recent = rows[-limit:] if len(rows) > limit else rows

        logging.info(f"Recent {len(recent)} interactions:")
        for i, row in enumerate(recent, 1):
            logging.info(
                f"  {i}. Q: {row['question'][:50]}... | Docs: {row['retrieved_docs_count']} | Time: {row['response_time_s']}s"
            )

        return recent

    def export_session_summary(self):
        stats = self.get_interaction_stats()
        if stats is None:
            logging.info("No session summary to export.")
            return

        summary_file = self.CURRENT_LOGS_DIR / "session_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Session Directory: {self.CURRENT_SESSION_DIR}\n")
            f.write(f"Total Interactions: {stats['total_interactions']}\n")
            f.write(f"Average Response Time: {stats['avg_response_time']:.2f}s\n")
            f.write(
                f"Average Documents Retrieved: {stats['avg_docs_retrieved']:.2f}\n"
            )

        logging.info(f"Session summary exported to {summary_file}")
