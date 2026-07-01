"""
Audit Service — Immutable Audit Logs.

Provides comprehensive audit logging:
- Document access
- Search activity
- User actions
- Admin actions

Logs are stored in an append-only manner to maintain integrity.
"""

import logging
import json
import hashlib
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Any
import os

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Audit action types."""
    # Document actions
    DOCUMENT_UPLOAD = "document.upload"
    DOCUMENT_DELETE = "document.delete"
    DOCUMENT_ACCESS = "document.access"
    DOCUMENT_SEARCH = "document.search"
    DOCUMENT_DOWNLOAD = "document.download"

    # Chat actions
    CHAT_MESSAGE = "chat.message"
    CHAT_FEEDBACK = "chat.feedback"

    # User actions
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_REGISTER = "user.register"

    # Admin actions
    ADMIN_USER_CREATE = "admin.user.create"
    ADMIN_USER_DELETE = "admin.user.delete"
    ADMIN_CONFIG_CHANGE = "admin.config.change"

    # System actions
    SYSTEM_ERROR = "system.error"
    SECURITY_BLOCK = "security.block"


class AuditLevel(str, Enum):
    """Audit log level."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SECURITY = "security"


@dataclass
class AuditLog:
    """Audit log entry."""
    timestamp: str
    action: str
    level: str
    user_id: Optional[str]
    organization_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    details: dict = field(default_factory=dict)
    previous_hash: Optional[str] = None
    hash: str = ""

    def __post_init__(self):
        """Generate hash for integrity."""
        content = f"{self.timestamp}{self.action}{self.user_id}{json.dumps(self.details, sort_keys=True)}"
        self.hash = hashlib.sha256(content.encode()).hexdigest()[:16]


class AuditService:
    """Service for immutable audit logging."""

    def __init__(self, audit_log_dir: str = "./data/audit"):
        """Initialize audit service.

        Args:
            audit_log_dir: Directory for audit log files
        """
        self.audit_log_dir = audit_log_dir
        self._current_hash: Optional[str] = None
        self._log_file: Optional[str] = None

        # Ensure directory exists
        os.makedirs(audit_log_dir, exist_ok=True)

        # Initialize log file for today
        self._init_log_file()

    def _init_log_file(self):
        """Initialize today's log file."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._log_file = os.path.join(self.audit_log_dir, f"audit_{today}.jsonl")
        self._current_hash = None

        # Load last hash if file exists
        if os.path.exists(self._log_file):
            try:
                with open(self._log_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_entry = json.loads(lines[-1])
                        self._current_hash = last_entry.get("hash")
            except Exception as e:
                logger.warning(f"Failed to load previous audit hash: {e}")

    def log(
        self,
        action: AuditAction,
        level: AuditLevel = AuditLevel.INFO,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> AuditLog:
        """Create an audit log entry.

        Args:
            action: Action being audited
            level: Log level
            user_id: User performing the action
            organization_id: Organization context
            resource_type: Type of resource being accessed
            resource_id: ID of resource being accessed
            ip_address: Client IP address
            user_agent: Client user agent
            details: Additional details

        Returns:
            Created audit log entry
        """
        # Rotate log file if day changed
        current_file = self._log_file
        self._init_log_file()
        if current_file != self._log_file:
            self._current_hash = None

        # Create log entry
        log_entry = AuditLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action.value,
            level=level.value,
            user_id=user_id,
            organization_id=organization_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            previous_hash=self._current_hash,
        )

        # Write to file (append-only)
        try:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(asdict(log_entry)) + "\n")
            self._current_hash = log_entry.hash
            logger.debug(f"Audit log: {action.value} by {user_id}")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

        return log_entry

    def log_document_access(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        """Log document access.

        Args:
            document_id: Document identifier
            user_id: User accessing document
            organization_id: Organization context
            ip_address: Client IP
        """
        return self.log(
            action=AuditAction.DOCUMENT_ACCESS,
            level=AuditLevel.INFO,
            user_id=user_id,
            organization_id=organization_id,
            resource_type="document",
            resource_id=document_id,
            ip_address=ip_address,
        )

    def log_search(
        self,
        query: str,
        results_count: int,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        """Log search activity.

        Args:
            query: Search query
            results_count: Number of results returned
            user_id: User performing search
            organization_id: Organization context
            ip_address: Client IP
        """
        return self.log(
            action=AuditAction.DOCUMENT_SEARCH,
            level=AuditLevel.INFO,
            user_id=user_id,
            organization_id=organization_id,
            resource_type="search",
            ip_address=ip_address,
            details={
                "query": query,
                "results_count": results_count,
            },
        )

    def log_security_event(
        self,
        event_type: str,
        details: dict,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        """Log security-related event.

        Args:
            event_type: Type of security event
            details: Event details
            user_id: Associated user
            ip_address: Client IP
        """
        return self.log(
            action=AuditAction.SECURITY_BLOCK,
            level=AuditLevel.SECURITY,
            user_id=user_id,
            ip_address=ip_address,
            details={
                "event_type": event_type,
                **details,
            },
        )

    def log_chat_message(
        self,
        message: str,
        conversation_id: str,
        user_id: Optional[str] = None,
        has_rag_context: bool = False,
    ):
        """Log chat message.

        Args:
            message: Message content
            conversation_id: Conversation ID
            user_id: User ID
            has_rag_context: Whether RAG was used
        """
        return self.log(
            action=AuditAction.CHAT_MESSAGE,
            level=AuditLevel.INFO,
            user_id=user_id,
            resource_type="conversation",
            resource_id=conversation_id,
            details={
                "message_length": len(message),
                "has_rag_context": has_rag_context,
            },
        )

    def get_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        action: Optional[AuditAction] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit logs.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            action: Filter by action type
            user_id: Filter by user
            limit: Maximum entries to return

        Returns:
            List of audit log entries
        """
        logs = []

        # Read all log files in date range
        if start_date and end_date:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            date_range = [start + __import__('datetime').timedelta(days=d) for d in range((end-start).days + 1)]
        else:
            # Default to today
            date_range = [datetime.now(timezone.utc)]

        for date in date_range:
            log_file = os.path.join(
                self.audit_log_dir,
                f"audit_{date.strftime('%Y-%m-%d')}.jsonl"
            )
            if not os.path.exists(log_file):
                continue

            try:
                with open(log_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)

                        # Apply filters
                        if action and entry.get("action") != action.value:
                            continue
                        if user_id and entry.get("user_id") != user_id:
                            continue

                        logs.append(entry)

                        if len(logs) >= limit:
                            break
            except Exception as e:
                logger.warning(f"Failed to read audit log {log_file}: {e}")

        return logs[:limit]

    def verify_integrity(self) -> bool:
        """Verify audit log integrity (chain of hashes).

        Returns:
            True if integrity is valid
        """
        if not os.path.exists(self._log_file):
            return True

        try:
            with open(self._log_file, "r") as f:
                lines = f.readlines()
                prev_hash = None

                for line in lines:
                    entry = json.loads(line)

                    if prev_hash and entry.get("previous_hash") != prev_hash:
                        logger.error(f"Chain broken at {entry.get('timestamp')}")
                        return False

                    prev_hash = entry.get("hash")

            return True
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
