"""
ACL Service — Access Control List for Retrieval.

Provides fine-grained access control for document retrieval:
- Organization-level access
- Role-based access
- Department-based access

Filters documents based on user permissions during retrieval.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class AccessLevel(str, Enum):
    """Access level enumeration."""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


@dataclass
class UserPermissions:
    """User permissions for access control."""
    user_id: str
    organization_id: str
    roles: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    access_level: AccessLevel = AccessLevel.PUBLIC
    allowed_document_ids: list[str] = field(default_factory=list)
    denied_document_ids: list[str] = field(default_factory=list)

    def can_access_document(self, doc_permissions: "DocumentPermissions") -> bool:
        """Check if user can access a document.

        Args:
            doc_permissions: Document's permission requirements

        Returns:
            True if access is allowed
        """
        # Check explicit deny
        if doc_permissions.document_id in self.denied_document_ids:
            return False

        # Check explicit allow
        if doc_permissions.document_id in self.allowed_document_ids:
            return True

        # Check organization
        if doc_permissions.organization_id:
            if doc_permissions.organization_id != self.organization_id:
                return False

        # Check roles
        if doc_permissions.required_roles:
            if not any(role in doc_permissions.required_roles for role in self.roles):
                return False

        # Check departments
        if doc_permissions.required_departments:
            if not any(dept in doc_permissions.required_departments for dept in self.departments):
                return False

        # Check access level
        if doc_permissions.access_level != AccessLevel.PUBLIC:
            if self.access_level == AccessLevel.PUBLIC:
                return False
            elif self.access_level == AccessLevel.RESTRICTED:
                if doc_permissions.access_level in [AccessLevel.CONFIDENTIAL, AccessLevel.SECRET]:
                    return False
            elif self.access_level == AccessLevel.CONFIDENTIAL:
                if doc_permissions.access_level == AccessLevel.SECRET:
                    return False

        return True


@dataclass
class DocumentPermissions:
    """Document permission requirements."""
    document_id: str
    organization_id: Optional[str] = None
    required_roles: list[str] = field(default_factory=list)
    required_departments: list[str] = field(default_factory=list)
    access_level: AccessLevel = AccessLevel.PUBLIC


class ACLService:
    """Service for access control list management."""

    # Default roles hierarchy
    ROLE_HIERARCHY = {
        "admin": 4,
        "manager": 3,
        "employee": 2,
        "contractor": 1,
        "guest": 0,
    }

    def __init__(self):
        """Initialize ACL service."""
        self._document_permissions: dict[str, DocumentPermissions] = {}

    def register_document(
        self,
        document_id: str,
        organization_id: Optional[str] = None,
        required_roles: Optional[list[str]] = None,
        required_departments: Optional[list[str]] = None,
        access_level: AccessLevel = AccessLevel.PUBLIC,
    ) -> None:
        """Register document permissions.

        Args:
            document_id: Unique document identifier
            organization_id: Organization that owns the document
            required_roles: Roles required to access document
            required_departments: Departments required to access document
            access_level: Minimum access level required
        """
        self._document_permissions[document_id] = DocumentPermissions(
            document_id=document_id,
            organization_id=organization_id,
            required_roles=required_roles or [],
            required_departments=required_departments or [],
            access_level=access_level,
        )
        logger.info(f"Registered ACL for document: {document_id}")

    def get_document_permissions(self, document_id: str) -> Optional[DocumentPermissions]:
        """Get permissions for a document.

        Args:
            document_id: Document identifier

        Returns:
            DocumentPermissions or None if not found
        """
        return self._document_permissions.get(document_id)

    def build_filter(
        self,
        user_permissions: UserPermissions,
    ) -> dict:
        """Build metadata filter for retrieval.

        Args:
            user_permissions: User's permissions

        Returns:
            Metadata filter dictionary for vector store query
        """
        # Build filter for documents the user can access
        allowed_docs = []
        denied_docs = []

        for doc_id, doc_perms in self._document_permissions.items():
            if user_permissions.can_access_document(doc_perms):
                allowed_docs.append(doc_id)
            else:
                denied_docs.append(doc_id)

        # Build filter
        filter_dict = {}

        # Add organization filter if user belongs to one
        if user_permissions.organization_id:
            filter_dict["organization_id"] = user_permissions.organization_id

        # Add department filter
        if user_permissions.departments:
            filter_dict["department"] = {"$in": user_permissions.departments}

        # Add role filter
        if user_permissions.roles:
            filter_dict["allowed_roles"] = {"$in": user_permissions.roles}

        # Add access level filter
        if user_permissions.access_level != AccessLevel.PUBLIC:
            # Include documents at or below user's access level
            level_order = list(AccessLevel)
            user_level_idx = level_order.index(user_permissions.access_level)
            allowed_levels = [l.value for l in level_order[:user_level_idx + 1]]
            filter_dict["access_level"] = {"$in": allowed_levels}

        # Exclude explicitly denied documents
        if denied_docs:
            filter_dict["document_id"] = {"$nin": denied_docs}

        return filter_dict

    def filter_results(
        self,
        results: list[dict],
        user_permissions: UserPermissions,
    ) -> list[dict]:
        """Filter retrieval results based on user permissions.

        Args:
            results: Retrieved documents
            user_permissions: User's permissions

        Returns:
            Filtered list of documents
        """
        filtered = []

        for result in results:
            doc_id = result.get("metadata", {}).get("document_id")

            if doc_id:
                doc_perms = self._document_permissions.get(doc_id)
                if doc_perms:
                    if user_permissions.can_access_document(doc_perms):
                        filtered.append(result)
                    else:
                        logger.debug(f"Filtered out document {doc_id} due to ACL")
                        continue
                else:
                    # Document not registered, apply default rules
                    # Assume public if no permissions defined
                    filtered.append(result)
            else:
                # No document ID, include by default
                filtered.append(result)

        return filtered

    def check_access(
        self,
        document_id: str,
        user_permissions: UserPermissions,
    ) -> bool:
        """Check if user can access a specific document.

        Args:
            document_id: Document identifier
            user_permissions: User's permissions

        Returns:
            True if access is allowed
        """
        doc_perms = self._document_permissions.get(document_id)
        if not doc_perms:
            # No permissions defined, allow access
            return True

        return user_permissions.can_access_document(doc_perms)

    def get_accessible_documents(
        self,
        user_permissions: UserPermissions,
    ) -> list[str]:
        """Get list of document IDs the user can access.

        Args:
            user_permissions: User's permissions

        Returns:
            List of accessible document IDs
        """
        return [
            doc_id
            for doc_id, doc_perms in self._document_permissions.items()
            if user_permissions.can_access_document(doc_perms)
        ]
