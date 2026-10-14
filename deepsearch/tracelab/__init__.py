"""DeepSearch TraceLab integration for protocol-based uploads and PEDR queries.

This package provides the integration with TraceLab's API:
- /api/v1/deepsearch/ingest: Upload MissionProtocolComplete payloads
- /api/v1/pedr/preflight: Check for existing research before web searches
- /api/v1/missions/{id}/logs: Live mission log push (TracelabLogHandler)
- /api/v1/missions/{id}/evidence: Trigger persisted evidence projection
- /api/v1/webhooks/deepsearch: Signed post-commit completion receipts

Authenticated API calls prefer a service-principal API key with JWT fallback.
"""

from .auth import (
    TokenInfo,
    TraceLabAuth,
    TraceLabAuthError,
    get_default_auth,
    is_secure_tracelab_url,
    reset_default_auth,
)
from .evidence_ledger import (
    EvidenceLedgerClient,
    EvidenceLedgerDeliveryCategory,
    EvidenceLedgerDeliveryResult,
    EvidenceLedgerTrigger,
)
from .log_handler import TracelabLogHandler
from .pedr import (
    PEDRClient,
    PEDRError,
    PreflightAction,
    PreflightMatch,
    PreflightResult,
    get_default_pedr_client,
    reset_default_pedr_client,
)
from .receipt import (
    CompletionReceipt,
    CompletionReceiptClient,
    ReceiptDeliveryResult,
)
from .uploader import (
    TraceLabUploader,
    UploadResult,
    upload_protocol,
)

__all__ = [
    # Auth
    "TraceLabAuth",
    "TraceLabAuthError",
    "TokenInfo",
    "get_default_auth",
    "is_secure_tracelab_url",
    "reset_default_auth",
    # Evidence Ledger trigger
    "EvidenceLedgerClient",
    "EvidenceLedgerDeliveryCategory",
    "EvidenceLedgerDeliveryResult",
    "EvidenceLedgerTrigger",
    # Log handler
    "TracelabLogHandler",
    # Completion receipt
    "CompletionReceipt",
    "CompletionReceiptClient",
    "ReceiptDeliveryResult",
    # PEDR
    "PEDRClient",
    "PEDRError",
    "PreflightAction",
    "PreflightMatch",
    "PreflightResult",
    "get_default_pedr_client",
    "reset_default_pedr_client",
    # Uploader
    "TraceLabUploader",
    "UploadResult",
    "upload_protocol",
]
