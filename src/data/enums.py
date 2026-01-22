from enum import Enum

class DocumentStatusEnum(Enum):
    DOWNLOADED = 1
    FAILED_PARSED = 2
    PARSED = 3
    FAILED_EMBEDDED = 4
    EMBEDDED = 5
    READY = 6

    @property
    def id(self) -> int:
        return self.value
