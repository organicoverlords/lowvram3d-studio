"""Explicit name for the strict CPU atlas-projection registration stage.

The compatibility implementation remains in register_mvadapter_front_reference
because existing receipts and callers use that path.
"""
from register_mvadapter_front_reference import main, register

__all__ = ["main", "register"]


if __name__ == "__main__":
    raise SystemExit(main())
