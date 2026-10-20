"""Read-only provenance utilities; no construction or experiment on import."""
from common import *
from legacy_conditioning import physical_signature
def signature(state):return physical_signature(state.get('physical',state))
