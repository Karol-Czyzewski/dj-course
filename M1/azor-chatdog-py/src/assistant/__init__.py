"""
Assistant module initialization
Exports the Assistant class and assistant factory functions.
"""

from .assistent import Assistant
from .azor import create_azor_assistant
from .perfectionist import create_perfectionist_assistant
from .business import create_business_assistant
from .optimist import create_optimist_assistant

ASSISTANT_BUILDERS = {
	'AZOR': create_azor_assistant,
	'PERFEKCJONISTA': create_perfectionist_assistant,
	'BIZNESMEN': create_business_assistant,
	'OPTIMISTA': create_optimist_assistant,
	'OPTYMISTA': create_optimist_assistant,  # alias (common spelling)
}

def available_assistants() -> list[str]:
	# Deduplicate aliases (keep primary names first)
	primary_order = ['AZOR', 'PERFEKCJONISTA', 'BIZNESMEN', 'OPTIMISTA']
	return primary_order

def create_assistant_by_name(name: str) -> Assistant:
    key = name.strip().upper()
    builder = ASSISTANT_BUILDERS.get(key)
    if not builder:
        return create_azor_assistant()
    return builder()

__all__ = [
	'Assistant',
	'create_azor_assistant',
	'create_perfectionist_assistant',
	'create_business_assistant',
	'create_optimist_assistant',
	'create_assistant_by_name',
	'available_assistants'
]
