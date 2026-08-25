from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

# Entities masked before any text reaches the embedder/vector store.
DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
    "LOCATION",
    "IBAN_CODE",
]

# Presidio's packaged default config points at en_core_web_lg (~700MB). Pin to
# the small model explicitly so it matches what setup actually downloads.
_NLP_CONFIGURATION = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerEngine:
    provider = NlpEngineProvider(nlp_configuration=_NLP_CONFIGURATION)
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


@lru_cache(maxsize=1)
def get_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


def mask_pii(text: str, entities: list[str] | None = None) -> str:
    if not text:
        return text
    entities = list(entities) if entities is not None else DEFAULT_ENTITIES
    results = get_analyzer().analyze(text=text, entities=entities, language="en")
    anonymized = get_anonymizer().anonymize(text=text, analyzer_results=results)
    return anonymized.text
