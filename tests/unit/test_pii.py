from src.security.pii import get_analyzer, get_anonymizer, mask_pii


def test_masks_person_email_and_phone():
    text = "Hi, my name is John Smith. Reach me at john.smith@example.com or 212-555-0198."
    masked = mask_pii(text)

    assert "<PERSON>" in masked
    assert "<EMAIL_ADDRESS>" in masked
    assert "<PHONE_NUMBER>" in masked
    assert "John Smith" not in masked
    assert "john.smith@example.com" not in masked
    assert "212-555-0198" not in masked


def test_masks_ssn_ip_and_location():
    # 123-45-6789 is presidio's own denylisted "canonical example SSN" and is
    # deliberately never flagged — use a non-denylisted fake number instead.
    text = "SSN 234-56-7890, IP 192.168.1.10, based in Boston."
    masked = mask_pii(text)

    assert "<US_SSN>" in masked
    assert "234-56-7890" not in masked
    assert "<IP_ADDRESS>" in masked
    assert "192.168.1.10" not in masked
    assert "<LOCATION>" in masked
    assert "Boston" not in masked


def test_masks_credit_card_and_iban():
    text = "Card 4111111111111111, IBAN GB33BUKB20201555555555."
    masked = mask_pii(text)

    assert "<CREDIT_CARD>" in masked
    assert "4111111111111111" not in masked
    assert "<IBAN_CODE>" in masked
    assert "GB33BUKB20201555555555" not in masked


def test_leaves_non_pii_text_unchanged():
    text = "The quarterly revenue grew by 12 percent this year."
    assert mask_pii(text) == text


def test_empty_string_returns_empty_string():
    assert mask_pii("") == ""


def test_entities_param_restricts_what_gets_masked():
    text = "Contact John Doe at john@x.com"
    masked = mask_pii(text, entities=["PERSON"])

    assert "<PERSON>" in masked
    assert "John Doe" not in masked
    assert "john@x.com" in masked  # EMAIL_ADDRESS excluded, must survive untouched


def test_analyzer_and_anonymizer_are_cached_singletons():
    assert get_analyzer() is get_analyzer()
    assert get_anonymizer() is get_anonymizer()
