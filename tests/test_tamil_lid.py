from meetassist import lid, tamil

def test_clean_tamil_collapses_duplicate_vowel_signs():
    assert tamil.clean_tamil("போோது வீடியோோம்") == "போது வீடியோம்"

def test_romanize_is_readable_tanglish():
    r = tamil.romanize("இப்போ நம்ம மொபைல் போன் யூஸ் பண்ணும் போது")
    assert r.startswith("ippo namma") and "pannum" in r and "podhu" in r
    assert tamil.romanize("hello world 123") == "hello world 123"
    assert "konjam" in tamil.romanize("கொஞ்சம்") and "vachu" in tamil.romanize("வச்சு")

def test_english_vs_tamil_routing():
    assert lid.is_english("Good morning everyone.", "குட் மார்னிங் எவரி ஒன்")
    assert lid.is_english("Please update your tickets before the end of the day.", "ப்ளீஸ் அப்டேட்")
    assert not lid.is_english("Ippo namma mobile phone uspanam budho", "இப்போ நம்ம மொபைல் போன் யூஸ்")
    # Whisper 'translated' Tamil into fluent English, but the Tamil recogniser heard Tamil particles -> still Tamil
    assert not lid.is_english("We analyze our previous activities and consider what we should do.", "இத எப்படி நடக்குதுன்னா அனலைஸ் பண்ணி")
    assert not lid.is_english("", "இது") and lid.is_english("Yes.", "யெஸ்")
