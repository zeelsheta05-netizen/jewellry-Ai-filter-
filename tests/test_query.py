"""Parser regression tests. Run: .venv/bin/python -m pytest tests -q"""
import pytest

from jewelsearch.query import correct_spelling, parse

# prompt -> expected fields (only the listed ones are checked)
CASES = [
    # the prompt from the first round of client testing
    ("silver ring + single dimond on center (no more dimonds on ring)and simple without dimond thin width",
     dict(category="ring", metal="white_gold", has=["solitaire", "thin", "minimal"], hasnot=["side_stones"],
          lacks=["diamond", "plain"], xcat=[])),
    ("bracelate for party", dict(category="bracelet", has=["party"])),
    ("i want a braclet with diamonds", dict(category="bracelet", has=["diamond"])),
    ("rign for my wife", dict(category="ring", has=["women"])),
    ("pendent with heart", dict(category="pendant", has=["heart"])),
    ("neckless for wedding", dict(category="necklace", has=["bridal"])),
    ("earings for daily office wear", dict(category="earrings", has=["everyday"])),
    ("i want the design that is wearable everyday, a rose gold ring for my mother",
     dict(category="ring", metal="rose_gold", has=["everyday", "women"])),
    ("gents ring bina stone", dict(category="ring", has=["plain", "men"], lacks=["diamond"])),
    ("ring without diamonds", dict(category="ring", has=["plain"], lacks=["diamond"])),
    ("ring with no stones or diamonds", dict(category="ring", has=["plain"])),
    ("ring, not yellow gold", dict(category="ring", metal=None, xmetal=["yellow_gold"])),
    ("thin band ring with side diamonds", dict(category="ring", has=["thin", "side_stones"])),
    ("wide men's band", dict(has=["wide", "men"])),
    ("oval diamond engagement ring", dict(category="ring", shape="oval", has=["engagement", "diamond"])),
    ("tennis bracelet with diamonds for party", dict(category="bracelet", has=["tennis", "diamond", "party"])),
    ("chain bracelet", dict(category="bracelet")),
    ("platinum look solitaire", dict(metal="white_gold", has=["solitaire"])),
    # found by the 100-prompt sweep
    ("Wide men's band with diamonds for my husband", dict(category="ring", has=["wide", "men", "diamond"])),
    ("Eternity band with diamonds all around", dict(category="ring", has=["tennis"])),
    ("Plain gold band for daily use", dict(category="ring", has=["plain", "everyday"])),
    ("Cluster earrings in rose gold", dict(category="earrings", metal="rose_gold", has=["cluster"])),
    ("Heavy bridal ring with lots of diamonds", dict(category="ring", has=["cluster", "bridal"])),
    ("पार्टी के लिए चमकदार हार", dict(category="necklace", has=["party", "diamond"])),
    ("Drop earrings with pear shaped diamonds", dict(category="earrings", shape="pear", has=["drop"])),
    ("Stackable thin rings with small diamonds", dict(category="ring", has=["thin"])),
    # Gujarati
    ("મારી મમ્મી માટે રોજ પહેરી શકાય એવી રોઝ ગોલ્ડ વીંટી જોઈએ છે",
     dict(category="ring", metal="rose_gold", has=["everyday", "women"])),
    ("લગ્ન માટે ભારે હીરાનો હાર", dict(category="necklace", has=["bridal", "statement", "diamond"])),
    ("નાની અને સાદી કાનની બુટ્ટી, સફેદ સોનું", dict(category="earrings", metal="white_gold", has=["minimal"])),
    ("હીરા વગરની પાતળી વીંટી", dict(category="ring", has=["plain", "thin"], lacks=["diamond"])),
    ("વચ્ચે એક હીરો હોય એવી વીંટી", dict(category="ring", has=["solitaire"])),
    ("હાર્ટ વાળું પેન્ડન્ટ", dict(category="pendant", has=["heart"])),
    ("ચાંદી જેવી બંગડી", dict(category="bracelet", metal="white_gold")),
    # Hindi
    ("मेरी माँ के लिए रोज़ पहनने वाली रोज़ गोल्ड अंगूठी चाहिए",
     dict(category="ring", metal="rose_gold", has=["everyday", "women"])),
    ("हीरे के बिना पतली अंगूठी", dict(category="ring", has=["plain", "thin"], lacks=["diamond"])),
    ("बिना हीरे की अंगूठी", dict(category="ring", has=["plain"])),
    ("एक हीरा वाली सॉलिटेयर अंगूठी", dict(category="ring", has=["solitaire"])),
    # Hinglish
    ("koi floral type ka pendant chahiye yellow gold me, gift ke liye",
     dict(category="pendant", metal="yellow_gold", has=["floral", "gift"])),
    ("ring me diamond nahi chahiye", dict(category="ring", has=["plain"], lacks=["diamond"])),
]


@pytest.mark.parametrize("prompt,exp", CASES, ids=[c[0][:40] for c in CASES])
def test_parse(prompt, exp):
    q = parse(prompt)
    if "category" in exp:
        assert q.category == exp["category"]
    if "metal" in exp:
        assert q.metal == exp["metal"]
    if "shape" in exp:
        assert q.shape == exp["shape"]
    for i in exp.get("has", []):
        assert i in q.intents, f"{i} missing from {q.intents}"
    for i in exp.get("lacks", []):
        assert i not in q.intents, f"{i} should not be in {q.intents}"
    for i in exp.get("hasnot", []):
        assert i in q.not_intents, f"{i} missing from not_intents {q.not_intents}"
    if "xcat" in exp:
        assert q.exclude_categories == exp["xcat"]
    if "xmetal" in exp:
        assert q.exclude_metals == exp["xmetal"]


def test_several_categories():
    q = parse("gift for my sister, pendant or earrings")
    assert set(q.categories) == {"pendant", "earrings"}


def test_all_100_prompts_parse():
    from pathlib import Path
    lines = [l.strip() for l in (Path(__file__).parent / "prompts_100.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    assert len(lines) == 100
    no_type = [l for l in lines if not parse(l).categories]
    # only the genuinely open-ended request names no type
    assert no_type == ["something simple for my wife's birthday"]


@pytest.mark.parametrize("wrong,right", [
    ("dimond", "diamond"), ("dimonds", "diamonds"), ("bracelate", "bracelet"), ("braclet", "bracelet"),
    ("rign", "ring"), ("pendnat", "pendant"), ("neklace", "necklace"), ("solitare", "solitaire"),
    ("engagment", "engagement"), ("everday", "everyday"),
])
def test_spelling(wrong, right):
    assert correct_spelling(wrong) == right


@pytest.mark.parametrize("word", ["simple", "without", "silver", "thin", "mother", "party", "floral", "wife"])
def test_spelling_leaves_real_words(word):
    assert correct_spelling(word) == word


def test_negated_words_not_sent_to_model():
    q = parse("silver ring + single dimond on center (no more dimonds on ring) and simple without dimond thin width")
    assert "diamond" not in (q.free_text() or "")
    assert "diamond" not in q.visual_text().replace("single centre diamond", "")


@pytest.mark.parametrize("prompt", ["rani haar", "रानी हार", "લગ્ન માટે ભારે રાણી હાર", "dulhan ke liye bhari rani haar"])
def test_rani_haar(prompt):
    q = parse(prompt)
    assert q.categories == ["necklace"] and "rani_haar" in q.intents


@pytest.mark.parametrize("prompt", ["heavy bridal ring with intricate design work", "nakshi kaam wali bridal bangle",
                                    "ભારે કામ વાળી દુલ્હન ની બુટ્ટી", "भारी काम वाली अंगूठी"])
def test_design_work(prompt):
    assert "ornate" in parse(prompt).intents


STRUCTURAL = [
    ("I want a ring with a single round diamond in the center held by four prongs, a thin plain band with no "
     "stones on the sides, very minimal and elegant",
     dict(category="ring", shape="round", has=["solitaire", "prong", "thin"], lacks=["plain"], hasnot=["side_stones"])),
    ("Looking for a three stone ring, one bigger diamond in the middle and two smaller diamonds on each side, "
     "set on a slim yellow gold band", dict(category="ring", has=["three_stone"], lacks=["solitaire", "statement"])),
    ("Wide men's ring with a flat top, a square princess cut diamond set flush in the middle, and a brushed "
     "plain gold band without other stones", dict(category="ring", has=["men", "wide"], lacks=["plain"])),
    ("A bold cocktail ring with a big flower shaped top made of marquise diamonds", dict(category="ring", lacks=["men"])),
    ("Choker necklace that sits close to the neck, made of flower motifs", dict(category="necklace", lacks=["men"])),
    ("Rigid bangle bracelet with a hinge opening, a single row of diamonds on the top half and plain polished gold "
     "underneath", dict(category="bracelet", has=["bangle", "single_row"], lacks=["plain", "halo"])),
    ("Long drop earrings with a stud at the top and a pear shaped diamond hanging at the bottom on a thin chain",
     dict(category="earrings", shape="pear", has=["drop"])),
    ("An open ring with a gap at the top", dict(category="ring", has=["open_design"])),
    ("Open cuff bracelet with two ends that face each other", dict(category="bracelet", has=["cuff"])),
    ("A leaf design ring, no big center stone", dict(category="ring", has=["leaf"], hasnot=["big_stone", "solitaire"])),
    ("Chandelier earrings for a bride, multiple layers of diamonds", dict(category="earrings", has=["chandelier", "rows"])),
    ("ek patli ring chahiye jisme beech me ek bada round diamond ho aur side me koi stone na ho",
     dict(category="ring", has=["thin", "big_stone"], hasnot=["side_stones"], lacks=["statement"])),
    ("लंबी लटकने वाली बाली चाहिए जिसके नीचे बूंद जैसा हीरा हो", dict(category="earrings", shape="pear", has=["drop"],
                                                              lacks=["hoop"])),
    ("પાતળી વીંટી જોઈએ છે જેમાં વચ્ચે એક મોટો ગોળ હીરો હોય", dict(category="ring", has=["thin", "big_stone"],
                                                       lacks=["statement"])),
    ("કાનની લટકતી બુટ્ટી જેમાં નીચે ટીપા જેવો હીરો હોય", dict(category="earrings", shape="pear", has=["drop"])),
]


@pytest.mark.parametrize("prompt,exp", STRUCTURAL, ids=[c[0][:40] for c in STRUCTURAL])
def test_structural(prompt, exp):
    test_parse(prompt, exp)


def test_real_words_are_not_corrected():
    for w in ["made", "half", "covered", "hinge", "petals", "prongs", "wrapping"]:
        assert correct_spelling(w) == w


def test_all_structural_prompts_have_a_type():
    from pathlib import Path
    lines = [l.strip() for l in (Path(__file__).parent / "prompts_structural.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    assert all(parse(l).categories for l in lines)
