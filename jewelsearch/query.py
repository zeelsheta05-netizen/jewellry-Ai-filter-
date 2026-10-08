"""Understand a shopper's prompt in English, Hindi, Gujarati or Hinglish.

Pipeline
  1. normalise punctuation and case
  2. correct misspelt Latin-script words against the lexicon ("dimond")
  3. find negation scopes ("no ...", "without ...", "... વગર", "... के बिना")
  4. match lexicon terms longest-first, each match consuming its span, which
     resolves overlaps such as Hindi "रोज़ गोल्ड" (rose gold) vs "रोज़"
     (daily) or Gujarati "હાર્ટ" (heart) containing "હાર" (necklace)
  5. resolve the matches into filters, wanted/unwanted design attributes and
     English phrases for the image model

Hard constraints (category, metal) come only from this lexicon, never from a
model, so they are never guessed.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

# (kind, value, visual phrase or None, [terms])
# kind: category | metal | intent | shape
LEXICON = [
    # ---- categories -------------------------------------------------------
    ("category", "earrings", None, [
        "earrings", "earring", "ear rings", "ear ring", "earing", "earings", "studs", "stud",
        "hoops", "hoop", "huggie", "huggies", "jhumka", "jhumki", "jhumkas", "bali", "baliyan", "tops", "kundal",
        "kaan ki bali", "kan ni butti", "butti", "buti",
        "કાનની બુટ્ટી", "બુટ્ટી", "બુટ્ટીઓ", "બુટી", "ઝુમખા", "ઝુમકા", "ઇયરિંગ", "ઈયરિંગ", "ઇયરરિંગ", "કડી",
        "कान की बाली", "बाली", "बालियां", "बालियाँ", "झुमका", "झुमकी", "झुमके", "इयररिंग", "ईयररिंग", "टॉप्स", "कुंडल",
    ]),
    ("category", "pendant", None, [
        "pendant", "pendants", "pandant", "pendent", "locket", "lockets",
        "પેન્ડન્ટ", "પેંડન્ટ", "પેન્ડલ", "પેંડલ", "લોકેટ", "લૉકેટ",
        "पेंडेंट", "पेन्डेन्ट", "पेंडल", "लॉकेट", "लोकेट",
    ]),
    ("category", "necklace", None, [
        "necklace", "necklaces", "neckless", "necklace set", "haar", "choker", "mangalsutra", "chain",
        "rani haar", "rani har", "rani haaar", "raani haar", "long haar", "layered necklace", "multi layer necklace", "multilayer necklace", "layered haar", "રાણી હાર", "રાની હાર", "લાંબો હાર", "रानी हार", "रानीहार", "लंबा हार",
        "હાર", "ગળાનો હાર", "નેકલેસ", "ચોકર", "મંગળસૂત્ર", "ચેન",
        "हार", "गले का हार", "नेकलेस", "चोकर", "मंगलसूत्र", "चेन",
    ]),
    ("category", "bracelet", None, [
        "bracelet", "bracelets", "brelcate", "bangle", "bangles", "kada", "kadu", "cuff", "kangan", "bangdi",
        "chain bracelet", "hand chain",
        "બ્રેસલેટ", "કડું", "કડા", "બંગડી", "બંગડીઓ", "પોચી",
        "ब्रेसलेट", "कंगन", "कड़ा", "कडा", "चूड़ी", "चूड़ियां",
    ]),
    ("category", "ring", None, [
        "ring", "rings", "anguthi", "angoothi", "anguthee", "vinti", "veenti", "band ring",
        "વીંટી", "વીટી", "વિંટી", "અંગૂઠી", "અંગુઠી", "રિંગ",
        "अंगूठी", "अंगुठी", "अँगूठी", "रिंग", "छल्ला",
    ]),
    # ---- metals -----------------------------------------------------------
    ("metal", "rose_gold", "rose gold", [
        "rose gold", "rosegold", "pink gold", "rose-gold", "gulabi sona",
        "રોઝ ગોલ્ડ", "રોઝગોલ્ડ", "રોઝ", "ગુલાબી સોનું", "ગુલાબી",
        "रोज़ गोल्ड", "रोज गोल्ड", "रोजगोल्ड", "गुलाबी सोना", "गुलाबी",
    ]),
    ("metal", "white_gold", "white gold", [
        "white gold", "whitegold", "safed sona",
        "સફેદ સોનું", "સફેદ સોનાની", "વ્હાઇટ ગોલ્ડ", "વાઇટ ગોલ્ડ", "સફેદ",
        "सफेद सोना", "सफ़ेद सोना", "व्हाइट गोल्ड", "सफेद", "सफ़ेद",
    ]),
    # the catalogue is gold only; these are shown as white gold with a note
    ("metal", "white_gold~silver", "white gold", [
        "silver", "platinum", "white metal", "silver colour", "silver color", "chandi", "chaandi",
        "ચાંદી", "ચાંદીની", "સિલ્વર", "પ્લેટિનમ", "चांदी", "चाँदी", "सिल्वर", "प्लैटिनम",
    ]),
    ("metal", "yellow_gold", "yellow gold", [
        "yellow gold", "yellowgold", "pila sona", "peela sona",
        "પીળું સોનું", "પીળા સોનાની", "યલો ગોલ્ડ", "પીળું", "પીળી",
        "पीला सोना", "येलो गोल्ड", "पीला", "पीली",
    ]),
    # ---- stone layout -----------------------------------------------------
    ("intent", "solitaire", "solitaire ring with one single centre diamond", [
        "solitaire", "single diamond", "single stone", "one diamond", "1 diamond", "one stone", "1 stone",
        "only one diamond", "single center diamond", "single centre diamond", "center diamond",
        "centre diamond", "center stone", "centre stone", "diamond on center", "diamond on centre",
        "diamond in center", "diamond in centre", "diamond in the center", "diamond in the centre",
        "diamond at center", "diamond at the center", "diamond in middle", "diamond in the middle",
        "ek hira", "ek heera", "ek diamond", "beech me hira",
        "સોલિટેર", "સોલિટેયર", "એક હીરો", "એક હીરા", "વચ્ચે હીરો", "વચ્ચે એક હીરો",
        "सॉलिटेयर", "सोलिटेयर", "एक हीरा", "बीच में हीरा", "बीच में एक हीरा",
    ]),
    ("intent", "halo", "halo setting, centre stone surrounded by a ring of small diamonds", [
        "halo", "surrounded by diamonds", "surrounded by small diamonds", "surrounded", "around it",
        "આજુબાજુ", "ચારે બાજુ", "चारों ओर", "हेलो", "હેલો",
    ]),
    ("intent", "side_stones", "small diamonds set along the band", [
        "side stones", "side stone", "side diamonds", "diamonds on band", "diamonds on the band",
        "pave", "pavé", "micro pave", "diamond band",
    ]),
    ("intent", "big_stone", "one large prominent centre diamond", ["bigstone"]),
    ("intent", "three_stone", "three stone design, a larger centre diamond with a smaller diamond on each side", [
        "three stone", "3 stone", "three stones", "trilogy", "three diamonds", "3 diamonds",
        "two smaller diamonds", "two small diamonds", "one on each side", "on each side",
        "તીન હીરા", "ત્રણ હીરા", "तीन हीरे", "तीन स्टोन",
    ]),
    ("intent", "prong", "stone held up by claw prongs", [
        "prong", "prongs", "four prong", "4 prong", "six prong", "6 prong", "claw", "claws", "prong setting",
    ]),
    ("intent", "bezel", "stone set in a smooth metal bezel rim", ["bezel", "bezel setting", "rubover", "rub over"]),
    ("intent", "channel", "diamonds set in a metal channel", ["channel", "channel set", "channel setting"]),
    ("intent", "split_shank", "split shank band that divides into two near the top", [
        "split shank", "split band", "double shank", "divides into two", "splits into two",
    ]),
    ("intent", "open_design", "open ended design with a gap between the two ends", [
        "open design", "gap", "adjustable", "open ended", "two ends", "open shank",
    ]),
    ("intent", "rows", "multiple rows of diamonds, layered design", [
        "rows", "multiple rows", "two rows", "double row", "three rows", "layers", "multiple layers", "layered",
        "multi row", "multi layer", "lines of diamonds", "कई लाइनों", "कई लाइन", "कई परत", "ઘણી લાઈન", "ઘણા પડ",
    ]),
    ("intent", "single_row", "a single row of diamonds", ["single row", "one row", "single line", "one line", "ek line"]),
    ("intent", "chandelier", "chandelier earrings with cascading layers of diamonds", ["chandelier", "chandeliers"]),
    ("intent", "openwork", "openwork design with cut-out gaps and filigree", [
        "openwork", "open work", "cut out", "cut outs", "cutout", "cutouts", "jali", "jaali", "filigree",
        "lattice", "hollow", "જાળી", "जाली",
    ]),
    ("intent", "vintage", "vintage design with milgrain beaded edges", [
        "vintage", "milgrain", "beaded edge", "beaded edges", "art deco", "retro",
    ]),
    ("intent", "medallion", "round medallion disc", ["medallion", "coin", "round disc", "disc"]),
    ("intent", "initial", "initial letter alphabet design", [
        "initial", "letter", "alphabet", "monogram", "name pendant", "અક્ષર", "अक्षर",
    ]),
    ("intent", "evil_eye", "evil eye motif", ["evil eye", "nazar", "nazariya", "નજર", "नज़र", "नजर"]),
    ("intent", "charm", "small charms hanging from the chain", ["charm", "charms"]),
    ("intent", "flat_top", "flat top signet style", ["flat top", "signet", "flat face"]),
    ("intent", "brushed", "brushed matte gold finish", ["brushed", "matte", "satin finish"]),
    ("intent", "butterfly", "butterfly motif", ["butterfly", "titli", "પતંગિયું", "तितली"]),
    ("intent", "star", "star motif", ["star", "stars", "tara", "તારો", "तारा"]),
    ("intent", "moon", "crescent moon motif", ["moon", "crescent", "chand", "ચાંદ", "चांद", "चाँद"]),
    ("intent", "bow", "bow and knot motif", ["bow", "knot", "ribbon"]),
    ("intent", "tennis", "tennis style, a continuous line of identical diamonds", [
        "tennis", "eternity", "ટેનિસ", "टेनिस",
    ]),
    ("intent", "cluster", "cluster of many small diamonds, fully studded", [
        "cluster", "lots of diamonds", "many diamonds", "full diamond", "full diamonds", "fully studded",
        "studded", "bharpur diamond", "ઘણા હીરા", "ભરપૂર હીરા", "बहुत सारे हीरे", "भरपूर हीरे",
    ]),
    ("intent", "diamond", "studded with sparkling diamonds", [
        "diamond", "diamonds", "heera", "hira", "heere", "cz", "american diamond", "stone", "stones", "nag",
        "sparkling", "sparkle", "sparkly", "shiny", "bling", "chamakdar", "chamkila",
        "ચમકદાર", "ચમકતી", "चमकदार", "चमकीला", "चमकीली",
        "હીરા", "હીરો", "હીરાનો", "હીરાની", "ડાયમંડ", "નંગ", "સ્ટોન", "हीरा", "हीरे", "हीरों", "डायमंड", "स्टोन", "नग",
    ]),
    ("intent", "coloured_stone", "with coloured gemstones", [
        "ruby", "emerald", "sapphire", "gemstone", "colour stone", "color stone", "colored stone",
        "coloured stone", "rangeen", "manek", "panna", "neelam",
        "માણેક", "પન્ના", "નીલમ", "રંગીન", "माणिक", "पन्ना", "नीलम", "रंगीन",
    ]),
    ("intent", "plain", "plain polished gold without any stones", [
        "plain gold", "only gold", "plain band", "plain", "સાદું સોનું", "सादा सोना",
    ]),
    # ---- band / size ------------------------------------------------------
    ("intent", "drop", "long dangling drop design", [
        "drop", "drops", "dangling", "dangler", "danglers", "latkan", "latakti", "latakta", "latakne", "hanging", "લટકણ", "લટકતી", "लटकन", "लटकने वाली",
    ]),
    ("intent", "thin", "very thin slim delicate band", [
        "thin", "slim", "sleek", "narrow", "fine", "thin width", "thin band", "slim band", "patli", "patla",
        "stackable", "stacking",
        "પાતળી", "પાતળું", "પાતળા", "पतली", "पतला",
    ]),
    ("intent", "wide", "wide broad thick band", [
        "wide", "broad", "thick", "wide band", "thick band", "chunky", "jadi", "jada",
        "જાડી", "જાડું", "પહોળી", "मोटी", "चौड़ी", "चौड़ा",
    ]),
    ("intent", "minimal", "simple minimal small delicate design", [
        "simple", "minimal", "minimalist", "sober", "light", "lightweight", "light weight", "small",
        "tiny", "delicate", "dainty", "subtle", "cute", "halka", "halki", "sadi", "saadi", "sadu", "chhoti", "nani",
        "નાની", "નાનું", "નાના", "સાદી", "સાદું", "સાદા", "હળવી", "હળવું", "હલકી",
        "हल्का", "हल्की", "हलका", "सादा", "सादी", "छोटा", "छोटी", "सिंपल",
    ]),
    ("intent", "ornate", "ornate intricate detailed design work, richly decorated", [
        "intricate", "ornate", "detailed", "design work", "heavy work", "heavy design", "fine work",
        "handwork", "nakshi", "nakashi", "nakkashi", "karigari", "kaam", "kaamwala", "jadtar", "jadau",
        "નકશી", "નક્શી", "જડતર", "જડાઉ", "કારીગરી", "ભારે કામ", "नक्काशी", "कारीगरी", "जड़ाऊ", "जड़ाऊ",
        "भारी काम",
    ]),
    ("intent", "statement", "big bold heavy statement design", [
        "heavy", "big", "bold", "statement", "grand", "large", "bhari", "bhaari", "moti", "bada",
        "ભારે", "મોટી", "મોટું", "મોટા", "भारी", "बड़ा", "बड़ी", "बडा",
    ]),
    # ---- occasion / wearer --------------------------------------------------
    ("intent", "everyday", "minimal lightweight everyday jewellery, low profile design", [
        "everyday", "every day", "daily", "daily wear", "regular wear", "regular", "office", "office wear",
        "wearable", "casual", "roj", "rojana", "rozana", "roz", "har din", "rojbroj",
        "રોજ", "રોજિંદા", "રોજિંદી", "દરરોજ", "રોજબરોજ", "ઓફિસ", "રોજ પહેરી",
        "रोज़", "रोज", "रोजाना", "रोज़ाना", "हर दिन", "ऑफिस", "रोज पहनने", "रोज़ पहनने",
    ]),
    ("intent", "bridal", "heavy ornate bridal statement jewellery with many diamonds, intricate design", [
        "bridal", "bride", "wedding", "marriage", "shaadi", "shadi", "lagna", "lagan", "dulhan",
        "લગ્ન", "લગન", "દુલ્હન", "વિવાહ", "શાદી", "शादी", "दुल्हन", "विवाह", "ब्राइडल",
    ]),
    ("intent", "engagement", "engagement ring with a sparkling centre diamond", [
        "engagement", "sagai", "proposal", "propose", "સગાઈ", "સગાઇ", "सगाई", "मंगनी",
    ]),
    ("intent", "party", "glamorous sparkling party jewellery", [
        "party", "party wear", "festive", "festival", "function", "occasion", "cocktail", "diwali",
        "પાર્ટી", "તહેવાર", "પ્રસંગ", "દિવાળી", "पार्टी", "त्योहार", "त्यौहार", "फंक्शन", "दिवाली",
    ]),
    ("intent", "floral", "floral flower motif design", [
        "floral", "flower", "flowers", "phool", "petal", "petals", "પાંખડી", "पंखुड़ी", "ફૂલ", "ફૂલવાળી", "ફ્લાવર", "फूल", "फूलों", "फ्लावर",
    ]),
    ("intent", "heart", "heart shaped design", [
        "heart", "hearts", "love", "dil", "દિલ", "હાર્ટ", "હૃદય", "दिल", "हार्ट",
    ]),
    ("intent", "infinity", "infinity symbol design", ["infinity", "ઇન્ફિનિટી", "इन्फिनिटी"]),
    ("intent", "leaf", "leaf motif design", ["leaf", "leaves", "patta", "પાન", "પાંદડું", "पत्ती", "पत्ता"]),
    ("intent", "twisted", "twisted intertwined crossover band", ["twisted", "twist", "crossover", "criss cross", "intertwined"]),
    ("intent", "geometric", "modern geometric design", ["geometric", "square", "triangle", "hexagon", "angular"]),
    ("intent", "traditional", "traditional indian ethnic ornate design", [
        "traditional", "ethnic", "temple", "antique", "kundan", "desi", "indian",
        "પરંપરાગત", "ટ્રેડિશનલ", "એન્ટિક", "पारंपरिक", "ट्रेडिशनल", "एंटीक",
    ]),
    ("intent", "modern", "modern contemporary design", [
        "modern", "contemporary", "trendy", "stylish", "western", "fashion",
        "મોડર્ન", "સ્ટાઇલિશ", "ટ્રેન્ડી", "मॉडर्न", "स्टाइलिश", "ट्रेंडी",
    ]),
    ("intent", "men", "bold masculine men's jewellery", [
        "men", "mens", "men's", "gents", "gent's", "male", "man", "husband", "father", "dad", "papa",
        "brother", "boyfriend", "bhai",
        "પુરુષ", "પુરુષો", "જેન્ટ્સ", "પપ્પા", "પિતા", "પતિ", "ભાઈ", "ભાઇ",
        "पुरुष", "जेंट्स", "पापा", "पिता", "पति", "भाई", "मर्द",
    ]),
    # who it is for / why: filters only (men's series excluded or kept), no
    # look of their own, so no phrase for the image model
    ("intent", "women", None, [
        "women", "woman", "ladies", "lady", "female", "mother", "mom", "mummy", "mum", "maa", "wife",
        "sister", "girlfriend", "daughter", "didi", "behen",
        "મમ્મી", "માતા", "મમ્મીને", "પત્ની", "બહેન", "દીકરી", "લેડીઝ",
        "माँ", "मां", "मम्मी", "माता", "पत्नी", "बहन", "बेटी", "लेडीज़", "लेडीज",
    ]),
    ("intent", "kids", "small cute jewellery for children", [
        "kid", "kids", "child", "children", "baby", "બાળક", "બાળકો", "बच्चे", "बच्चा", "बच्ची",
    ]),
    ("intent", "gift", None, [
        "gift", "present", "anniversary", "birthday", "ભેટ", "ગિફ્ટ", "गिफ्ट", "तोहफा", "उपहार",
    ]),
]

# Words that name a type AND a form of it ("hoop" = earrings, hoop-shaped).
# term -> (intent, phrase for the image model)
_HOOP = ("hoop", "round hoop earrings")
_STUD = ("stud", "small stud earrings on the earlobe")
_JHUMKA = ("jhumka", "long dangling drop earrings")
_BANGLE = ("bangle", "rigid round bangle")
SUBTYPES = {
    **dict.fromkeys(["hoops", "hoop", "huggie", "huggies", "kundal", "કડી", "कुंडल"], _HOOP),
    **dict.fromkeys(["studs", "stud", "tops", "टॉप्स"], _STUD),
    **dict.fromkeys(["jhumka", "jhumki", "jhumkas", "ઝુમખા", "ઝુમકા", "झुमका", "झुमकी", "झुमके"], _JHUMKA),
    **dict.fromkeys(["bangle", "bangles", "kada", "kadu", "kangan", "bangdi", "કડું", "કડા", "બંગડી",
                     "બંગડીઓ", "कंगन", "कड़ा", "कडा", "चूड़ी", "चूड़ियां"], _BANGLE),
    "cuff": ("cuff", "open cuff bracelet"),
    **dict.fromkeys(["chain bracelet", "hand chain"], ("flexible", "flexible chain bracelet")),
    **dict.fromkeys(["choker", "ચોકર", "चोकर"], ("choker", "short choker necklace")),
    **dict.fromkeys(['rani haar', 'rani har', 'rani haaar', 'raani haar', 'long haar', 'layered necklace', 'multi layer necklace', 'multilayer necklace', 'layered haar', 'રાણી હાર', 'રાની હાર', 'લાંબો હાર', 'रानी हार', 'रानीहार', 'लंबा हार'],
                    ("rani_haar", "long heavy multi-layered bridal necklace covered in diamonds")),
    **dict.fromkeys(["chain", "ચેન", "चेन", "mangalsutra", "મંગળસૂત્ર", "मंगलसूत्र"],
                    ("pendant_chain", "thin chain necklace with a small pendant")),
}

# Stone cut: matches the design_id suffix (see search.family_and_shape).
SHAPES = {
    "oval": ["oval", "ઓવલ", "અંડાકાર", "ओवल", "अंडाकार"],
    "pear": ["pear", "drop shape", "drop shaped", "teardrop", "boond", "પિયર", "ટીપા", "ટીપું", "ટીપા જેવો",
             "पियर", "बूंद"],
    "baguette": ["baguette", "baguettes", "बैगेट", "બેગેટ"],
    "emerald": ["emerald cut", "એમરાલ્ડ કટ", "एमराल्ड कट"],
    "princess": ["princess", "princess cut", "square diamond", "પ્રિન્સેસ", "प्रिंसेस"],
    "cushion": ["cushion", "cushion cut", "કુશન", "कुशन"],
    "marquise": ["marquise", "માર્કિઝ", "मार्कीज़"],
    "radiant": ["radiant", "radiant cut"],
    "asscher": ["asscher", "asscher cut"],
    "round": ["round", "round cut", "round diamond", "gol", "ગોળ", "राउंड", "गोल"],
}
LEXICON += [("shape", s, f"{s} cut centre stone", terms) for s, terms in SHAPES.items()]

# Negation cues -> directions to look for the negated term, in order of
# preference: ">" = the next matched term, "<" = the previous one.
NEG_CUES = {
    "no more": ">", "not any": ">", "do not want": ">", "don't want": ">", "dont want": ">", "no": ">",
    "not": ">", "without": ">", "avoid": ">", "except": ">", "never": ">", "zero": ">", "free of": ">",
    "bina": "><", "bagair": "><", "baghair": "><", "nahi": "<>", "nahin": "<>", "nai": "<", "free": "<",
    "vagar": "<", "vagar ni": "<", "na joie": "<",
    "વગર": "<", "વગરની": "<", "વગરનું": "<", "વગરના": "<", "નહીં": "<", "નહિ": "<", "ના જોઈએ": "<",
    "के बिना": "<", "बिना": "><", "नहीं": "<", "नहीं चाहिए": "<", "मत": "<",
    "na ho": "<", "nahi ho": "<", "na hoy": "<", "no hoy": "<", "ના હોય": "<", "ન હોય": "<", "न हो": "<",
    "नहीं हो": "<",
}
CONNECTORS = {"or", "and", "nor", "ya", "aur", "ane", "ke", "અને", "કે", "या", "और", "or any", "any"}
NEG_GAP = 3  # max words between a cue and the term it negates

WEAK_CATEGORY_TERMS = {"chain", "ચેન", "चेन"}
STONE_INTENTS = {"diamond", "solitaire", "halo", "side_stones", "cluster", "tennis", "three_stone",
                 "coloured_stone", "big_stone", "single_row", "rows", "channel", "prong", "bezel"}

CATEGORY_NOUN = {
    "ring": "ring", "earrings": "pair of earrings", "pendant": "pendant",
    "necklace": "necklace", "bracelet": "bracelet",
}
STOPWORDS = {
    "i", "a", "an", "the", "for", "my", "me", "want", "need", "that", "is", "which", "can", "be", "of", "to",
    "in", "on", "at", "with", "and", "or", "design", "designs", "show", "some", "please", "any", "only",
    "chahiye", "ke", "ka", "ki", "liye", "mein", "koi", "type", "wala", "wali", "vala", "vali",
    "jewellery", "jewelry", "looking", "like", "would", "it", "should", "joie", "joiye", "mate", "also",
    "very", "more", "less", "something", "piece", "her", "his", "their", "him", "our", "wear", "she", "he",
    "width", "look", "looks", "have", "has", "just", "but", "one", "who", "are", "was", "this", "these",
    "gold", "sona", "metal", "get", "give", "find", "from", "as", "so", "too", "all", "set", "center",
    "centre", "middle", "single", "shape", "shaped", "design", "style", "around", "else", "nothing",
    "pehne", "pehni", "pehenne", "costly", "looking", "him",
}
_LATIN = re.compile(r"[a-z]")


def _compile():
    entries = []
    for kind, value, phrase, terms in LEXICON:
        for t in terms:
            t = t.lower()
            if _LATIN.search(t):
                rx = re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])")
            else:
                # Indic scripts: substring match (\b is unreliable with
                # combining vowel signs); longest-first ordering limits noise.
                rx = re.compile(re.escape(t))
            entries.append((len(t), kind, value, phrase, t, rx))
    entries.sort(key=lambda e: -e[0])
    return entries


_ENTRIES = _compile()
_VOCAB = sorted({w for _, _, _, _, t, _ in _ENTRIES for w in t.split() if _LATIN.fullmatch(w[0]) and w.isalpha()}
                | STOPWORDS | {w for c in NEG_CUES for w in c.split() if w.isascii()})
_VOCAB_SET = set(_VOCAB)


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Damerau-Levenshtein (optimal string alignment), early exit above limit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


_ENGLISH = set((Path(__file__).with_name("english_words.txt")).read_text().split())


def _is_english(word: str) -> bool:
    """Real words (incl. simple inflections) are never "corrected": the
    dictionary lacks most plurals and verb forms, so strip common endings."""
    if word in _ENGLISH:
        return True
    for suf, add in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
                     ("ly", ""), ("er", ""), ("est", "")):
        if word.endswith(suf) and len(word) - len(suf) >= 3 and word[:-len(suf)] + add in _ENGLISH:
            return True
    return False


def correct_spelling(word: str) -> str:
    """Map a misspelt Latin word to the closest lexicon word, conservatively."""
    if len(word) < 4 or word in _VOCAB_SET or not word.isalpha() or _is_english(word):
        return word
    limit = 1 if len(word) <= 5 else 2
    best, best_d = word, limit + 1
    for v in _VOCAB:
        if v[0] != word[0] or len(v) < 4:
            continue
        d = _edit_distance(word, v, limit)
        if d < best_d:
            best, best_d = v, d
    return best


@dataclass
class ParsedQuery:
    raw: str
    normalised: str = ""
    corrections: dict = field(default_factory=dict)
    category: str | None = None          # main type (names the piece for the model)
    categories: list[str] = field(default_factory=list)   # every type asked for
    exclude_categories: list[str] = field(default_factory=list)
    metal: str | None = None
    exclude_metals: list[str] = field(default_factory=list)
    metal_note: str | None = None
    shape: str | None = None
    intents: list[str] = field(default_factory=list)       # wanted
    not_intents: list[str] = field(default_factory=list)   # unwanted
    phrases: list[str] = field(default_factory=list)
    neg_phrases: list[str] = field(default_factory=list)
    leftover: str = ""
    terms: list = field(default_factory=list)   # (term as matched, kind, value, negated), in prompt order

    def visual_text(self) -> str:
        # Metal is deliberately left out: it is applied as a filter from the
        # file names, and designs are embedded in one metal colour, so a colour
        # word here would favour whichever designs happen to share that colour.
        noun = CATEGORY_NOUN.get(self.category, "piece of jewellery")
        return ", ".join([f"a photo of a {noun}"] + self.phrases)

    def base_text(self) -> str:
        """The generic part of visual_text(), without anything specific."""
        return f"a photo of a {CATEGORY_NOUN.get(self.category, 'piece of jewellery')}"

    def negative_text(self) -> str | None:
        if not self.neg_phrases:
            return None
        noun = CATEGORY_NOUN.get(self.category, "piece of jewellery")
        return ", ".join([f"a photo of a {noun}"] + self.neg_phrases)

    def free_text(self) -> str | None:
        """Unmatched, non-negated Latin words, passed to the image model as-is."""
        words = [w for w in re.findall(r"[a-z]+", self.leftover) if w not in STOPWORDS and len(w) > 2]
        return " ".join(words) or None


def bank_queries(category: str) -> list[str]:
    """Generic queries for one type: the type alone, with each style phrase,
    and with pairs of phrases. Used to measure how "hub-like" a design is."""
    noun = CATEGORY_NOUN[category]
    phrases = sorted({p for kind, _, p, _ in LEXICON if kind in ("intent", "shape") and p})
    out = [f"a photo of a {noun}"] + [f"a photo of a {noun}, {p}" for p in phrases]
    out += [f"a photo of a {noun}, {a}, {b}" for i, a in enumerate(phrases) for b in phrases[i + 1::7]]
    return out


_BIG_WORDS = r"(?:big|bigger|biggest|large|larger|huge|bada|bade|badi|bada sa|mota|moto|મોટો|મોટા|મોટી|बड़ा|बड़े|बड़ी|बडा)"
_STONE_WORDS = (r"(?:diamond|diamonds|stone|stones|hira|heera|solitaire|round|oval|pear|princess|cushion|emerald|"
                r"marquise|હીરો|હીરા|ગોળ|हीरा|हीरे|गोल)")
# a size word directly before a stone (optionally with one word between)
_BIG_STONE = re.compile(r"(?<![a-z])" + _BIG_WORDS + r"(?![a-z])(?=\s+(?:\S+\s+)?" + _STONE_WORDS + r")")


def _normalise(prompt: str) -> str:
    t = prompt.lower().replace("’", "'")
    t = re.sub(r"[+&(),.;:!?\"/\\\-_|*]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _words_between(text: str, a: int, b: int) -> list[str]:
    return text[a:b].split()


def _apply_negation(text: str, matches: list[dict]) -> str:
    """Mark matches negated by a cue; returns text with the cues blanked out."""
    matches.sort(key=lambda m: m["start"])
    for cue in sorted(NEG_CUES, key=len, reverse=True):
        pat = (r"(?<![a-z])" + re.escape(cue) + r"(?![a-z])") if cue.isascii() else re.escape(cue)
        for c in list(re.finditer(pat, text)):
            if text[c.start():c.end()].strip() != cue:   # already blanked by a longer cue
                continue
            for d in NEG_CUES[cue]:
                if d == ">":
                    after = [m for m in matches if m["start"] >= c.end()
                             and len(_words_between(text, c.end(), m["start"])) <= NEG_GAP]
                    if not after:
                        continue
                    chain = [after[0]]
                    for m in after[1:]:
                        gap = set(_words_between(text, chain[-1]["end"], m["start"]))
                        stones = {chain[-1]["value"], m["value"]} <= STONE_INTENTS
                        # "no stones or diamonds"; adjacent only for one
                        # stone description ("no big centre stone")
                        if (gap and gap <= CONNECTORS) or (not gap and stones):
                            chain.append(m)
                        else:
                            break
                else:
                    before = [m for m in matches if m["end"] <= c.start()
                              and len(_words_between(text, m["end"], c.start())) <= 1]
                    if not before:
                        continue
                    chain = [before[-1]]
                for m in chain:
                    m["neg"] = True
                break
            text = text[:c.start()] + " " * (c.end() - c.start()) + text[c.end():]
    return text


def parse(prompt: str) -> ParsedQuery:
    q = ParsedQuery(raw=prompt)
    fixed = []
    for w in _normalise(prompt).split(" "):
        c = correct_spelling(w) if w.isascii() else w
        if c != w:
            q.corrections[w] = c
        fixed.append(c)
    q.normalised = " ".join(fixed)
    text = " " + _BIG_STONE.sub(" bigstone ", q.normalised) + " "

    # 1. lexicon matches, longest first, each consuming its span
    matches = []
    for _, kind, value, phrase, term, rx in _ENTRIES:
        m = rx.search(text)
        while m:
            matches.append({"start": m.start(), "end": m.end(), "kind": kind, "value": value,
                            "phrase": phrase, "neg": False, "term": term})
            text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]
            m = rx.search(text)
    # 2. negation, measured on the text with matches blanked (only filler
    #    words remain between a cue and its term)
    text = _apply_negation(text, matches)
    q.terms = [(m["term"], m["kind"], m["value"], m["neg"]) for m in sorted(matches, key=lambda m: m["start"])]

    cats, xcats, metals, xmetals, weak = [], [], [], [], []
    wanted, unwanted = {}, {}
    for m in matches:
        kind, value, phrase, is_neg = m["kind"], m["value"], m["phrase"], m["neg"]
        if kind == "category":
            if m["term"] in WEAK_CATEGORY_TERMS and not is_neg:
                weak.append(value)
                continue
            (xcats if is_neg else cats).append(value)
            sub = SUBTYPES.get(m["term"])
            if sub and not is_neg:
                wanted.setdefault(sub[0], sub[1])
        elif kind == "metal":
            if value.endswith("~silver"):
                value = value.split("~")[0]
                if not is_neg:
                    q.metal_note = "The catalogue has no silver; white gold is the closest look."
            (xmetals if is_neg else metals).append(value)
        elif kind == "shape":
            if not is_neg:
                q.shape = q.shape or value
                wanted.setdefault("shape:" + value, phrase)
        else:
            (unwanted if is_neg else wanted).setdefault(value, phrase)

    # "chain" only names a necklace when nothing else is named: "drop earrings
    # on a thin chain", "heart pendant on a chain" are not necklaces
    if not cats and weak:
        cats = weak
        wanted.setdefault(*SUBTYPES["chain"])
    # "wide men's band", "eternity band": a band with no other type is a ring
    if not cats and re.search(r"(?<![a-z])bands?(?![a-z])", q.normalised):
        cats = ["ring"]
    # several types may be asked for ("pendant or earrings"); the most
    # mentioned one names the piece in the model text
    q.categories = sorted(set(cats), key=lambda c: (-cats.count(c), cats.index(c)))
    q.category = q.categories[0] if q.categories else None
    q.exclude_categories = sorted(set(xcats) - set(q.categories))
    q.metal = max(set(metals), key=metals.count) if metals else None
    q.exclude_metals = sorted(set(xmetals) - {q.metal})

    # Stone logic. "single diamond ... no more diamonds" = a solitaire without
    # side stones, not a ring without any stone.
    if "three_stone" in wanted:
        wanted.pop("solitaire", None)    # "a bigger diamond in the middle" + two sides
    # "one big diamond" with no other stone layout described is a solitaire
    if "big_stone" in wanted and not {"halo", "three_stone", "side_stones", "cluster", "rows", "tennis"} & set(wanted):
        wanted.setdefault("solitaire", "solitaire ring with one single centre diamond")
    has_stones = any(k in STONE_INTENTS or k.startswith("shape:") for k in wanted)
    if "diamond" in unwanted:
        del unwanted["diamond"]
        if has_stones:   # "a centre diamond, no stones on the sides"
            unwanted.setdefault("side_stones", "many small diamonds set along the band, pave")
        else:
            wanted.setdefault("plain", "plain polished gold without any stones")
    if "plain" in wanted and has_stones:
        # "a row of diamonds on top, plain gold underneath", "a centre diamond
        # on a plain band": plain describes a part, not a piece without stones
        del wanted["plain"]
    if "solitaire" in wanted or "plain" in wanted:
        wanted.pop("diamond", None)      # already implied / contradicted
    if "plain" in wanted:
        unwanted.setdefault("diamond", "studded with diamonds and gemstones")
    if "thin" in wanted and "wide" in wanted:
        del wanted["wide"]
    if q.category == "ring" and "men" in wanted:
        wanted["men"] = "bold wide men's band ring"

    q.intents = [k for k in wanted if not k.startswith("shape:")]
    q.not_intents = list(unwanted)
    q.phrases = [p for p in wanted.values() if p]
    q.neg_phrases = [p for p in unwanted.values() if p]
    q.leftover = re.sub(r"\s+", " ", text).strip()
    return q
