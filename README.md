# Design Finder

Type what you want in English, Hindi, Gujarati or Hinglish ("a rose gold ring
for my mother, wearable every day"), or add a photo of a piece, and get the 8
closest designs from the jewellery library (an S3 bucket, see [Dataset storage](#dataset-storage-s3)). Search runs on this Mac and every
component is open source.

## How a search works

1. **Understand the prompt** (`jewelsearch/query.py`). A trilingual lexicon
   extracts type, metal, stone shape and intents (solitaire, thin band,
   everyday, bridal, for men, ...). Misspelt English/Hinglish words are
   corrected ("dimond", "bracelate"), and negation is understood ("no side
   stones", "bina stone", "હીરા વગર", "हीरे के बिना"). Type and metal are never
   guessed by a model. Silver/platinum map to white gold with a note.
2. **Filter** to the type, metal and gender asked for, then on design
   attributes: thin / wide band and forms (stud, bangle, choker, ...) are strict,
   "no stones" uses the measured stone share, unwanted features (e.g. side
   stones) are removed. The diamond layout (solitaire, centre + side stones,
   many small diamonds) is strict too, read by the trained diamond reader from
   every render of every design. Checked against the job cards and CAD files
   on 11 layout searches (pages 1 and 2), the asked layout went from 63% of the
   results to 97% ("with side stones": from none to 95%). The zero-shot
   "solitaire" filter it replaced kept 1,039 designs, only 39.5% of them solitaires.
   When a filter would leave too few designs it relaxes, and the page always says
   so in the shopper's own word, for example "Too few designs here are clearly
   “पतली” to keep only those, so it orders the results instead." A filter is never
   relaxed silently (`tests/test_search.py` checks every prompt file). Today 5 of
   239 test prompts get such a note.
3. **Rank** by similarity between the prompt and each design's vectors
   (SigLIP2 `ViT-B-16-SigLIP2-256`, Apache 2.0) plus attribute match.
   Unwanted features are subtracted as a negative text vector.
4. **Diversify** with MMR so the 8 results are related but not near-copies;
   stone-cut variants of one design (`DDLR-092-AC`, `-CU`, `-EM`, ...) appear
   once and are listed in the detail view instead.

Cards show each design's front view, tightly cropped. "More like this" finds
the nearest designs of the same type by image alone.

## Manual search

The **Manual search** tab above the search box replaces the words with picks
(`jewelsearch/manual.py`, `POST /api/manual`). One choice per group:

| Group | Options | How it acts |
|---|---|---|
| Jewellery type | ring, earrings, pendant, necklace, bracelet | filter |
| Metal colour | yellow, white, rose gold | filter |
| For | women, men (filter), kids (AI) | |
| Diamonds & gemstones | with diamonds, no stones, solitaire, centre + side stones, many small diamonds, coloured gemstones | no stones, solitaire, side stones, many diamonds: filter; others AI |
| Diamond setting | halo, three stone, tennis line, rows, single row, bezel, prong, channel | AI |
| Centre stone cut | round, oval, pear, princess, cushion, emerald, marquise, radiant, asscher | ranked first |
| Shape (after a type) | rings: thin, wide, split shank, signet, open ended; earrings: stud, hoop, drop, chandelier, jhumka; bracelets: bangle, cuff, flexible chain; necklaces: choker, rani haar | thin, wide and forms: filter; others AI |
| Style & motif | floral, heart, infinity, leaf, twisted, geometric, butterfly, star, moon, bow, evil eye, initial, traditional, modern | AI |
| Look & occasion | minimal, statement, ornate, everyday wear, bridal, engagement, party | AI |
| Finish | matte / brushed, milgrain edges, openwork / filigree | AI |

**The picks become the prompt the AI search already understands**, for example
"rose gold oval solitaire thin rings for women". So both tabs share the same filters,
ranking and notes, a manual search lands in the search history like a typed one, and
**Edit in words** moves it to the AI tab to refine. Every option's words were chosen so
the prompt parses back to exactly the picks; `tests/test_manual.py` checks every option
and every pair of options (over 2,000 pairs).

- **filter**: designs without it are left out.
- **AI** (✦): the image model ranks designs by how much they show it. These are read
  from the renders, so no design is left out for them and no number is shown.
- **Centre stone cut**: designs with the cut are ranked first. The number is the designs
  known to have it, named in their id or read clearly from their renders. Both sources
  were right every time they could be checked against job cards and CAD files.

**Every number is strict**: how many designs pass all filters with that option picked,
none loosened. Options no design has with the other picks are greyed out. So are
contradictions: "No stones" with any setting or cut, and a solitaire with a many-diamond
setting. A pick that later picks rule out (oval first, then thin) is flagged under the
button. The numbers come in 6 to 11 ms. The picks are kept in the page's memory only.

Gold purity (touch: 14K, 18K, 22K) is not a filter: every design is made in each, and
it is chosen when ordering (`/buy`).

## Suggested prompts under the search bar

The 5 prompts under the search bar are chosen for each shopper by the server
(`jewelsearch/recommend.py`, `GET /api/prompt-suggestions`). They are no longer a
fixed list. Each chip has an icon for its kind, and its tooltip says why it is there.

| Kind | Icon | What it is |
|---|---|---|
| Search again | clock | The shopper's latest search and their most frequent ones. Two of them, or three for someone who often repeats searches. |
| For you | sparkle | Their favourite type, metal and looks, the type they are looking at now, a look they like in another type, and the next refinement of what they search |
| Popular | trend arrow | What other shoppers searched in the last 30 days |
| Explore | compass | Well-stocked types and looks of the collection |

**What it learns from:** the shopper's search history, weighted by recency with a
14-day half-life and by repeats, and their saved designs, both the type and look of
each and its picture. It also uses everyone's recent searches and the collection itself.
Searches that were refused as not jewellery are ignored.

**How it picks:**
1. **Read the meaning.** Every search is read by the search's own parser: type,
   metal, looks, stone cut and language.
2. **Write candidates.** New prompts are written from those meanings in the
   shopper's script: English, ગુજરાતી or हिन्दी. Each one is parsed back and must
   mean exactly that. It must also fill a page of designs without loosening a
   filter, the same rule the dropdown suggestions follow.
3. **Score them.** Candidates are embedded with the image model's text encoder and
   scored on six things: how much of the shopper's profile they match, the type
   they are looking at now, similarity to their own searches, how strongly they
   pick out the designs they saved among all designs, popularity, and how many
   designs they find.
4. **Pick five.** The latest search goes first. At most 1 popular and 1 explore
   prompt are shown, and at least 1 of the two when the shopper has history. At most
   3 chips share a type, no two chips mean the same thing, and the rest are picked
   by score while staying different from each other. A new shopper gets popular and
   explore prompts, one per type.

**Privacy:** other shoppers' searches are read as meaning only, meaning type, metal,
looks and cut. The database query never selects the words anyone typed. A search
counts as popular only when at least 2 other people made it.

**The same suggestions are also used** in three places: the empty search box
dropdown, under "Recent searches", the example buttons shown after a refused
search, and the animated placeholder. Answers are cached per shopper until they
search again or save a design. Other shoppers' searches are read again every 10
minutes. A shopper's suggestions take about 50 to 250 ms the first time, then a few
ms.

**Measured** with `scripts/eval_prompt_suggestions.py` on 80 simulated shoppers,
with 8 searches each and the last one held out:

| Shown under the bar | Next search among them | Repeat searches | New searches |
|---|---|---|---|
| These suggestions | 65% | 85% | 36% |
| The old fixed examples | 10% | 4% | 18% |
| The shopper's own last 5 searches | 72% | 100% | 33% |
| The most popular searches only | 11% | 9% | 15% |

These suggestions were in the shopper's script 100% of the time, and showed 2.6
different types on average among the 5. The simulated shoppers follow the same
kind of taste the recommender assumes, so these numbers are a sanity check, not a
field result. Repeating the last 5 searches catches more repeats, but it never
suggests anything new or from other shoppers. The chips give up some of that for
discovery and still predict new searches better. Tests: `tests/test_recommend.py`.

## Only jewellery searches

A search that isn't about jewellery gets no designs, only a message such as
**Can't find designs like “pizza”**, with real searches to try instead. Before
this, every prompt returned 8 designs, even "weather today". The check is in
`jewelsearch/domain.py`. It runs before every text search, cheapest step first:

1. **The search's own words.** A prompt with no jewellery word the search knows is
   refused, for example "pizza", "hello", "આજનું હવામાન" or "123456". A prompt made
   only of words it knows is accepted without any model, for example "for my
   mother", "wedding" or any Gujarati or Hindi prompt it understands. About 3 in 4
   real prompts are decided here.
2. **A local judge** reads prompts that mix known words with unknown ones, such as
   "boxing ring", "lehenga for wedding" or "Plain gold band for daily use". It is
   Qwen3-1.7B, an Apache 2.0 language model that runs on this Mac. It answers with
   one of three words: yes, other (a kind of jewellery the collection doesn't have,
   such as an anklet), or no. It doesn't know Hindi or Gujarati names of pieces,
   so "haar" or "bali" are put in English for it first. It takes about 0.4 to 0.9 s
   per new prompt, and answers are cached.
3. **A picture check** follows when the judge says yes. The image model scores the
   whole prompt as a caption against every design here. If the best match falls
   far below a plain "a photo of a ring", the prompt is refused. This catches the
   judge's mistakes, such as "mehndi design for bride" or "necklace hanger stand".

**A budget is noted, not refused.** "rose gold ring under 30000" shows rose gold
rings with the note that prices aren't in the catalogue yet. "18k" and "22k" are
read as gold purity, never as a budget. A typed design id such as "DDLR-092" is
looked up directly.

**With a photo,** words that aren't about jewellery are left out with a note, and
the photo alone is used. Suggestions under the box are never offered for a prompt
with unknown words.

Measured with `scripts/eval_domain.py` on the lists in `tests/`:

| Set | Right |
|---|---|
| Jewellery prompts, must be accepted | 229 / 229 |
| Not-jewellery prompts, must be refused | 178 / 188 |
| Not-jewellery, the three sets written later | 80 / 86 |

The judge's instructions were written against a tuning set. The other sets were
written later, each before its first run. The picture threshold, `PICTURE_MIN`,
was chosen after seeing all of them, with a margin on the jewellery side.

The misses are double meanings that both checks accept, such as "key chain",
"ring cake design" or "chain stores in india". Hindi or Gujarati prompts where
the unknown words change the meaning also slip through, such as "शादी का जोड़ा"
(a wedding outfit). Unknown Gujarati and Hindi words are treated as filler, because
the judge reads those scripts poorly and refused real Gujarati prompts in tests.

The judge needs about 3.5 GB of memory and loads when the server starts, adding
about 7 s. Without it, the picture check works alone. Tests: `tests/test_domain.py`.

## Suggestions while typing

The search box suggests searches as you type, like a web search engine
(`jewelsearch/suggest.py`, `GET /api/suggest`). Up to ten rows appear under the box:

1. **What you typed.** Enter or a click runs it.
2. **Your own past searches** that match, with a clock icon and **Remove**. They come from
   your search history in the database. Another user's searches are never shown, because
   prompts can name family members or occasions.
3. **Completions** of the word being typed: "bold ri" becomes "bold rings", "rose g" becomes
   "rose gold", "rings for " becomes "rings for men". A half-typed word is finished even
   when it adds no meaning: "hoop" already means hoop earrings, but "hoop ear" still becomes
   "hoop earrings". Only correctly spelt words the search understands are offered. A misspelt prompt gets its corrected form first
   ("bracelate for party" becomes "bracelet for party").
4. **Refinements**: the prompt plus one more thing, written in your script (English,
   ગુજરાતી or हिन्दी). Each kind of refinement is skipped when the prompt already says it.
   A prompt that names no type ("floral") gets the types only.

| Refinement | English | ગુજરાતી | हिन्दी |
|---|---|---|---|
| Type | floral **rings** | ફૂલ **વીંટી** | फूल **अंगूठी** |
| Wearer | bold rings **for men** | વીંટી **પુરુષો માટે** | अंगूठी **पुरुषों के लिए** |
| Metal | **rose gold** bold rings | **રોઝ ગોલ્ડ** વીંટી | **रोज़ गोल्ड** अंगूठी |
| Stones | **solitaire** rings, rings **without stones** | **સોલિટેર** વીંટી | **सॉलिटेयर** अंगूठी |
| Form | **thin** rings, **stud** earrings, **cuff** bracelets, **choker** necklaces | **પાતળી** વીંટી | **पतली** अंगूठी |
| Occasion | rings **for everyday wear**, **for wedding** | વીંટી **લગ્ન માટે** | अंगूठी **शादी के लिए** |
| Stone cut | **oval** rings (only cuts enough designs have) | **ઓવલ** વીંટી | **ओवल** अंगूठी |

**Every suggestion is checked against the collection before it is shown.** It is parsed
exactly like a search and must keep everything the prompt asked for and add what it claims.
It must also fill at least a full page (8 designs) without any filter being relaxed. So a
suggestion never leads to an empty page or to a "showing other metals" note. "For men" is
offered only for rings and bracelets, "without stones" only where enough plain designs
exist, and a stone cut only where 8 or more designs have it. Today that means rings only.

A row shows its number of designs when it narrows the prompt by more than 10%, for example
"bold rings for men · 87 designs". Rows that only change the order of results, such as an
occasion or a stone cut, show no number.

**With a photo,** the words refine the photo. Suggestions show no counts and no type rows,
and they are checked against the photo's type when the reading is sure. A photo of earrings
gets "stud" and "drop", never "for men".

Keys: ↓ and ↑ move through the rows and put each one in the box. Enter searches, and Esc
restores what you typed. Each suggestion takes about 3 to 12 ms on the server, and answers
are cached in server memory. The page keeps them in memory too, never in browser storage.
Tests: `tests/test_suggest.py`.

## Browse all categories (no prompt)

**Browse all categories**, under the search box, is for visitors without a particular
request. It shows a tile per category (rings, earrings, pendants, necklaces, bracelets) with
its number of designs. A tile lists every design of that category, 24 at a time, with the
**richest pieces first** (measured stone share), one design per stone-cut family, and
look-alike designs spaced apart. Links work too: `/?browse=all`, `/?browse=necklace`.

Browsing deliberately skips the search's hubness correction. That correction pushes the
classic, fully set pieces down, which is why plain prompts such as "necklace" show lighter
designs (see `SearchEngine.browse_order`).

## Search by a product link

Paste a product page link into the search box, with words if you like, such as
"similar in white gold https://…", and press Find. The jewellery picture on that page
is found and matched exactly like a photo you uploaded. You get the Design DNA, the
closest designs, filters and "Show 8 more" (`jewelsearch/linksearch.py`,
`POST /api/link-search`). No paid search or image API is used. Everything runs on
this Mac with the same open models as the rest of the search.

1. **Fetch the page safely.** A link straight to a picture is used as it is.
2. **Collect its pictures, likeliest first.** The product image in the page's
   structured data (JSON-LD) comes first. Then Open Graph and Twitter images,
   `itemprop="image"` and `<link rel="image_src">`. Last come `<img>` and `<picture>`
   elements, including lazy-loading attributes and the largest `srcset` size. Logos,
   icons, banners and tracking pixels are skipped.
3. **Let the image model choose.** Up to 8 pictures are downloaded and read like an
   upload. They are downloaded before any model runs: when other shoppers are searching, a
   model waits its turn on the GPU, and that wait must not use up the link's time. The one that looks most like jewellery and closest to a design of the
   collection wins.
   On a page whose words, and yours, never mention jewellery, the picture must look
   like a product photo. Its jewellery reading must be at least 0.9, and its
   closeness to the nearest design at least 0.75. Shop product photos measured
   1.00 and 0.79 to 0.90. News, sari and watch pages with jewellery worn in a photo
   reached at most 0.69 closeness. A picture that clears that bar is used without
   reading the page's words at all: the jewellery check runs once per name, title and
   link, and took 25 of 32 s on a shop page while the GPU was busy.
4. **No usable picture?** This happens when the shop blocks automatic visits, builds
   its pictures with JavaScript, or shows none. The jewellery words of your text, the
   product's name, the page title and the link itself are searched instead, for
   example ".../diamond-stud-earring-in-14k" becomes "diamond stud earring". Those
   words must pass the jewellery check, so "Boxing ring - Wikipedia" is not searched.
5. **Nothing at all?** You get **Couldn't find a jewellery design in that link**,
   with the reason, a **Search with a photo** button and prompts to try.

Tried on real pages:

| Result | Pages |
|---|---|
| Picture found and matched | ORRA, Mejuri, Reeds, 1stDibs, Wikipedia "Engagement ring" and "Bangle" |
| Shop blocked automatic visits, link words used | Etsy, Pandora, Candere |
| Refused, not jewellery | BBC News, Wikipedia "Sari", "Boxing ring", "Cat" |

A link takes about 4 to 15 s, mostly the other website. The linked picture is kept
like an uploaded photo: in memory for an hour, never on disk, never in the search
history. The page shows a copy made by the server, never the other site's picture.

**Safety.** Fetching a link a user chose could reach inside this Mac's network
(SSRF), so the fetcher has these limits:
- Only http and https links on ports 80 and 443, and no user name or password in
  the link.
- Every host is resolved, with a 5 s limit, and all of its addresses must be public.
  This is checked again on every redirect, with at most 4 redirects.
- The connection goes to the address that was checked, so DNS rebinding can't
  switch it. The certificate is still checked for the real name.
- Pages are cut at 3 MB and pictures at 8 MB, both after decompression. There are
  time limits, at most 25 s per link, and 20 links per user per 10 minutes.

Tests: `tests/test_linksearch.py` replaces name lookups and web requests, so no test
reaches the internet.

## From the web panel (demo)

A right-side panel shows similar products from other jewellery shops, like Google's
product view: picture, name, shop, price and the shop's link (opens in a new tab).
It fills after a text search, after "More like this", and from "Similar on the web"
in a design's details; clicking a product shows a large preview with Visit / Copy link
and related products. It shows similar products, not copies of a design.

The products come from a small pool made once (about 20 minutes, ~130 MB of pictures):

    .venv/bin/python scripts/build_web_products.py --per-shop 500

It reads the public product feeds of the shops in `jewelsearch/webproducts.py`
(robots.txt checked first), keeps in-stock products of our five types, and reads each
first picture with SigLIP2 and DINOv2. The app loads the pool at start-up; without it
the panel says the web collection isn't ready. Pictures are served from this app at
320 px (cards) and 640 px (preview), fitted whole into a square frame, enlarged at most
2x and never cropped. Not continuous yet: run the script again to refresh.

## Search by photo

Next to the search box, the photo button (and **Search with a photo** under it) takes a picture
of a piece: a phone photo, a screenshot, or a photo from a website. It can also be dropped
anywhere on the page or pasted. Phones offer the camera. The page shows the photo's **Design
DNA** and the designs closest to it. Words typed next to the photo refine the search ("rose
gold", "no side stones", "thin band"), and the type and metal filters work as in a text search.

**Design DNA** is what the AI reads from the photo:

| Trait | How it is read | How reliable (see below) |
|---|---|---|
| Type | the nearest designs' own types (70%) and the image model's reading (30%) | right 97–98% of the time when sure enough to filter (about 95% of photos) |
| Metal colour | the image model, on the cropped piece | right 98% of the time when shown (85% of photos) |
| Stones, band, look, form | the index's attributes, read on the photo and on its nearest designs | the same attributes the text search uses |
| Diamond layout: solitaire, centre + smaller diamonds, even-sized diamonds | a trained reader on the image model's vector | right 89% of the time when shown (91% of photos), on designs it never saw |
| Centre stone cut (round, oval, pear, princess, emerald, …) | a second trained reader | right 91% of the time when shown (55% of photos with a centre stone), on designs it never saw |
| Diamond setting: halo, bezel, three stone, tennis line, rows | the photo must outscore 99% of the collection's designs of its type | about 3 in 4 or better |
| Style motifs (floral, heart, twisted, star, …) | the same percentile test | about 3 in 4 or better per motif |
| Diamond-set band ("Details") | a vision-language model (Qwen3-VL-2B, Apache 2.0) answers yes / no on the cropped piece | right about 17 of 18 times, checked by eye on catalogue rings |

Each trait with a **+** adds its word to the search box, so it becomes the same filter a typed word
would be. The words stay readable and editable. Tapping **Oval cut centre stone**, for example, adds
"oval", which the search reads like a typed word.

**Diamonds are read from pictures, never looked up.** The same two readers read the uploaded
photo and every catalogue design's own renders (all its views), so a photo is matched to designs
whose diamonds *look* the same: the layout and the centre stone's cut. Every design card shows its
centre cut as a tag ("oval centre") when its renders read clearly. The colour and clarity of a
diamond can't be judged from a photo, so they aren't shown.

The readers are single softmax layers on top of the SigLIP2 vector, trained with
`scripts/train_diamond_dna.py`. The job cards and CAD files are used there only as the answers to
learn from and to check against (`scripts/diamond_labels.py`); the search never reads them. The
training pictures are the catalogue views plus two simulated shopper photos per design (another
metal colour, another angle, a background), so the readers work on phone photos, not only renders.
Checked on design families the reader never saw (5 folds):

| Reader | Simulated photos: right / shown | Catalogue renders: right / shown | Image model alone (zero-shot), photos |
|---|---|---|---|
| Layout | 89% / 91% | 92% / 92% | 70% |
| Centre cut | 91% / 55% | 99% / 39% (the card tags) | 65% |

On 11 real photos the layout was right for 8 of 9, and the centre cut for 3 of the 4 it showed (a
pear-cut green emerald read as "emerald cut"). Retrain after the index is rebuilt; the server
ignores a reader made for another index.

**Design details** (`jewelsearch/details.py`): a small vision-language model reads three yes / no
details from the photo and from every design's front view (`scripts/build_details_index.py`, about
6 hours once, `data/index/details.npy`): "halo", "clusters on the band" and "diamond-set band".
Checked on the whole collection, only the band means what its name says. The model says "halo"
and "clusters" for most rings with any small diamonds (halo for 70% of rings whose job card has
no stone standing out). Its answers are consistent, though: a photo and the design's own render
agree 92–95%, and a shop's worn photo and its studio shot 95–100%. So all three are matched with
the collection (weight 0.5), and only **Diamond-set band** is shown, in the Design DNA and as
"Shares: diamond-set band" on cards. Two-tone metal is not read any more (a product's worn and
studio photos agreed only 65% of the time). Measured with the details on:

| Details | 300 simulated photos: first / first page | 88 styled shop photos* | 100 worn shop photos* |
|---|---|---|---|
| Off | 78.0% / 94.7% | 23.9% / 65.9% | 19.0% / 55.0% |
| All three matched, weight 0.5 (live) | 78.0% / 93.3% | 25.0% / 68.2% | 23.0% / 62.0% |

\* Svaraa photos: the first result agrees with the product's own studio shot / the studio shot's
first result is on the first page. A photo takes about 3.2 s to read with the details, 0.8 s without.
Without `details.npy`, or with `JEWEL_DETAILS=0`, photo search works without them (the model takes
about 4.5 GB of memory).

**How the matching works**

1. The photo is shrunk to 1024 px and re-encoded on the phone (location data doesn't leave it),
   then checked on the server (image types, size, pixel count, EXIF rotation).
2. **Find the piece** (`jewelsearch/photo.py`): the background is modelled from the photo's
   borders (handles gradients and vignettes); cast shadows (same colour as the background, only
   darker) are not part of the piece. The piece is cropped and padded like the catalogue crops.
   A busy background (worn on a hand, next to a face, on silk with tweezers) defeats that cut-out:
   it finds nothing, or it finds the person. There the **piece finder** takes over
   (`SearchEngine._find_piece`): a trained reader of DINOv2's patches (one per 14 px of the photo
   at 336 px, `scripts/train_piece_finder.py`) marks the patches that are jewellery, and the crop
   around the most certain group of them is used. When the cut-out and the finder disagree (the
   finder may see a necklace's pendant but not its thin chain), the crop closer to the collection
   wins. The page outlines the piece that was used on the photo in the Design DNA panel.
3. **Embed** the crop and its mirror image twice: with the same SigLIP2 model as the index,
   which knows what a picture shows ("a halo ring") and is how words and pictures meet, and with
   DINOv2 (Meta, Apache 2.0, `jewelsearch/dino.py`), which is much better at telling apart
   near-identical designs: the exact shape of a head, a band, a cluster. SigLIP2 alone sees the
   top 100 rings of a halo photo as almost equally close (0.90 to 0.92).
4. **Compare** with every view of every design (`data/index/views.npy` and
   `data/index/dino_views.npy`); a design's best view counts, since a photo can show any angle.
   The two closeness scores are standardised within the filtered designs and added (weight 1 each).
   DINOv2 alone also matches the pose of the photo too much, so both are needed.
5. **Filter and rank**: type (from the words, a pick, or the photo when sure), metal and the
   strict word filters exactly as in a text search. Words with a picture meaning add a text
   score (weight 0.6 against the photo's 1.0). Agreement on stones, band, look and form adds a
   little, and so does agreement of each design's diamonds with the photo's, both read from
   pictures (weight 0.3). On the simulated photos that raised results sharing the photo's diamond
   layout from 82% to 84% and its centre cut from 61% to 68%, and found the design itself slightly
   more often (first result 59% → 61%). Typed cut words ("oval ring") still match the cut named in
   design ids plus what the words describe: adding the image-read cuts made them less exact.
6. **Labels** come from how far the first result leads the next design family on the two
   picture models together, within the filters, and only the first card can carry one:
   **Same design** a lead of 1.3 or more (shown for 42% of the test photos, right in all of them in
   the latest run and 96–99% in earlier ones), **Very close** 0.8 to 1.3, and both models must
   rate the design well on their own (so one model's spike can't make a match). Every other card
   is **Similar style**. Measured on first cards, clean / worn-style photos (small samples, 13 to
   30 photos per label): Very close right 88% / 62%. A lead of 0.5 to 0.8 was right 77% / 60% on
   those, but on 198 real shop photos it agreed with the product's own studio shot only 5 times
   in 16, so **Close match** is no longer given. Before, every card scoring 4 or 5 standard deviations above the
   rest was "Close match" or "Very close": a quarter of them (clean photos) and an eighth (worn)
   were the photo's design, and a hand photo put a wide multi-row ring first as a "Close match"
   while the right ring sat fifth. When nothing is a clear match the page says so: "No design
   here is a clear match for this one; these are the closest in style."

The photo's own metal colour only chooses the colour the cards show. Every design comes in all
three colours, so the photo's colour isn't a filter. Typing "rose gold" or picking a metal is.
If the type isn't clear from the photo, every type is shown with a note, never a guessed filter.

**Privacy:** the photo is kept in the server's memory for an hour, so filters and "Show 8 more"
don't send it again. It is never written to disk and never shown to another user. Photo searches
are not added to the search history.

**Build the per-view index** after every `build_index.py` run (about 20 minutes on an M2; it reads
`data/crops` only, not the drive). Without it, photo search still works and compares with the
design and front-view vectors, but finds fewer designs. The file carries a fingerprint of the
index, so the server ignores it after the index is rebuilt.

```sh
.venv/bin/python scripts/build_view_index.py
.venv/bin/python scripts/build_dino_index.py      # the same views with DINOv2 (~25 min)
.venv/bin/python scripts/train_piece_finder.py --backgrounds data/piece_finder/backgrounds   # the piece finder (~15 min)
```

The piece finder learns from catalogue crops pasted onto ordinary photos (people, hands, tables,
fabric) in `data/piece_finder/backgrounds` (Wikimedia Commons, see the README.txt there); it does not
depend on the index, so it only needs retraining if the DINOv2 model changes. Without
`data/index/piece_finder.npz` photo search works as before, with the background cut-out alone.

**Measured quality** (`scripts/eval_photo_search.py`): there are no real shopper photos with
known answers yet, so test photos are made from catalogue renders the index hasn't seen: the
same design in another metal colour, often from another angle, on white, plain, skin-tone or
wood/fabric backgrounds, rotated, resized, recoloured, blurred and JPEG-compressed.

| Background | Design found first | Found on the first page (8) | Before the piece finder (first / page) | Before DINOv2 (first / page) |
|---|---|---|---|---|
| Catalogue screenshot (white) | 87% | 96% | 87% / 96% | 76% / 95% |
| Plain / velvet | 80% | 99% | 81% / 97% | 65% / 91% |
| Skin tone | 76% | 96% | 73% / 95% | 57% / 84% |
| Wood, fabric, stone (piece small) | 71% | 93% | 60% / 80% | 36% / 76% |
| All (300 photos) | 78% | 96% | 75% / 92% | 57–62% / 83–89% |

Rings are hardest (3,338 similar rings): 63% first and 95% on the first page, up from 45% first. Real photos are harder than these simulated ones.

**Worn photos** (a ring on a hand, earrings by a face) were not in that test, and were where real shop
"lifestyle" photos failed. A second test pasted 240 catalogue pieces (in a metal the index hasn't
seen) at worn size onto 22 real photos of hands and faces (Wikimedia Commons, not used to train the
finder), with the same blur, colour and JPEG changes:

| Worn-style photos | Before the piece finder | Now |
|---|---|---|
| Design found first | 3.8% | 37.9% |
| Found on the first page | 8.8% | 61.7% |
| Rings: first / first page | 1.2% / 3.1% | 32.5% / 56.2% |
| Type read right | 53% | 92% |
| Wrong first card labelled "Close match" or better | 182 of 240 | 14 of 240 |

Pieces there are small and pasted, not worn, so treat these as a direction, not a promise.

**Real shop photos** (Svaraa, 107 products, the kind of link shoppers paste): each product's styled
photo (on fabric, stone) and worn photo (on a model) were searched and compared with what the
product's own studio shot on white finds. These are other brands' designs, so "right" here means
"agrees with the studio shot", not "found the design":

| Real shop photos | Before the piece finder | Now |
|---|---|---|
| Worn (104): same first design as the studio shot | 8.7% | 23.1% |
| Worn: the studio shot's first design on the first page | 22.1% | 60.6% |
| Styled (94): the studio shot's first design on the first page | 59.6% | 69.1% |

**Camera angle.** Shop photos never share the catalogue's four camera angles, and DINOv2 is
sensitive to angle. Test photos taken from the turntable videos at random moments (240, half on
white, half on a photo) found the design first 28.7% of the time at DINOv2 weight 1.0 and 30.4%
at 0.5 (rings 21.7% / 25.0%), while the tests above, whose photos reuse the catalogue's angles,
prefer 1.0 by 2–3 points. The weight stays 1.0 for now; the difference is within the noise.
On 11 real photos from Wikimedia Commons the type was right for 10 (a pendant on a long chain read
as a necklace) and the metal colour was right whenever it was shown. A photo takes under a second
(median 0.8 s while the design-details batch was using the GPU too).

What was tried and left out, because it made results worse on the same photos: query expansion
(averaging in the top matches), a hubness correction, whitening the background around the piece,
and reading the metal from pixel colours (83% against the model's 99.7%).

## What the index knows about each design

| Field | Source |
|---|---|
| type | id prefix → folder names → image model; the model overrides the id rule only when a folder name agrees (`scripts/build_index.py`) |
| metals | file names (`@R/@W/@Y`) |
| front view | most left-right symmetric view, majority vote per series (`make_crops.py` stats) |
| stones / band / weight | zero-shot image-model scores on the front view (`jewelsearch/attributes.py`) |
| stone share | colourless-bright pixels on a yellow/rose render (`scripts/stone_pixels.py`) |
| men's | "Mens ring"/"Gent's Ring" folders, SBMR and GGR series |

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Building the index (one time, then after new designs are added)

Indexing only reads the dataset storage (the S3 bucket, or the SSD before the
move, see [Dataset storage](#dataset-storage-s3)); all output goes to `data/`.
The admin upload panel is the only thing that writes to the storage. The CAD
scripts (`build_cad_specs.py`, `build_tryon_models.py`) download only the files
they process into a size-capped cache, `data/storage/cache/` (20 GB, set
`STORAGE_CACHE_GB` to change).

```sh
python3 scripts/audit_dataset.py                    # scan file names -> data/audit/
python3 scripts/build_catalog.py                    # one record per design -> data/catalog.jsonl
.venv/bin/python scripts/make_crops.py              # tight 512px crops + view stats -> data/crops/ (~35 min)
.venv/bin/python scripts/build_index.py             # vectors, front view, attributes -> data/index/ (~17 min on M2)
.venv/bin/python scripts/stone_pixels.py            # measured stone share per design (~1 min)
.venv/bin/python scripts/build_view_index.py        # every view's vector, for search by photo (~20 min)
.venv/bin/python scripts/build_dino_index.py        # every view with DINOv2, for exact photo matches (~25 min)
.venv/bin/python scripts/train_diamond_dna.py       # the diamond readers for search by photo (~20 min)
```

`make_crops.py` skips images already processed, so re-running after adding
designs only processes the new files. It removes the chain from pendant crops
using the category in the index, so on a first-ever build run it once more
after `build_index.py` for designs whose type only the model could tell.

## Checking quality

```sh
.venv/bin/python -m pytest tests -q                  # parser + end-to-end ranking tests
.venv/bin/python scripts/eval_sheets.py out/ "prompt 1" "prompt 2"   # contact sheets of the top 8
.venv/bin/python scripts/eval_photo_search.py        # search by photo on 300 simulated shopper photos (~1 min)
```

## Running

```sh
scripts/start_live.sh            # live: this Mac + a public https link (Ctrl+C stops all)
# or, only on this Mac:
.venv/bin/uvicorn jewelsearch.server:app --port 8765
```

`start_live.sh` runs the app and a Cloudflare quick tunnel together, prints the public
link, and keeps the Mac awake while it runs. If the app stops on its own, it is started again
behind the same tunnel, so the link keeps working. Pages say "try again in a minute" while it
loads. An app that stops 3 times in a row within 2 minutes of starting is left stopped. The
link changes on every start of the script; set `PUBLIC_URL` to a fixed https domain (e.g. a named tunnel) to
keep one. **On phone** (top right, and in the try-on dialog) shows a QR code that opens the
same page on a phone, whose camera needs https: from a try-on, a ring with a 3D model opens
the live camera, other designs their photo try-on.

**All models share one GPU lock** (`jewelsearch/config.py` `GPU`). PyTorch's Apple GPU backend
is not thread-safe: two requests running models at the same moment crashed the app ("failed
assertion _status < MTLCommandBufferStatusCommitted"). Any new model must run under that lock
(`tests/test_gpu_lock.py`).

Open http://localhost:8765.

**Catalogue pictures are the dataset's original renders, unchanged**: cards, the
detail view and its angles, category tiles, search history, favourites, the buy page,
the jeweler panel and job cards all show the render file exactly as it is in the
dataset (2600 x 2600 PNG, about 2 MB), sent by `/media/<id>`. Nothing is cropped,
resized or re-encoded; the browser only scales it to fit the tile, on white. Only the
try-on previews make their own images. `data/crops` (512 px cut-outs) is what the
image models read, never shown, with two exceptions:
- when an original can't be read (the dataset drive isn't connected), its preview
  stands in so the page still works, never cached in its place, and the page says
  "The original photos are offline";
- orders store preview links, which outlive catalogue changes; pages show the
  originals while the design is in the catalogue.

Cost, measured through the public link: the first page of 8 results is about 16 MB
of pictures (12.9 s on this Mac's connection, history thumbnails included) instead of
about 0.4 MB. Each original is kept a day in the browser cache, so a picture seen once
loads instantly; after that day, an unchanged file is answered "not modified". With
S3 storage the browser downloads the original straight from the bucket, never a WebP
copy (`scripts/migrate_to_s3.py` makes those only with `--image-web-copies`).

## Dataset storage (S3)

The dataset is addressed by URL, never by a disk path, and the code holds no
storage location: `.env` says where it is (template: [.env.example](.env.example)).

```sh
JEWEL_STORAGE=file:///Volumes/Storage     # now: the SSD
JEWEL_STORAGE=s3://<bucket>               # after `connect`: the bucket
JEWEL_STORAGE_FALLBACK=file:///Volumes/Storage   # during the move only: files not moved yet
S3_BUCKET=<bucket>
S3_REGION=ap-south-1                      # the bucket's region (R2: auto)
S3_ENDPOINT=https://...                   # only for non-AWS providers (R2, DigitalOcean, Akamai, E2E ...)
AWS_ACCESS_KEY_ID=...                     # an access key for this app only (policy below)
AWS_SECRET_ACCESS_KEY=...
```

Without `JEWEL_STORAGE` the app still searches, and full-size media, try-on
photos and uploads answer "storage not available" instead of guessing a place.

All reads and writes go through [jewelsearch/storage.py](jewelsearch/storage.py).
A file keeps its dataset name (`01/01/Loat - 02/Ring/DDLR-423/1174@Y-#viwe1.png`)
as its ID, and the storage turns that name into a signed link for the browser,
the bytes for the server, or a cached local file for the CAD tools.

**How a full-size image or video is fetched.** The search API gives each page
ready-made links, `/media/<id>`, where the ID is an opaque fingerprint of the
file: pages never see, build or send dataset paths. The server maps the ID back
to a catalogue file (anything else is a 404), checks the signed-in user like
every route, and answers with a redirect to a signed link that works for one
hour. The browser downloads straight from the bucket, so big videos never pass
through this Mac or the tunnel. The bucket itself stays private: `check`
refuses a bucket that serves files without a signature.

**Bucket layout**

    originals/<dataset name>                       every SSD file once, unchanged
    originals/<dataset name>.zst                   CAD and text files, zstd-compressed (only when >= 10 % smaller)
    web/<category>/<design>/<metal>/view<N>.webp   display copy of each catalogue render
    web/<category>/<design>/<metal>/turntable.mp4  smaller H.264 copy of each 3D video
    index/manifest.sqlite                          snapshot of the storage index
    index/search-<date>.tar.zst                    backup of the search index (crops, vectors, catalogue)

**The storage index** (`data/storage/manifest.sqlite`, SQLite) is what makes
lookups quick. It records every dataset name, the object holding its bytes, its
SHA-256, size, type and compression, the web copies, and which renders and
videos belong to which design (`design_files`, indexed by design id, category
and metal). The app answers "where is this file", "what does this design have"
and "what is in this folder" with one indexed query and never lists the bucket.
A new server without the file downloads the snapshot from `index/`.

**Smart compression**

| Data | Shown to users | Kept in `originals/` |
|---|---|---|
| PNG renders | WebP at full resolution, quality 90 | the PNG, lossless |
| MP4 turntables | H.264, at most 1920 px wide, ready to stream (`--videos encode`) | the original |
| CAD (.3dm, .stl ...), text | not shown | zstd level 9, when it saves >= 10 % |
| identical copies (`PHOTOS` folders) | | stored once, all names point at it |
| PNG, MP4, xlsx, PDF | | as they are (already compressed) |

Originals go to the S3 Intelligent-Tiering storage class: files nobody opens
move to cheaper tiers by themselves, with no retrieval fee or delay. Web copies
stay in Standard.

### Switching to the bucket (the day the storage is bought)

1. Put the provider's settings in `.env` (the `S3_...` and `AWS_...` lines above).
2. Plug in the SSD, then run (about 15 minutes):

   ```sh
   .venv/bin/python scripts/migrate_to_s3.py connect
   ```

   It checks the keys, upload speed and that the bucket is private, sets the
   bucket up (no public access, failed uploads cleaned after 3 days, versioning
   where offered), starts the storage index, and switches `.env` to the bucket
   with the SSD as fallback. Restart the app (`scripts/start_live.sh`). From now
   on new uploads go to the bucket, and files not moved yet still show from the SSD.
3. Move everything in the background (hours to days, depending on upload speed;
   `check` prints an estimate). It can be stopped and started again:

   ```sh
   nohup .venv/bin/python scripts/migrate_to_s3.py move > data/storage/move.log 2>&1 &
   ```

4. When it reports "complete and verified":

   ```sh
   .venv/bin/python scripts/migrate_to_s3.py finish    # refuses while anything is missing
   ```

   This removes the SSD fallback. Restart the app: it uses only the bucket.
   Keep the SSD unchanged as an offline backup for a while.

On a new or rebuilt server, `migrate_to_s3.py restore` brings back the storage
index and the search index (crops, vectors, catalogue, try-on models) from the
bucket.

### The single steps

`connect` and `move` run these. The SSD is only read. Every step can be stopped
and run again; finished work is skipped. `--source` defaults to the SSD in `.env`.

```sh
M=".venv/bin/python scripts/migrate_to_s3.py"
$M check                                        # keys, speed, bucket is private: test write, read, delete
$M setup --versioning                           # block public access; unfinished uploads removed after 3 days; undo for deletes
$M scan --estimate                              # list the SSD, estimate the savings on a sample
$M web --limit 50                               # trial: 50 renders; look at them in the bucket
$M web                                          # all WebP display copies + the 3D videos
$M index                                        # snapshot of the storage index
$M originals                                    # every file, de-duplicated and compressed (the long one)
$M web --videos encode                          # smaller videos (needs imageio-ffmpeg; slow, prints its progress)
$M verify --sample 500                          # sizes of every object + SHA-256 of a random sample
$M index                                        # final snapshots
$M status                                       # what moved, what is left, what compression saved
```

`verify` takes any broken entry out of the storage index, so running `web` or
`originals` again sends those files again. Keep the SSD unchanged until
`verify` passes after the last step.

**Access key policy** for the app (replace `BUCKET`). `setup` also needs
`s3:PutLifecycleConfiguration` and `s3:PutBucketPublicAccessBlock`, so run it
once with an admin key, or set those two in the console.

```json
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:ListBucketMultipartUploads"], "Resource": "arn:aws:s3:::BUCKET"},
  {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
   "Resource": "arn:aws:s3:::BUCKET/*"}]}
```

**Cloudflare R2** (S3-compatible, no download fees) works with the same code:
set `S3_REGION=auto` and `S3_ENDPOINT=https://<account id>.r2.cloudflarestorage.com`,
and use an R2 API token with Object Read & Write for the bucket. Differences
from AWS, already handled: no Intelligent-Tiering (originals stay in Standard),
no versioning (add a bucket lock rule on `originals/` in the R2 dashboard
instead), no public-access setting (R2 buckets are private unless a public URL
is switched on, so leave it off), optional checksums sent only when required.
Data placement follows the bucket's location hint (`apac`) on a best-effort
basis; R2 has no India region.

Nothing personal goes to the bucket: body photos for try-on stay in
`data/bodyphotos/` on this server, and the search backup leaves them out.

## Sign-in and access control

Everything (pages, search API, thumbnails, renders, videos) requires a signed-in
user whose account an administrator has approved. The browser talks to this
server, plus the private dataset bucket through one-hour signed links that only
this server hands out (after the same sign-in check). Supabase and the bucket
are called from the server alone, and their secret keys never leave this machine.

- **Accounts:** Supabase Auth stores emails and hashed passwords.
  `public.profiles` holds the `approved` flag.
- **Session:** after sign-in the server sets an HttpOnly, SameSite cookie,
  signed with `data/session_secret` (created automatically). It is valid for 7 days.
- **Approval:** approval is re-checked every minute. Unticking `approved`
  locks a user out within a minute.
- **Brute-force limits:** 8 attempts per email and 30 per IP every 5 minutes,
  and 5 registrations per IP per hour.

One-time Supabase setup:

1. Create a free project at https://supabase.com.
2. **SQL Editor → New query**: paste [supabase/schema.sql](supabase/schema.sql) and run it.
3. **Authentication → Sign In / Providers → Email**: turn **Confirm email** off.
   Admin approval already gates access, and Supabase's built-in mailer only
   sends a few emails an hour.
4. **Project Settings → API Keys**: copy the project URL and the **secret** key
   (`sb_secret_…`, or the legacy `service_role` key).
5. `cp .env.example .env`, fill in both values, and restart the server.

**Continue with Google (optional):** the button calls this server's
`/api/auth/google`. The server runs the OAuth PKCE exchange with Supabase and
sets the normal session cookie. New Google users wait for approval like
everyone else. To switch it on:

1. In Google Cloud Console, open **APIs & Services → Credentials → Create
   credentials → OAuth client ID** and choose **Web application**. Add this
   authorized redirect URI: `https://YOUR-PROJECT-REF.supabase.co/auth/v1/callback`.
2. In Supabase, open **Authentication → Sign In / Providers → Google**. Enable
   it and paste the Client ID and Client secret.
3. In Supabase, open **Authentication → URL Configuration → Redirect URLs** and
   add `https://<your public address>/api/auth/google/callback`, plus
   `http://localhost:8765/api/auth/google/callback` for local use. If the
   public address changes (the quick tunnel gives a new one on each restart),
   add the new one too, or set `PUBLIC_URL` in `.env` to a fixed domain.

To approve someone, open **Table Editor → profiles** and tick `approved` for
their row. Untick it to revoke access. To remove a user completely, delete them
under **Authentication → Users**.

## Admin panel: uploading new designs

Admins get **Admin panel · Upload designs** in the user menu (`/admin`). Pick a
category (or **Other…** and type your own category name,
e.g. `Mangalsutra`, which becomes its own folder), optionally a folder name (e.g. `Loat - 15`), then add files or whole
folders (button or drag and drop). Any file type is accepted. Files are saved
with their original names, in the dataset storage, as

    New Dataset/<Rings|Earrings|Pendants|Necklaces|Bracelets|your category>/[folder]/...

(in S3: `originals/New Dataset/...`, recorded in the storage index at once).

- Nothing is overwritten. A clashing name is saved as `name (2).ext`.
- Uploads go in 32 MB chunks, so big videos and CAD files pass the Cloudflare
  tunnel's 100 MB limit. A dropped connection is retried per chunk. Half-sent
  files never appear in a category folder: in S3 they are an unfinished
  multipart upload, on a drive they wait in `New Dataset/.incoming/`.
- On a drive, 5 GB is always kept free; S3 has no limit. Hidden and system
  files (`.DS_Store`, `._*`, `Thumbs.db`) are skipped.
- Every saved file is logged in `data/uploads.log` (time, admin, path, size).
  The panel shows this log.
- New files appear in search only after the index is rebuilt (see above).

To make someone an admin, run [supabase/admin.sql](supabase/admin.sql) once in
the SQL Editor. Then tick `is_admin` for their row in **Table Editor →
profiles**. Admin rights are re-checked every minute, and anyone without the
tick gets 403.

## Proceed to buy (`/buy`)

Every design card in the search results has a **Proceed to buy** button, and so does the
design dialog (which favourites also open). It opens `/buy?uid=…&id=…&metal=…` in the metal
colour the card was showing. The page shows:

- the design's pictures and 3D video per metal colour
- choices: **metal colour**, **gold purity** (14K / 18K, and 22K in yellow gold only), and for
  rings the **ring size** (Indian sizes with US size and inner diameter). The choices live in
  the page address only, so a link can be shared; nothing is stored.
- **Your selection**: gold weight for the chosen purity, and total diamonds
- **Product details**, **gold weight by purity** and **diamond details** (cut, size, count,
  carats, setting type)

**Order this design** sends the order to the jeweler (see the jeweler panel below). The customer picks a
quantity and gives a phone number with its country code. A ring also needs a size the customer chose or
ticked, because the preselected size is only the size the design was made in. No price is shown and no
payment is taken: the jeweler calls with the price.

**Where the figures come from** (each is labelled on the page):

| Source | Designs | What it gives |
|---|---|---|
| Job card: the client's own `<design>@5.XLSX` sheets | ~210 | Gold weight per purity after polishing, the diamond list with setting types, ring size. Always used when present. |
| The CAD file (`.3dm` / `.stl`) | ~3,300 | Measured from the same single piece as the try-on model: metal volume, every stone's cut and size, ring size, piece size. Marked "≈" and "estimated". |
| Neither | the rest | "Confirmed when you order". Nothing is guessed. |

CAD figures are shown only where they were checked against the job cards:

- **Gold weight** for rings and pendants whose metal surface is closed. The factor from CAD
  volume to 18K grams (~12 g/cm³, not 18K's ~15.5, because 0.2–0.3 mm is polished away) is
  fitted on the rings that have both. About 90% of those land within 10% of the card.
  Earrings get no CAD weight. Their cards include findings such as screw posts that the CAD
  lacks, and a file may hold one earring or the pair.
- **Diamonds**: round stones use the trade size chart from the job cards, other cuts their
  volume in the file. Carat totals match the cards (median ratio 1.00). For earrings, only the
  cuts and sizes are listed, because the file doesn't say whether it holds one or the pair.
- The purity ratios are the job cards' fixed ones: 14K = 0.85 × 18K and 22K = 1.14 × 18K.

**Build the data** after adding designs or job cards. Use the CAD venv. The run reads the drive
only and takes about 30 minutes, and later runs measure only new designs:

```sh
.venv-cad/bin/python scripts/build_cad_specs.py          # job cards + CAD -> data/cad/specs.json
.venv-cad/bin/python scripts/build_cad_specs.py --cards-only   # only re-read the job cards
```

A few STL files are huge (up to 660 MB, about 10 GB of memory each), so files over 150 MB
are measured one at a time at the end, and workers are replaced every 10 files.
`--cards-only` rebuilds `data/cad/specs.json` from what has been measured so far.

The server picks up a new `data/cad/specs.json` without a restart
(`jewelsearch/purchase.py` decides what is shown).

## Jeweler panel (`/jeweler`)

The manufacturing side runs orders here: the orders customers place with **Order this design** on
the buy page. Jewelers sign up like everyone else. An admin approves them and then ticks
`is_jeweler` for their row in **Table Editor → profiles**. Admins can open the panel too. The user
menu shows **Jeweler panel · Orders** to both.

One-time setup: run [supabase/orders.sql](supabase/orders.sql) in the Supabase SQL Editor. It adds the
`is_jeweler` column and the `orders` and `order_events` tables, locked to the server like the others.
Until it has run, placing an order and the panel say that orders aren't set up yet.

**Stages.** An order moves one stage at a time: New request, Price sent, Customer approved, Advance
received, In production, Quality check, Final bill, Ready / dispatched, Delivered. It can move back
one stage, or be cancelled with a reason until it is delivered. Some stages need something first:

| To enter | Needs |
|---|---|
| Price sent | a saved price |
| Advance received | a recorded payment |
| Final bill | the actual gold weight, entered at quality check |

**Price.** The jeweler types every figure: the gold weight (prefilled from the job card or the 3D
estimate, times the quantity), the gold rate per gram, the making charge (per gram, % of gold or fixed),
the diamond quality and value, one other charge and GST (prefilled at 3%). The server adds them up and
rounds the total to the rupee; it never fills in a rate itself. The price can be changed until the
customer approves it. After that it is locked unless the order is moved back to "Price sent".

**Final bill.** The weight measured at quality check replaces the quoted weight. Gold and a per-gram or
percentage making charge follow it. Diamonds and other charges stay as quoted. The panel shows what
was paid and the balance.

**Also on each order:** the design exactly as ordered (pictures, metal, purity, size, gold weight,
diamonds, frozen when the order was placed, so index or spec rebuilds never change it), the customer's
contact details with Call, WhatsApp and Email buttons, their note, payments (cash, UPI, bank, card,
cheque), internal notes, and the full history of who did what. **Print job card** opens an A4 job card
for the workshop with the design, the diamond list, the customer's instructions and blank rows for
each process. It doesn't show the customer's contact details.

**Two people on one order:** every change is checked against the version of the order the jeweler
was looking at. If someone else changed it in between, nothing is overwritten: the page reloads and
says so.

Not built yet: a "My orders" page for customers, online payment, and the custom-design chat.

## Brand designs: other jewellers' designs with our version (`/brand-import`, demo)

The team (jewelers and admins) lists a design from another jeweller's website:

1. **Team panel** (`/brand-import`, menu "Brand designs"): paste the product link and press
   **Fetch details**. `jewelsearch/brands.py` reads the page with link search's safe fetcher
   (public addresses only, size and time limits): name, brand, SKU, price and MRP from JSON-LD,
   shop tags and, on Shopify shops, the product JSON; the specification shown as label/value
   pairs or tables (price breakups); stated figures in the description ("18 KT Yellow Gold(2.150 g)
   with diamonds (0.280 ct, FG-SI)"). Menus, headers and footers are skipped. Pictures are
   downloaded here and kept exactly as downloaded (the team picks which to keep).
2. **Our version**: name, category, metal colours, purities, gold weight per purity (job-card
   ratios fill the others), diamonds, quality, size, and our pricing (24K rate priced by gold
   content, making charge, diamond value, other charges, GST). The price per purity uses the
   same sums as the jeweler's quote (`orders.bill`).
3. **Submit and display in catalog** saves it (Supabase `brand_designs`, pictures in
   `data/brand_designs/`) and opens the category in the catalogue with the design first,
   marked "Just listed". "All categories" also gets an "Other brands" tile.
4. **Design page** (`/brand-design?id=`): the original and our version side by side (our column
   highlighted), our specification and price breakup, the original details as read, and at the
   end **Buy with us** (the usual `/buy` page and order, kind `brand`, our listed pricing
   prefilled for the jeweler) and **Buy with seller** (the seller's page in a new tab).

Setup: run `supabase/brand_designs.sql` in the Supabase SQL editor once (it also allows the
`brand` order kind), then restart the app. Tested shops: Giva, Palmonas (Shopify), Melorra,
Svaraa and CaratLane read; Tanishq blocks automatic visits (403); CaratLane's price is built in
the browser, so it shows "Not shown". Tests: `tests/test_brands.py` (no internet).

## Sketch to Design (`/sketch`, AI demo)

Draw on the sketchpad, take a photo with the camera, or upload / paste / drop a picture of a sketch. Pick type
(or Auto-detect), background, model / quality, optional notes in any language, and extra sparkle. The server
makes the picture through one wrapper (`jewelsearch/sketch.py`) over two providers; shoppers never leave the
page. "Change this" edits the last picture with a short instruction, and "Find similar in our collection" runs
the free local photo search on the result. Menu: Sketch to design · AI. Page `static/sketch.html`, tests
`tests/test_sketch.py`.

**Providers** (keys in `.env`, server only; only connected providers' models are offered, free ones first):
- **Pollinations** (`POLLINATIONS_KEY`, secret `sk_` key from enter.pollinations.ai). A free account's quest
  credits pay for FLUX.2 Klein (default, cheapest), FLUX Kontext and GPT Image mini. Nano Banana / 2 / Pro there need paid
  credits and are listed only with `POLLINATIONS_PAID=1`. The sketch goes up as a file to `/v1/images/edits`; no public link is made.
- **Google direct** (`GEMINI_API_KEY`). Nano Banana 2 Lite, 2.1 and Pro. Google has no free API tier for image
  models (pricing page, 2026-10-07).

**How credit is saved**
- The prompt comes from a fixed template plus the picks, never from an AI text call. Type and metal written in
  Gujarati / Hindi / English are read by the local parser. Prompts are about 80 words.
- Pictures are shrunk to 1024 px JPEG in the browser and again on the server. An empty canvas is refused.
- One 1:1 1K picture per call.
- Same picture + same choices + same model returns the stored picture for free (shared cache).
- Edits send the last picture with a one-line instruction instead of starting over.
- Words-only requests go through the local jewellery judge first, so off-topic text spends nothing.
- Before each call: per-user daily count, team daily and monthly caps (`SKETCH_*` in `.env`), one call at a
  time per user.

Models offered: FLUX.2 Klein (default), FLUX Kontext, GPT Image mini (Pollinations); Nano Banana / 2 / Pro via
Pollinations with `POLLINATIONS_PAID=1`; Nano Banana 2 Lite, 2.1 and Pro with a Google key. The page shows no
prices or free / paid notes (2026-10-07 rule: no public or paid plans); the limits above still apply quietly.

Files: pictures in `data/sketch/out/`, spend ledger and gallery in `data/sketch/ledger.sqlite`.
Real Pollinations calls checked 2026-10-07 on a free account: Kontext, Klein and GPT Image mini each made a
1024 px picture from a test ring sketch in 14-30 s; Nano Banana answered 402 (paid credits needed).

## AI pictures on our own computer (no account, no daily limit)

Both AI panels can draw on this Mac with FLUX.2 Klein 4B (Apache-2.0), 4-bit weights
`mflux-community/flux2-klein-4b-mflux-q4` (4.6 GB, Hugging Face cache) run by mflux 0.21 in its own environment
`.venv-imagegen` (kept apart from the app's packages). It is the default model in both panels when installed;
`LOCAL_IMAGEGEN=0` in `.env` switches it off. Setup on a new Mac:

```
python3 -m venv .venv-imagegen && .venv-imagegen/bin/pip install "mflux==0.21.0"
.venv-imagegen/bin/hf download mflux-community/flux2-klein-4b-mflux-q4
```

Each picture is a separate process that exists only while it draws (`--low-ram`, offline), one at a time across
both panels; it waits up to 2 min for `LOCAL_MIN_FREE_GB` (default 2) of free memory, else says the studio is busy.
Measured 2026-10-07 next to the live app (which itself uses ~10 GB): 768 px 5.8 GB / ~3.5 min, 1024 px 10.3 GB
/ ~6-7 min per 2x2 variation tile. So ~30 pictures a day take ~2-3 h of machine time. Pictures are made as
background jobs the pages poll (`/api/sketch/job/{id}`), because the tunnel ends requests at 100 s.

## Design Variations (`/variation`, AI demo)

Add a sketch or photo (camera, upload, paste or drag and drop), crop and rotate it, pick how many variations
(1, 4, 9, 16, 25 or 36), a look (design illustration, realistic photo, pencil sketch), a model ("Auto" = GPT
Image mini) and optional direction words in any language. The set comes back as one labelled grid. The viewer
zooms (wheel, pinch, drag, double-click resets); tapping a variation picks it and shows its idea; Download,
Share, Sketch to Design, Variation (use it as the new source), Touch Up, Analyze (design reading + closest
designs in our collection, local models) and Copy link work on the picked one or the whole set.
Code `jewelsearch/variation.py` (same wrapper, ledger and limits as Sketch to Design), page `static/variation.html`,
tests `tests/test_variation.py` (real tiles in `tests/data/`).

**No fixed styles (2026-10-07 rule):** one planning call to Pollinations' text model (GPT-5.4 Nano, reads the
picture, ~0.001 pollen) names N ideas made for this exact piece, follows the direction words, and is told
which ideas were already shown for the same picture, so every set is new. If the planner is unreachable,
ideas are random technique x motif x form combinations (still different each time).

**Drawing:** free models draw a 2x2 grid reliably but not bigger ones (tested: 4x4 came back 3x3 or 4x5 with
stones lost), so each call asks for one 2x2 tile of four named ideas and the server joins the tiles: 16
variations = 4 calls, ~512 px per cell. Tiles are split at a white gutter or a hairline between touching
panels (photo looks), searched off-centre too; a 3-across grid or a single picture is never cut. Raw tiles
are kept in `data/sketch/tiles/` (newest 300). The model writes no text; the server draws the captions (Arial,
accents included). With Auto, if the account runs out of balance mid-set, the rest is drawn with FLUX.2 Klein.
Sets run as background jobs the page polls (the tunnel ends requests at 100 s).

## Ring try-on (`/tryon`)

Live camera try-on in the phone browser: the ring follows the user's ring
finger at real size. Everything runs on the phone. The camera video is never
uploaded, and the page is the only one allowed to use the camera.

**1. Convert CAD designs to web models** (separate venv, the search app is untouched):

```sh
python3 -m venv .venv-cad && .venv-cad/bin/pip install -r requirements-cad.txt
.venv-cad/bin/python scripts/cad_to_glb.py --category ring --limit 50      # or pass .3dm files
```

Output goes to `data/tryon/models/<slug>.glb` + `.json`, and each run is logged to
`data/tryon/convert_report.jsonl`.

- Reads MatrixGold `.3dm` files: metal and gem layers are separated, and each gem
  cut is stored once and instanced (a 197-stone band is about 90 KB).
- The ring axis, centre and size come from MatrixGold's "Finger Sizes / Ring
  Rail" curve. Without that curve, they are estimated from the metal (within about 0.5 mm).
- **Metal comes from the `.stl` next to the `.3dm`** when there is one: that is the file
  that gets cast, so it is complete. The `.3dm`'s own render meshes often miss parts
  (metal on "Heads" or other layers, parts saved without a mesh). Without an STL, a
  file whose metal is only partly meshed is rejected (`incomplete_mesh`).
- **Stones come from the `.3dm`** (an STL for casting has none). A stone saved without a
  render mesh is rebuilt from its facet corners (cut stones are convex). Before this,
  about 700 models showed plain gold where the design has diamonds.
- The metal is decimated to 60k triangles. That looks right on a phone but is
  deliberately too coarse to cast from.
- `.tools/gltfpack` (meshoptimizer, MIT) compresses each GLB about 20×.

**2. Open** `/tryon` (or `/tryon?model=<slug>&metal=rose`) on the phone
through the https address. The camera needs https.

- **Verification:** the ring appears only when the hand passes these checks: hand
  found, whole hand in frame, right distance, back of the hand to the camera,
  ring finger straight. Light, sharpness and steadiness show as tips. Each
  check was tested on MediaPipe's real hand photos: palm-side is rejected.
- **Real size:** the user picks a ring size (IN / US / mm; average by default).
  The band is resized radially to that size. The head keeps its real mm size.
- **Realism:** One-Euro smoothing (no jitter), invisible finger cylinders
  for occlusion, lighting and tint matched to the camera picture, and polished-metal
  and faceted-stone materials. The ring is softened to the camera's own sharpness and
  fades in and out. A contact shadow was tried and then removed: users saw it as a dark smear.
- **No lag behind the hand:** the screen shows a canvas that holds exactly the frame
  that was analysed, not the live `<video>`. The ring and the finger always come from the same frame.
  The canvas is capped at about 2.2 MP. Add `?debug=1` to see fps and detection time.
- **Calibrated on a real user frame:** the ring sits at 0.57 of the knuckle →
  middle-joint bone, finger width = 0.95 × knuckle spacing, and MediaPipe depth is
  damped ×0.6. Constants are in `static/tryon/hand-tracker.js`.
- **Capture:** the shutter freezes the frame. The user can then switch design, metal and
  size on the photo, then save or share it (phone share sheet, or a download on a laptop).
  Nothing is uploaded.
- Design thumbnails are rendered in the browser from the same GLBs.

three.js r186, MediaPipe tasks-vision 1.0.1 and the hand model are self-hosted
in `jewelsearch/static/vendor/` (no CDN, strict CSP).

**Status (2026-09-30): try-on is PAUSED at a basic first version.** What works: saved
hand/face/neck photos with verification; a "Try on" button on rings (all, via the catalogue
photo), earrings (catalogue photo, plus 3D where checked) and pendants (3D); the live ring
camera (`/tryon`, only 3D models that passed the check); the **On phone** QR button when
started with `scripts/start_live.sh`. Known limits: the catalogue-photo try-on is a flat
image (it doesn't wrap the finger or cast shadows), 3D looks like computer graphics, and only
~1,000 of ~3,700 designs have been converted with the current converter. Planned next step
when resumed: server-side path-traced rendering (Blender Cycles) of the verified CAD models
posed on the user's photo, for photo-real results.

## Instant try-on on your own photo

Each user adds a **hand**, **face** and/or **neck** photo once, at **My try-on photos**
(`/tryon/me`, also in the user menu). After that, a **Try on** chip on every design card, and
"Try it on me" in the design details, show that design on their own photo straight away.

| Photo | Used for | Checks before it can be saved |
|---|---|---|
| Hand | rings, bracelets | hand found, fingers and wrist in the photo, distance, back of the hand, ring finger straight, light, sharpness |
| Face | earrings | face found, looking straight at the camera, distance, both ears in the photo, earlobes not covered by hair, light, sharpness |
| Neck | necklaces, pendants | face and chin, both shoulders, facing the camera, neck not covered by a collar, light, sharpness |

- **Camera or upload.** With the camera, the photo is taken automatically once every check
  is green. An uploaded photo that fails a check is refused, and the page says which check failed.
- **Anchors** (earlobes, neck sides, neck base) are placed automatically, and the user can drag them
  to correct them. They are saved with the photo, so the preview needs no model when it opens.
- **Real size** comes from knuckle spacing (hand) or the iris, about 11.7 mm (face and neck).
- **Privacy:** photos are stored only on this server, in `data/bodyphotos/<user id>/` (mode 700).
  They are served only to the same signed-in user with `Cache-Control: no-store`. Saving needs an
  explicit consent tick, and users can delete one photo or all of them.

**Models for every design:** `.venv-cad/bin/python scripts/build_tryon_models.py` converts
every catalogue design that has CAD. It is resumable; `--retry` converts failed designs again
and `--force` converts everything again (after a converter change). It stops if the dataset
drive is not mounted. It writes `data/tryon/designs.json`.
Each CAD file is cleaned first, keeping one physical piece:
- layers that never hold the piece are skipped: cutters, the finger-size rail, notes, lights
- touching objects are grouped; pieces within 2 mm join the design (a ring head, a drop
  earring's links), while extra variants further away are dropped
- an earring pair is split to one earring: an empty band between the two, or the right
  half being a sideways copy of the left with nothing in the middle
- a stone is kept only if it sits on the metal it belongs to
- a piece outside the real size range for its category is rejected

**Real photo try-on (rings, earrings):** the try-on dialog opens in **Real photo** mode, which
places the design's own catalogue front render (view 4) on the saved photo instead of the 3D
model: it looks real and is exactly the design. `/api/tryon/photo/<uid>?metal=…` serves the
render trimmed and scaled (cached in `data/tryon/photos/`); the page removes its soft floor
shadow, splits an earring pair (one per lobe), and for a ring removes the back of the band
(the part under the finger) while keeping the head whole. **3D** stays available with a toggle
and is still what the live camera uses. Known limit: a ring whose wide gold head has no
diamonds (e.g. open loops) can be clipped at its outer edges.

**Check every model against its design:** `.venv/bin/python scripts/check_tryon_models.py`
(about 20 minutes on the GPU, only new or changed models on later runs). Each model is
rendered on white and compared with its own catalogue front photo using the search model
(SigLIP2). A card shows **Try on** only when its model passed (similarity ≥ 0.78), because a
model that doesn't look like the design is worse than no try-on. The result is published to
`data/tryon/fidelity.json` only when every model has been checked, and the server picks it up
without a restart. Run it after every `build_tryon_models.py` run: until then, new models
stay hidden.

## Dataset notes

- File names follow `<design_id>[-NN]@<R|W|Y>-#<viwe|view><N>.<png|mp4>`:
  R/W/Y = rose/white/yellow gold, views 1-4 are PNG renders, view 5 is the
  turntable video. "viwe" is the spelling used in most files.
- The same design often exists in several folders (e.g. a `PHOTOS` copy);
  records with coded ids (`DDLR-123`, `12#10003P`) are merged. Bare-number
  ids (`11`, `1 (303)`) are reused by unrelated batches, so they are kept
  per folder.
- Single-photo designs (`1 (NNN)` in the `PHOTOS` batches) only have a
  three-quarter view, so no front view exists for them.
- Folders named "NOT UPLOAD" are the owner's e-commerce label and are indexed
  like everything else; the name is not used for categorisation.
- Type is read from the design id prefix first, then folder keywords, and
  the image model fills in the rest (`category_source` in the index says
  which).
