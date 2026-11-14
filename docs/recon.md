# Site recon — taladrod.com (14 Nov 2026)

Inspected with a real browser (Claude desktop built-in browser) plus same-origin `fetch()`.

## Listing page `https://www.taladrod.com/w40/isch/schc.aspx?fno:new`

- Server-rendered HTML, `charset=utf-8` (no TIS-620 issues).
- **All results are embedded in the HTML** as one inline script:
  `var SchDataJSON={"cars":[...], "ncar":"227", ...};var CacheName=...`
  `cars` is in display order (newest first). On 14 Nov it held 224–227 cars.
- **No server-side pagination.** `schcc_js.js` renders the cars 10 rows at a time
  (`pageSize = 10`, `pageNum++` on scroll, infinite scroll). Tried `pg:2`, `p:2`,
  `page:2`, `pn:2`: none change the result. So "pagination" = walking the embedded
  list page by page, in display order, until `MAX_ADS` or the data ends.
- Useful fields per car: `cid` (Ad ID), `yr4`, `namemmt` (e.g. `BMW X1 2.0 sDrive20d M SPOR F48 ปี16-23`),
  `model`, `prc` (`"649,000"`), `title` (seller headline), `img` (thumbnail), `type` (`car`; `upg` rows are promos and get skipped).
- Detail URL: `https://www.taladrod.com/w40/icar/cardet.aspx?cid={cid}`.
- **No body-type field** anywhere (the `bt` filter is price (baht), not body type). Categorization has to come from a model dictionary.

## Detail page `.../icar/cardet.aspx?cid=...`

- Title block: `<div class='CD_MMT_Title'>` → year in the first `<span>`, then
  `"BMW X1, 2.0 sDrive20d M SPORT"` (brand = first word, model = up to the comma,
  trim = after the comma), then `โฉม ...` (generation).
- Mileage: plain text `เลขไมล์ 101,000 กม.` inside the info `<div>`.
- Price: `<b class='prcTxt' id='xPrice'>649,000</b>`.
- Ad ID echo: `รหัสอ้างอิง 2932903`.
- Images: the first one is a plain `<img>`; the rest are `img.cdImgCar` inside `#cdImgCarLowerContainer`.
  Full-size URL pattern is `https://imgc2.taladrod.com/c/cidx/014/089/74_{N}.jpg?v=...`
  (N = 1..15 seen). `og:image` gives the `_1.jpg` URL, so the ad's prefix is known.
  The same page also shows *other* cars' images ("similar cars", `_1T3.jpg` thumbnails), so
  the parser keeps only `<img>` whose path matches the ad's own prefix + `_{N}.jpg`.

## Anti-bot behaviour (important)

- The site sits behind **AWS WAF on CloudFront**. After ~20 detail requests at ~0.6 s
  intervals, responses became `HTTP 202`, empty body, header `x-amzn-waf-action: challenge`.
- The pipeline therefore: waits 1.5–3.0 s between requests, treats 202/429/5xx and
  WAF challenges as retryable with exponential backoff, and if the challenge persists
  it **stops cleanly and keeps the checkpoint** so the run can be resumed later.
  It does **not** try to solve or bypass the challenge.

## Fixtures (`tests/fixtures/`)

- `list_new.html`: real listing HTML trimmed to 11 cars.
- `detail_15img.html`: real detail page for cid 2932903 (15 images), finance widget removed.
- `detail_8img.html`: real detail page for cid 2932854 (8 images, VIP layout).
- `detail_no_mileage.html`: derived from `detail_8img.html` with the mileage line removed
  (no real ad without mileage was found in the sample).
