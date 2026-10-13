# Brand catalogue — `dataset/brands/`

Source of truth for the typosquatting layer (`src/typosquatting/`, Damerau-Levenshtein +
Jaro-Winkler) and for the verified-official-domain shortcut of the pipeline (`src/pipeline.py`).

| File | Content |
|---|---|
| `vn_brand_domains.csv` | Vietnamese brands frequently impersonated by phishing, with official domains |
| `brands.csv` | International brands (Zenodo catalogue, 86 brands) |
| `brands_extra.csv` | International brands added by this project (gaming, crypto, banks, payments, logistics, travel...) |
| `secondary_official_domains.csv` | Legitimate secondary domains of brands already listed (CDN, infrastructure, regional sites, group domains) |

## `vn_brand_domains.csv`

### Columns

| Column | Meaning |
|---|---|
| `brand` | Label used for string matching (usually the first part of the main domain, no accents, no TLD, letters/digits only). **Unique.** |
| `domain` | One official domain. A brand may have several rows (e.g. `viettel.vn` + `viettel.com.vn`). |
| `category` | `bank` \| `e_wallet` \| `ecommerce` \| `telecom` \| `airline` \| `public_service` \| `logistics` \| `insurance` \| `securities` \| `consumer_finance` \| `transport` \| `conglomerate` |
| `verified` | `verified` = name **and** domain checked against an official source · `unverified` = not yet |
| `note` | Rename / domain change / warnings (free text, EnglishT). |

A brand is `verified` only when every one of its domains is verified; only verified domains are
trusted by the pipeline shortcut (`verified_official_domains()`: such a page is labelled clean
without further checks, except marketplaces, user content and URLs that could redirect).

### Verification procedure
1. **Primary source = the State Bank of Vietnam (SBV)**, the licensing authority (not a third party
   re-telling it):
   - Banks: `sbv.gov.vn` — "List of state-owned commercial banks" / "List of domestic joint-stock
     commercial banks" (latest found: **30/9/2024**, 31 domestic joint-stock banks).
   - E-wallets / payment intermediaries: `sbv.gov.vn` — "List of non-bank organisations licensed to
     provide payment intermediary services" (latest: **09/12/2024**, ~50 organisations).
2. **Domain**: confirmed on the brand's own website (footer links / contact page).
3. **Secondary sources, for quick lookup only — NOT final verification**: Vietnamese Wikipedia "List
   of banks in Vietnam" (links to Wayback copies of the SBV pages).

Brands added on 2026-10-08 (logistics, insurance, securities, consumer finance, ride hailing,
more banks and public services) were verified by reading the official homepage title
(`verified`) or by a web search naming the official site; domains that block automated access
(Manulife, VNDirect, TCBS, Vingroup/VinFast/Vinhomes, SBV) or could not be reached (NCSC) are
`unverified` and need a manual check.

### Notable renames and additions
- **Sacombank** — SBV decision 36/QĐ-QLGS4 (01/6/2026) renamed the registered name from "Sài Gòn
  Thương Tín" to **"Sài Gòn Tài Lộc"** (EN: Saigon Treasure Commercial Joint Stock Bank). The short
  name stays **SACOMBANK** and the domain `sacombank.com.vn` is unchanged (notice on
  sacombank.com.vn), so label and domain are kept; only the note records the name history.
- **LPBank** — formerly LienVietPostBank; SBV approved the short name **LPBank** (2023). The draft row
  `lienvietpostbank` was replaced by `lpbank` / `lpbank.com.vn`.
- **DongA Bank → Vikki Bank** (from 14/02/2025, decisions 42/QĐ-TTGSNH2 and 237/QĐ-NHNN) after
  the compulsory transfer to HDBank. New domain `vikkibank.vn`.
- **Ngân hàng Bản Việt → BVBank** (rebranded 2022), domain `bvbank.net.vn`.
- **Kienlongbank** (`kienlongbank.com`) and **PGBank** (`pgbank.com.vn`, formerly Petrolimex Bank) —
  two of the 31 SBV-listed banks missing from the first draft.
- **NAPAS** (`napas.com.vn`) — national interbank switching; a high-value impersonation target.
- **Moca** — still licensed by SBV (30/4/2026), but the consumer wallet on Grab stopped taking new
  users and refunded balances from 1/7/2024; kept with a note on its lower relevance.
- **SCB** — under special control since 10/2022 but operating normally as of 10/2026; head office
  moved 09/2026, domain `scb.com.vn` unchanged.
- **GDT (tax authority)** — renamed "Cục Thuế" (under the Ministry of Finance) but `gdt.gov.vn` is
  unchanged; e-tax services moved from `thuedientu.gdt.gov.vn` to `dichvucong.gdt.gov.vn`
  (from 01/7/2025), same registrable domain.
- **MBV** — new name of OceanBank (`mbv.com.vn`).
- **Xanh SM → Green SM** — `xanhsm.com` now redirects to `greensm.com`; both are listed.
- **VNPT / Viettel** — extra domains `vnpt.vn` and `vietteltelecom.vn`.

## `brands.csv` and `brands_extra.csv` (international)

Columns: `category`, `description`, `identifier` (label used for matching, alphanumeric),
`name`, `website`. The registrable domain of `website` is the official domain. These brands are
matched with stricter rules than Vietnamese ones (an exact label on another TLD is not flagged;
a brand embedded in a longer domain label needs a lure word such as "login" or "support").

## `secondary_official_domains.csv`

Legitimate domains owned by a brand already listed (e.g. `googleapis.com`, `shopee.co.th`,
`viettelpost.vn`, `hsbc.com`). Without this file the typosquatting rule would flag them as
"brand embedded in domain" or "exact brand label".

- Columns: `brand` (must match an existing label), `domain`, `verified` (`verified` \|
  `unverified`), `note`.
- Used only as an **exemption list** (typosquatting rule, LLM override) — never by the trusted-domain
  shortcut, which only takes verified Vietnamese domains.
- A wrong domain here makes the rule miss exactly that domain, so every `unverified` row should be
  checked (WHOIS / the brand's own site) before it becomes `verified`.
