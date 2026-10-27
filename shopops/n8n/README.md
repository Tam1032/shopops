# n8n workflows

Two workflows, tested on n8n 2.44.7 (`n8n import:workflow` + `n8n execute`; the
Google Sheets and Telegram nodes were disabled in that test because they need
credentials).

| File | Trigger | What it does |
|---|---|---|
| `triage_reviews.json` | every 15 min, and the marketplace webhook `POST /webhook/shopops-review` | lists new reviews, calls `POST /triage/{id}` on the agent, appends a row to the approval sheet, sends a Telegram alert when the agent proposes escalation |
| `approve_actions.json` | every 5 min | reads the sheet, runs rows with `approved = yes` and `status = pending` through `POST /actions/{id}/approve` (the edited `draft` column becomes the reply text), marks them `done` |

## Setup

1. `docker compose up --build` from the repo root (n8n runs at http://localhost:5678).
2. Create a Google Sheet with a tab named `queue` and this header row:
   `action_id | review_id | review | sentiment | topic | proposed | draft | approved | status`
3. In n8n: import both files (Workflows → Import from file), add Google Sheets
   OAuth2 and Telegram credentials, and replace `YOUR_SHEET_ID` and `YOUR_STAFF_CHAT_ID`.
4. Activate both workflows, then `make demo` to make the marketplace post 3 new reviews.

Staff review drafts in the sheet, edit the `draft` cell if needed, and type `yes`
in `approved`. Nothing reaches customers before that.

The HTTP nodes use the compose service names (`marketplace`, `agent`); change them
if n8n runs elsewhere.
