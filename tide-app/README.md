# Signal Commons — RMIT Hackathon 2026, Challenges 1 × 2

Signal Commons lets researchers share multilingual AI safety signals while contributors retain control of their original prompts through TideCloak.

## Project Overview

- **Vision & originality:** A contributor submits a prompt in Portuguese, French, Italian or English. The single Kaggle-trained Challenge 1 model labels it `BENIGN`, `JAILBREAK` or `PROMPT_INJECTION`; the same app turns the result into a private, shareable research signal.
- **Trust constraint insight:** A central archive of raw adversarial prompts gives its operator unilateral custody of sensitive submissions. Signal Commons stores encrypted originals and publishes only label and language counts.
- **Value & impact:** Researchers can spot cross-language attack patterns. Contributors can retrieve their own original and withdraw their contribution from the commons.
- **Why TideCloak matters:** TideCloak verifies identity and encrypts the original in the contributor's browser. The server checks the Tide access token for each private API action and cannot decrypt the stored ciphertext by itself.
- **Prototype & storytelling:** Sign in, classify a fresh prompt with the Kaggle checkpoint, encrypt and save it, inspect aggregate counts, decrypt as the owner, show an unauthenticated API denial, then withdraw it. The classifier sees plaintext transiently for inference; this prototype protects stored originals, not inference-time text. Withdrawal removes the app's copy and aggregate entry; it is not cryptographic key revocation.

## Run locally

Requirements: Node.js 20.9+, Python environment from the repository root, a completed **Kaggle-trained** Challenge 1 checkpoint, and a running TideCloak server. Start a server using the [official TideCloak getting-started repository](https://github.com/tide-foundation/tidecloak-gettingstarted). The scaffold's provisioning script needs Bash, `curl` and `jq`.

1. Keep the realm/client names in `init/realm.json` aligned with the initializer defaults (`nextjs-test` / `myclient`). Set `SUBSCRIPTION_EMAIL` in your shell rather than editing the tracked example file. Change default development admin credentials before any nonlocal deployment.
2. From `tide-app`, run `npm ci`. On Linux/macOS, run `npm run init`; on Windows, run the Git Bash command below because the default `bash.exe` may point to WSL. Complete the enrollment URL shown by the script; it writes `tidecloak.json`. The realm grants `appUser` plus `_tide_message.selfencrypt` and `_tide_message.selfdecrypt` to contributors.
3. Set `PROTECTOR_CHECKPOINT` to the final checkpoint **downloaded from the completed Kaggle notebook**. The checkpoint must contain `training_origin.json` with `platform: kaggle`, `protector_config.json`, model files and tokenizer files. Optionally set `PROTECTOR_PYTHON` to the Python executable in the repository's `.venv`.
4. Run `npm run dev` and open `http://localhost:3000`. Sign in with TideCloak, submit a prompt, and use **Decrypt my original** in your private list.

PowerShell example for step 3:

```powershell
$env:PROTECTOR_CHECKPOINT = 'D:/RMIT Hackathon 2026/project/artifacts/kaggle_final/checkpoint'
$env:PROTECTOR_PYTHON = 'D:/RMIT Hackathon 2026/.venv/Scripts/python.exe'
npm run dev
```

For initialization in PowerShell with Git Bash installed, set `$env:SUBSCRIPTION_EMAIL = 'you@example.com'` and run `& 'C:\Program Files\Git\bin\bash.exe' init/tcinit.sh` after installing `jq` in your `PATH`.

Use the actual downloaded checkpoint directory; the example name may differ. The app runs inference only and never trains a competition model locally.

## Data and API boundaries

| Endpoint | Access | Content |
| --- | --- | --- |
| `POST /api/classify` | Verified Tide token + `appUser` | Plaintext arrives transiently; returns a label and short-lived, single-use receipt. |
| `POST /api/cases` | Verified Tide token + receipt | Stores ciphertext and metadata in `.data/cases.json`. |
| `GET /api/cases` | Verified Tide token | Returns only the caller's metadata. |
| `GET /api/cases/:id` | Verified owner token | Returns the owner's ciphertext for browser decryption. |
| `DELETE /api/cases/:id` | Verified owner token | Removes the contribution and its aggregate count; later reads return 404. |
| `GET /api/stats` | Public | Returns aggregate counts only. |

The JSON store is a local prototype backend. A local signing key is generated in `.data/receipt.key` so classification receipts remain valid across API routes and restarts. Use `RECEIPT_SIGNING_KEY` (at least 32 random bytes encoded as hex) plus a transactional database and durable single-use receipt ledger before running multiple server replicas. No prompt, token or model checkpoint is intended for Git. `.data`, `.env.local` and `node_modules` are ignored.

## Verify

```powershell
npm run typecheck
npm run build
```

Positive Tide login/encryption and Kaggle model inference require the server, enrollment and final checkpoint above. The app refuses to substitute a local-only or non-Kaggle model.
