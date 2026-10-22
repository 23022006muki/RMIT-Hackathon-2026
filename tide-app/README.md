# Signal Commons — RMIT Hackathon 2026, Challenges 1 × 2

Signal Commons is a multilingual AI safety research commons. A contributor submits an English, Portuguese, French or Italian prompt. The single Kaggle-trained Challenge 1 model classifies it as `BENIGN`, `JAILBREAK` or `PROMPT_INJECTION`. TideCloak then self-encrypts the original in the contributor's browser; the app stores ciphertext and classification metadata. Researchers can see aggregate patterns without receiving a central archive of raw prompts. A contributor can retrieve and decrypt their own original.

The trust constraint is enforced by TideCloak identity, role-gated encryption and server verification of access tokens. The app server cannot decrypt stored ciphertext by itself. The classifier does receive plaintext transiently during inference, so this prototype protects **stored** originals, not end-to-end privacy from the classifier service. TideCloak matters because a normal app administrator should not be the sole key holder for contributed research material. This is a working path through classification, sealing, aggregate statistics and owner retrieval, rather than a mock dashboard.

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
| `GET /api/stats` | Public | Returns aggregate counts only. |

The JSON store is a local prototype backend. A local signing key is generated in `.data/receipt.key` so classification receipts remain valid across API routes and restarts. Use `RECEIPT_SIGNING_KEY` (at least 32 random bytes encoded as hex) plus a transactional database and durable single-use receipt ledger before running multiple server replicas. No prompt, token or model checkpoint is intended for Git. `.data`, `.env.local` and `node_modules` are ignored.

## Verify

```powershell
npm run typecheck
npm run build
```

Positive Tide login/encryption and Kaggle model inference require the server, enrollment and final checkpoint above. The app refuses to substitute a local-only or non-Kaggle model.
