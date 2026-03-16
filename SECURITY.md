# Security Policy

## Supported Versions

| Version         | Supported |
| --------------- | --------- |
| `main` (latest) | ✅        |
| `develop`       | ✅        |
| Older branches  | ❌        |

## Scope

This project is a research and demonstration codebase for ML/hardware co-design. It does not expose network services, user authentication, or sensitive data processing. The primary security concerns are:

- **Dependency vulnerabilities** in PyTorch, torchvision, Amaranth, or Pydantic.
- **Unsafe deserialization** via `torch.load()` (mitigated by `weights_only=True`).
- **Arbitrary code execution** via malicious model checkpoints or dataset files.

## Reporting a Vulnerability

Do **not** open a public GitHub issue for security vulnerabilities.

Instead, use one of the following:

1. **GitHub Private Security Advisory** — preferred:
   Go to `Security` → `Advisories` → `Report a vulnerability` on the repository page.

2. **Direct contact** — via GitHub profile if the advisory mechanism is unavailable.

Please include:

- A clear description of the vulnerability.
- Steps to reproduce.
- The potential impact.
- A suggested fix if available.

You will receive an acknowledgment within **72 hours** and a resolution timeline within **7 days** for confirmed issues.

## Security Practices in This Codebase

- `torch.load()` is called with `weights_only=True` to prevent arbitrary code execution from untrusted checkpoints.
- No external network calls are made at runtime outside of dataset download functions.
- Dataset download URLs are hardcoded to known official sources (PyTorch mirrors, Stanford CS231n CDN).
- No credentials, API keys, or personal data are stored or processed.
