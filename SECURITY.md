# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| latest  | :white_check_mark: |

## Reporting a Vulnerability

Med-Psych RAG handles medical and psychological data. Security is critical.

If you discover a security vulnerability, please do **not** open a public issue.

Instead, email the maintainer directly at **amir.asaad@example.com** with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact

You should receive a response within 48 hours. If you do not, please follow up.

We will acknowledge receipt, investigate, and work on a fix before disclosing the issue publicly.

## Best Practices for Deployment

- Always use environment variables or a `.env` file for secrets (see `.env.example`).
- Never commit `.env` or any file containing real API keys.
- Run the service behind a reverse proxy (e.g., nginx) with TLS in production.
- Restrict network access to the Ollama endpoint and ChromaDB storage.
- Regularly update dependencies to patch known vulnerabilities.
