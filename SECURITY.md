# Security Policy

## Reporting a vulnerability

Report a vulnerability privately through GitHub: open the repository's **Security** tab and
choose **Report a vulnerability**. Please do not open a public issue for it.

Include what you found, how to reproduce it, and which version or commit you used. You
will get an answer in that private report.

## Scope

CogniBoiler is a portfolio project meant to run on one machine with Docker Compose. Its
published ports bind 127.0.0.1, and the secrets in `.env` are generated per machine by
`python dev_tools_scripts_runner.py dev-secrets`. Reports about deploying the published
images elsewhere are welcome too.
