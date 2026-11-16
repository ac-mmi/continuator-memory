# Aurora Authentication Guide

Aurora services authenticate browser and mobile clients with OAuth 2.0.

## OAuth Providers

Supported identity providers are Google Workspace and Okta Workforce.

## Token Lifecycle

### Access Token Refresh

Access tokens expire after one hour. Clients must refresh using the refresh token before expiry.

### Refresh Token Storage

Refresh tokens are stored encrypted at rest in the Aurora session vault.

## Troubleshooting Sign-In

If sign-in loops, clear the local session cookie named `aurora_sid` and retry.
