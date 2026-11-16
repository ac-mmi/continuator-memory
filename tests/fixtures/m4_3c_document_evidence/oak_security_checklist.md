# Oak Security Checklist

Oak platform security controls for service-to-service credentials.

## API Key Management

### Key Rotation

Service API keys must be rotated every 90 days. Rotation does not use OAuth refresh tokens.

### Key Storage

API keys are stored in the Oak secrets manager, never in application source.

## Audit Logging

All key issuance events are written to the Oak security audit log.
