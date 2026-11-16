# Pine Incident Runbook

Pine on-call engineers follow severity-based response windows.

## Severity Definitions

### SEV1

SEV1 is a full production outage. Acknowledge within 5 minutes and begin mitigation immediately.

### SEV2

SEV2 is major degradation. Acknowledge within 15 minutes.

## Escalation

### Primary Contacts

Page the primary on-call via PagerDuty schedule `pine-primary`. If no answer in 10 minutes, page `pine-secondary`.
