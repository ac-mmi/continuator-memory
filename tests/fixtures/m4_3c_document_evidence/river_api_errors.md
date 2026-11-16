# River API Error Catalog

River returns structured JSON errors with HTTP status codes.

## Client Errors

### HTTP 400

HTTP 400 means the request body failed schema validation.

### HTTP 429

HTTP 429 means the client exceeded the River rate limit of 100 requests per minute.

## Server Errors

### HTTP 503

HTTP 503 means a dependency is unavailable; clients should retry with exponential backoff.
