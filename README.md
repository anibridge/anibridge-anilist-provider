# anibridge-anilist-provider

An [AniBridge](https://github.com/anibridge/anibridge) provider for [AniList](https://anilist.co/).

_This provider comes built-in with AniBridge, so you don't need to install it separately._

## Configuration

```yaml
list_provider_config:
  anilist:
    token: ...
    # rate_limit: null
```

### `token`

`str` (required)

Your AniList API token. You can generate one [here](https://anilist.co/login?apiVersion=v2&client_id=34003&response_type=token).

### `rate_limit`

`int | None` (optional, default: `null`)

The maximum number of API requests per minute.

If unset or set to `null`, the provider will use a default _global_ rate limit of 30 requests per minute. It is important to note that this global rate limit is shared across all AniList provider instances, i.e. they collectively use 30 requests per minute. If you override the rate limit, a new rate limit, local to the provider instance, will be created.
