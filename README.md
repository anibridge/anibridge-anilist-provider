# anibridge-anilist-provider

An [AniBridge](https://github.com/anibridge/anibridge) provider for [AniList](https://anilist.co/).

_This provider comes built-in with AniBridge, so you don't need to install it separately._

## Configuration

```yaml
list_provider_config:
  anilist:
    token: ...
    # prefetch_list: false
```

### `token`

`str` (Required)

Your AniList API token. You can generate one [here](https://anilist.co/login?apiVersion=v2&client_id=34003&response_type=token).

### `prefetch_list`

`bool` (Optional, default: `False`)

Whether to prefetch and cache the user's list on startup. It is recommended to keep this disabled if you have AniBridge backups enabled (the default behavior), as that performs an equivalent prefetch on startup anyway.
