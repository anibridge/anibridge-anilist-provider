# anibridge-anilist-provider

An [AniBridge](https://github.com/anibridge/anibridge) provider for [AniList](https://anilist.co/).

_This provider comes built-in with AniBridge, so you don't need to install it separately._

## Configuration

### `token` (`str`)

Your AniList API token. You can generate one [here](https://anilist.co/login?apiVersion=v2&client_id=34003&response_type=token).

```yaml
list_provider_config:
  anilist:
    token: ...
```
