"""Multiciudad: catálogo, rutas aisladas por ciudad y parámetro ?city= de la API."""

import pytest

from aqi import api
from aqi.config import UnknownCityError, get_settings, slugify


def _other_slug() -> str:
    s = get_settings()
    return next(c.slug for c in s.cities if c.slug != s.city.slug)


def test_catalog_has_unique_slugs_and_includes_default():
    s = get_settings()
    slugs = [c.slug for c in s.cities]
    assert len(slugs) == len(set(slugs)) >= 2 and s.city.slug in slugs


def test_each_city_has_isolated_paths():
    s = get_settings()
    per_city = [s.for_city(c.slug).paths for c in s.cities]
    for attr in ("raw_file", "processed_file", "champion_dir", "reports_dir"):
        assert len({getattr(p, attr) for p in per_city}) == len(per_city)


def test_slugify_strips_accents_and_symbols():
    assert slugify("Ñuñoa Perú") == "nunoa-peru"


def test_unknown_city_raises():
    with pytest.raises(UnknownCityError):
        get_settings().for_city("atlantis")


def test_cities_endpoint_matches_contract(client):
    body = client.get("/cities").json()
    assert {c["slug"] for c in body} == {c.slug for c in get_settings().cities}
    assert sum(c["default"] for c in body) == 1
    assert all(set(c) == {"slug", "name", "latitude", "longitude", "available", "default"} for c in body)


def test_unknown_city_is_404_with_detail(client):
    r = client.get("/model-info?city=atlantis")
    assert r.status_code == 404 and "Available" in r.json()["detail"]


def test_city_without_model_is_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "champion_dir", lambda slug=None: tmp_path)
    r = client.get(f"/model-info?city={_other_slug()}")
    assert r.status_code == 503 and isinstance(r.json()["detail"], str)


def test_each_city_is_served_by_its_own_champion(client, trained_champion, monkeypatch):
    other = _other_slug()
    monkeypatch.setattr(api, "champion_dir", lambda slug=None: trained_champion[0])
    r = client.get(f"/model-info?city={other}")
    assert r.status_code == 200 and r.json()["city"] == get_settings().for_city(other).city.name
    assert client.get("/model-info").json()["city"] == get_settings().city.name
