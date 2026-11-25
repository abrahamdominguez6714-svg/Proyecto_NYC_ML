"""Carga de datos: descarga desde la API de NYC Open Data (con caché local) y armado de X, y.

Cambios respecto a la versión de clase:
  * `load_xy` ahora también devuelve `created_date` y deja todo ORDENADO por esa fecha:
    sin ese orden no se puede hacer una validación temporal.
  * Se verifica que ninguna variable prohibida (closed_date, Diff, el target, unique_key)
    se cuele entre las predictoras.
  * La caché lleva las fechas en el nombre (ver config.CACHE).
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    API, CACHE, COLUMNS, FEATURES, FORBIDDEN_FEATURES, MAX_ROWS, PAGE_SIZE,
    QUERY, TARGET, TIME_COL,
)
from src.target import make_target


class Dataset:
    """Encapsula la obtención y preparación de los datos."""

    def load_data(self) -> dict:
        """Devuelve el payload {api, query, downloaded_at, rows}. Usa la caché si existe."""
        if CACHE.exists():
            payload = json.loads(CACHE.read_text(encoding="utf-8"))
            if payload["query"] != QUERY or payload["api"] != API:
                raise ValueError(
                    "La caché no corresponde a esta consulta. Bórrala o renómbrala para descargar de nuevo."
                )
            return payload

        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        token = os.environ.get("SOCRATA_APP_TOKEN")  # opcional: sube el límite de peticiones
        if token:
            session.headers["X-App-Token"] = token

        rows = []
        for offset in range(0, MAX_ROWS, PAGE_SIZE):
            response = session.get(
                API, params={**QUERY, "$limit": PAGE_SIZE, "$offset": offset}, timeout=90)
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise ValueError("Respuesta inesperada de Socrata.")
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                break
        else:
            raise ValueError("Se alcanzó el límite de seguridad; no se guarda una muestra truncada.")
        if not rows:
            raise ValueError("La API no devolvió filas. Revisa la consulta.")

        payload = {"api": API, "query": QUERY,
                   "downloaded_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE.with_suffix(".tmp")           # se escribe a un temporal y luego se renombra:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(CACHE)                        # así una descarga interrumpida no deja caché corrupta
        return payload

    def load_df(self) -> pd.DataFrame:
        """Datos crudos como DataFrame (solo las columnas pedidas a la API)."""
        payload = self.load_data()
        return pd.DataFrame(payload["rows"]).reindex(columns=COLUMNS)

    def load_df_target(self) -> pd.DataFrame:
        """Datos con target, filtrados y ordenados cronológicamente."""
        df = make_target(self.load_df())
        return df.sort_values([TIME_COL, "unique_key"], kind="stable").reset_index(drop=True)

    def load_xy(self):
        """Devuelve (X, y, created): predictoras, target y fecha de creación, todo alineado y ordenado."""
        leaked = set(FEATURES) & FORBIDDEN_FEATURES
        if leaked:
            raise ValueError(f"Variables prohibidas entre las predictoras (fuga de información): {leaked}")
        df = self.load_df_target()
        return df[FEATURES], df[TARGET], df[TIME_COL]
