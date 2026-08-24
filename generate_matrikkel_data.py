#!/usr/bin/env python3
"""End-to-end pipeline that produces an address <-> bygningsnummer crosswalk
entirely from Geonorge's Matrikkelen datasets. Consolidates four scripts that
were developed and run separately in geonorge/ into one file:

  1. download_dataset()        resumable bulk download via the Geonorge
                                nedlasting order API      (was download_matrikkel.py)
  2. bygning_fgdb_to_csv()     Matrikkelen - Bygningspunkt -> bygningsnummer
                                + WGS84 lat/lon CSV        (was matrikkel_to_csv.py)
  3. adresse_fgdb_to_csv()     Matrikkelen - Adresse -> matrikkelnummer
                                (kommunenummer-gnr/bnr/seksjonsnr/festenr) +
                                street address text + WGS84 lat/lon CSV
                                                            (was matrikkel_adresse_to_csv.py)
  4. match_bygningsnummer()    nearest-neighbor spatial match: attaches the
                                nearest building's bygningsnummer + distance
                                to every address, since neither dataset
                                carries a field the other can join on
                                                            (was match_bygningsnummer_via_adresse.py)

Run the whole pipeline:
    python generate_matrikkel_data.py --email you@example.com

Run one stage at a time (e.g. to resume after an interrupted download, or
re-run a later stage without re-downloading):
    python generate_matrikkel_data.py --stage download-bygning --email you@example.com
    python generate_matrikkel_data.py --stage download-adresse --email you@example.com
    python generate_matrikkel_data.py --stage bygning-csv
    python generate_matrikkel_data.py --stage adresse-csv
    python generate_matrikkel_data.py --stage match

Downloads are resumable: Ctrl+C finishes the current 1MB chunk, saves state,
and exits; re-running the identical stage picks up where it stopped rather
than placing a second order (Geonorge generates national extracts on demand,
so a second order means waiting for several GB to regenerate from scratch).

NOT included: joining the flood-depth / water-body / spurious-depth-area
.tif rasters (RP10..RP500 etc). Those come from a different source (JRC/
Copernicus European flood hazard maps, clipped to Norway) -- not Geonorge.
That step lives separately in join_norway_adresse.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyogrio
import pyogrio.raw
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- config

NEDLASTING_API = "https://nedlasting.geonorge.no/api"
DOWNLOAD_CHUNK = 1 << 20
STATE_VERSION = 1

BYGNING_UUID = "24d7e9d1-87f6-45a0-b38e-3447f8d7f9a1"   # Matrikkelen - Bygningspunkt
ADRESSE_UUID = "f7df7a18-b30f-4745-bd64-d0863812350c"   # Matrikkelen - Adresse

# Default locations -- match what earlier pipeline runs in this project produced.
BASE_DIR = Path(r"c:\Users\OPL\Documents\pyth")
GEONORGE_DIR = BASE_DIR / "geonorge"
BYGNING_DATA_DIR = GEONORGE_DIR / "data"
ADRESSE_DATA_DIR = GEONORGE_DIR / "data_adresse"
BYGNING_ZIP = BYGNING_DATA_DIR / "Basisdata_0000_Norge_25833_MatrikkelenBygning_FGDB.zip"
ADRESSE_ZIP = ADRESSE_DATA_DIR / "Basisdata_0000_Norge_25833_MatrikkelenAdresse_FGDB.zip"
BYGNING_CSV = BASE_DIR / "matrikkel_bygning.csv"
ADRESSE_CSV = GEONORGE_DIR / "matrikkel_adresse.csv"
MATCH_CSV = BASE_DIR / "adresse_bygningsnummer_match.csv"

ADRESSE_LAYERS = ["vegadresse", "matrikkeladresse"]
ADRESSE_ID_FIELDS = [
    "matrikkelnummeradresse_kommunenummer",
    "matrikkelnummeradresse_gardsnummer",
    "matrikkelnummeradresse_bruksnummer",
    "matrikkelnummeradresse_seksjonsnummer",
    "matrikkelnummeradresse_festenummer",
    "adressetekstutenadressetilleggsnavn",
]

# Little-endian 2D WKB point: byte order + geometry type + x + y.
WKB_POINT_LE_2D = np.dtype(
    [("order", "u1"), ("type", "<u4"), ("x", "<f8"), ("y", "<f8")]
)

_stop = False


def _on_sigint(signum: int, frame: Any) -> None:
    global _stop
    if _stop:  # second Ctrl+C -> give up immediately
        raise KeyboardInterrupt
    _stop = True
    print("\n[!] Finishing current chunk, then stopping. Ctrl+C again to force.",
          file=sys.stderr)


# =============================================================================
# 1. Download -- Geonorge nedlasting order API
# =============================================================================


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": STATE_VERSION, "order": None, "files": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != STATE_VERSION:
        raise SystemExit(f"{path} was written by an incompatible version; delete it to start over.")
    return state


def _save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _order_key(uuid: str, area: str, projection: str, fmt: str) -> str:
    """Identifies the order parameters, so changing them starts a fresh order."""
    raw = "|".join([uuid, area, projection, fmt])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _validate_selection(session: requests.Session, uuid: str, area: str,
                         projection: str, fmt: str) -> None:
    """Fail early on combinations the API does not offer (e.g. national FGDB in 25832)."""
    areas = session.get(f"{NEDLASTING_API}/codelists/area/{uuid}", timeout=30).json()
    entry = next((a for a in areas if a["code"] == area), None)
    if entry is None:
        raise SystemExit(f"Area {area!r} is not available for {uuid}.")

    offered = {
        f["name"]: {p["code"] for p in f.get("projections", [])}
        for f in entry.get("formats", [])
    }
    if fmt not in offered:
        raise SystemExit(f"Format {fmt!r} not offered for area {area!r}. "
                          f"Available: {', '.join(sorted(offered)) or 'none'}")
    if projection not in offered[fmt]:
        raise SystemExit(f"Projection {projection!r} not offered for {fmt} in area {area!r}. "
                          f"Available: {', '.join(sorted(offered[fmt]))}")

    print(f"[ok] {entry['name']} / {fmt} / EPSG:{projection}")


def _place_order(session: requests.Session, uuid: str, area: str, projection: str,
                  fmt: str, email: str) -> str:
    areas = session.get(f"{NEDLASTING_API}/codelists/area/{uuid}", timeout=30).json()
    entry = next(a for a in areas if a["code"] == area)

    payload = {
        "email": email,
        "orderLines": [{
            "metadataUuid": uuid,
            "areas": [{"code": entry["code"], "type": entry["type"], "name": entry["name"]}],
            "projections": [{
                "code": projection,
                "name": f"EPSG:{projection}",
                "codespace": f"http://www.opengis.net/def/crs/EPSG/0/{projection}",
            }],
            "formats": [{"name": fmt}],
        }],
    }
    resp = session.post(f"{NEDLASTING_API}/order", json=payload, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    ref = body.get("referenceNumber") or body.get("ReferenceNumber")
    if not ref:
        raise SystemExit(f"No reference number in order response: {body}")
    return ref


def _fetch_order(session: requests.Session, ref: str) -> list[dict]:
    resp = session.get(f"{NEDLASTING_API}/order/{ref}", timeout=60)
    resp.raise_for_status()
    files = resp.json().get("files", [])

    out = []
    for f in files:
        url = f.get("downloadUrl") or f.get("DownloadUrl")
        if not url:
            continue
        name = f.get("name") or url.rsplit("/", 1)[-1].split("?")[0]
        out.append({
            "name": name,
            "url": url,
            "status": f.get("status", "ReadyForDownload"),
            "size": f.get("fileSize") or f.get("size"),
        })
    return out


def _wait_until_ready(session: requests.Session, ref: str, state: dict,
                       state_path: Path, poll: int) -> list[dict]:
    """Poll the order until Geonorge has finished generating every file."""
    while not _stop:
        files = _fetch_order(session, ref)
        state["files"] = {f["name"]: f for f in files}
        _save_state(state_path, state)

        if files and all(f["status"] == "ReadyForDownload" for f in files):
            return files

        pending = [f for f in files if f["status"] != "ReadyForDownload"]
        label = f"{len(pending)} of {len(files)} still processing" if files else "order queued"
        print(f"[..] {label}; re-checking in {poll}s")

        for _ in range(poll):
            if _stop:
                break
            time.sleep(1)
    return []


def _human(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _download_file(session: requests.Session, entry: dict, outdir: Path,
                    timeout: int) -> bool:
    """Download one file, resuming a .part if present. Returns True when complete."""
    dest = outdir / entry["name"]
    part = dest.with_name(dest.name + ".part")
    expected = entry.get("size")

    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        print(f"[ok] {dest.name} already downloaded ({_human(dest.stat().st_size)})")
        return True

    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}

    with session.get(entry["url"], headers=headers, stream=True, timeout=timeout) as r:
        if done and r.status_code == 200:
            print(f"[!!] {dest.name}: server ignored Range, restarting from 0")
            done = 0
        elif r.status_code not in (200, 206):
            r.raise_for_status()

        length = r.headers.get("Content-Length")
        total = (done + int(length)) if length else expected
        mode = "ab" if done else "wb"
        if done:
            print(f"[>>] {dest.name}: resuming at {_human(done)}")

        with open(part, mode) as fh:
            last = 0.0
            for chunk in r.iter_content(DOWNLOAD_CHUNK):
                fh.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last > 0.5:
                    pct = f" {done / total * 100:5.1f}%" if total else ""
                    print(f"\r     {dest.name}: {_human(done)}{pct}", end="", flush=True)
                    last = now
                if _stop:
                    fh.flush()
                    os.fsync(fh.fileno())
                    print(f"\n[--] {dest.name}: paused at {_human(done)}")
                    return False
    print(f"\r     {dest.name}: {_human(done)} done{' ' * 12}")

    if total and done < total:
        return False  # truncated; a later run resumes from here
    os.replace(part, dest)
    return True


def _download_with_retries(session: requests.Session, entry: dict, outdir: Path,
                            ref: str, timeout: int, retries: int) -> bool:
    for attempt in range(1, retries + 1):
        try:
            if _download_file(session, entry, outdir, timeout):
                return True
            if _stop:
                return False
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code in (401, 403, 404, 410):
                # Download links are short-lived; ask the order for a fresh one.
                print(f"[!!] {entry['name']}: link returned {code}, refreshing from order")
                fresh = next((f for f in _fetch_order(session, ref)
                              if f["name"] == entry["name"]), None)
                if fresh:
                    entry.update(fresh)
                    continue
            print(f"[!!] {entry['name']}: {exc}")
        except requests.RequestException as exc:
            print(f"[!!] {entry['name']}: {exc}")

        if attempt < retries:
            backoff = min(60, 2 ** attempt)
            print(f"[..] retry {attempt}/{retries} in {backoff}s")
            time.sleep(backoff)
    return False


def download_dataset(uuid: str, out_dir: Path, email: str | None, *,
                      area: str = "0000", projection: str = "25833", fmt: str = "FGDB",
                      poll: int = 30, timeout: int = 120, retries: int = 5) -> bool:
    """Place (or resume) a Geonorge order and download every file to out_dir.

    Safe to interrupt and re-run: state.json in out_dir tracks the order
    reference and per-file byte offsets, so re-running the identical call
    resumes rather than placing a second (costly, slow-to-regenerate) order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = _load_state(state_path)

    key = _order_key(uuid, area, projection, fmt)
    session = requests.Session()

    order = state.get("order")
    if order and order.get("key") != key:
        raise SystemExit(
            f"{state_path} holds an order for different parameters "
            f"({order.get('params')}). Use a different out_dir, or delete the state file."
        )

    if not order:
        _validate_selection(session, uuid, area, projection, fmt)
        if not email:
            raise SystemExit("email is required to place a new order.")
        ref = _place_order(session, uuid, area, projection, fmt, email)
        state["order"] = {
            "key": key,
            "referenceNumber": ref,
            "params": {"uuid": uuid, "area": area, "projection": projection, "format": fmt},
            "placed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save_state(state_path, state)
        print(f"[ok] Order placed, reference {ref}")
    else:
        ref = order["referenceNumber"]
        print(f"[ok] Resuming order {ref} (placed {order.get('placed')})")

    files = _wait_until_ready(session, ref, state, state_path, poll)
    if _stop:
        print("[--] Stopped while waiting. Re-run to continue.")
        return False

    print(f"[ok] {len(files)} file(s) ready")

    remaining = []
    for entry in files:
        if _download_with_retries(session, entry, out_dir, ref, timeout, retries):
            state["files"][entry["name"]]["complete"] = True
        else:
            remaining.append(entry["name"])
        state["files"][entry["name"]].update(entry)
        _save_state(state_path, state)
        if _stop:
            break

    if _stop:
        print("[--] Paused. Re-run the same call to resume.")
        return False
    if remaining:
        print(f"[!!] Incomplete after retries: {', '.join(remaining)}")
        return False

    print(f"[ok] All files downloaded to {out_dir.resolve()}")
    return True


# =============================================================================
# 2/3. FGDB -> CSV conversion (shared WKB decoding, EPSG:25833 -> EPSG:4326)
# =============================================================================


def _find_gdb(zip_path: Path) -> str:
    """Return the /vsizip/ path of the .gdb directory inside the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        names = {n.split("/")[0] for n in zf.namelist() if "/" in n}
    gdbs = sorted(n for n in names if n.lower().endswith(".gdb"))
    if not gdbs:
        raise SystemExit(f"[!!] no .gdb directory found inside {zip_path}")
    if len(gdbs) > 1:
        print(f"[!!] multiple .gdb found, using {gdbs[0]}", file=sys.stderr)
    abs_zip = os.path.abspath(zip_path).replace("\\", "/")
    return f"/vsizip/{abs_zip}/{gdbs[0]}"


def _decode_points(wkb) -> tuple[np.ndarray, np.ndarray]:
    """Decode an array of WKB point blobs into (x, y) float arrays.

    Missing geometries come back as NaN so the caller can drop them while
    keeping row alignment with the attribute arrays.
    """
    n = len(wkb)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)

    # Fast path: every blob is the same 21-byte little-endian 2D point, so the
    # whole chunk decodes as one structured array.
    if n and all(g is not None and len(g) == WKB_POINT_LE_2D.itemsize for g in wkb):
        rec = np.frombuffer(b"".join(bytes(g) for g in wkb), dtype=WKB_POINT_LE_2D)
        if np.all(rec["order"] == 1) and np.all(rec["type"] == 1):
            return rec["x"].copy(), rec["y"].copy()

    # Slow path: mixed endianness, Z/M points, or null geometry.
    for i, g in enumerate(wkb):
        if g is None or len(g) < 21:
            continue
        b = bytes(g)
        endian = "<" if b[0] == 1 else ">"
        gtype = int.from_bytes(b[1:5], "little" if b[0] == 1 else "big")
        if gtype % 1000 != 1:  # not a Point (2D/Z/M/ZM variants all end in 1)
            continue
        x[i], y[i] = np.frombuffer(b, dtype=f"{endian}f8", count=2, offset=5)
    return x, y


def bygning_fgdb_to_csv(zip_path: Path = BYGNING_ZIP, out_csv: Path = BYGNING_CSV, *,
                         layer: str = "bygning", id_field: str = "bygningsnummer",
                         chunk: int = 250_000, limit: int = 0, decimals: int = 6) -> None:
    """Matrikkelen - Bygningspunkt FGDB -> CSV of bygningsnummer + WGS84 lat/lon.

    The zip is never unpacked -- the .gdb is read in place through GDAL's
    /vsizip/ virtual filesystem, streamed out in `chunk`-sized reads so peak
    memory stays at a few hundred MB regardless of dataset size (4.4M rows
    nationwide).
    """
    if not os.path.exists(zip_path):
        raise SystemExit(f"[!!] no such file: {zip_path}")

    src = _find_gdb(zip_path)
    info = pyogrio.read_info(src, layer=layer)
    total = info["features"]
    if limit:
        total = min(total, limit)
    if id_field not in info["fields"]:
        raise SystemExit(
            f"[!!] field '{id_field}' not in layer '{layer}'. "
            f"Available: {', '.join(info['fields'])}"
        )
    print(f"[ok] {src}")
    print(f"[ok] layer '{layer}': {info['features']:,} features, {info['crs']}")

    transformer = Transformer.from_crs(info["crs"] or "EPSG:25833", "EPSG:4326", always_xy=True)

    tmp = str(out_csv) + ".part"
    written = skipped = offset = 0
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([id_field, "lat", "lon"])

        while offset < total and not _stop:
            n = min(chunk, total - offset)
            _, _, wkb, fields = pyogrio.raw.read(
                src, layer=layer, columns=[id_field],
                skip_features=offset, max_features=n, read_geometry=True,
            )
            ids = fields[0]
            got = len(ids)
            if got == 0:
                break

            x, y = _decode_points(wkb)
            ok = ~np.isnan(x)
            skipped += int((~ok).sum())

            lon, lat = transformer.transform(x[ok], y[ok])
            w.writerows(
                zip(ids[ok].tolist(),
                    np.round(lat, decimals).tolist(),
                    np.round(lon, decimals).tolist())
            )

            written += int(ok.sum())
            offset += got
            print(f"[..] {offset:,}/{total:,} read, {written:,} written", flush=True)

    if _stop:
        print(f"[--] interrupted; partial output left at {tmp}")
        return

    os.replace(tmp, out_csv)
    print(f"[ok] wrote {written:,} rows to {out_csv}"
          + (f" ({skipped:,} without geometry skipped)" if skipped else ""))


def _adresse_process_layer(src, layer, transformer, writer, chunk, limit):
    info = pyogrio.read_info(src, layer=layer)
    total = info["features"]
    if limit:
        total = min(total, limit)
    print(f"[ok] layer '{layer}': {info['features']:,} features, {info['crs']}")

    written = skipped = offset = 0
    while offset < total and not _stop:
        n = min(chunk, total - offset)
        meta, _, wkb, fields = pyogrio.raw.read(
            src, layer=layer, columns=ADRESSE_ID_FIELDS,
            skip_features=offset, max_features=n, read_geometry=True,
        )
        # pyogrio returns fields in the layer's own schema order, not the
        # order passed via `columns=`, so map by name rather than position.
        field_order = list(meta["fields"])

        def col(name):
            return fields[field_order.index(name)]

        kommunenummer = col("matrikkelnummeradresse_kommunenummer")
        gardsnummer = col("matrikkelnummeradresse_gardsnummer")
        bruksnummer = col("matrikkelnummeradresse_bruksnummer")
        # nullable (float64, NaN where absent) -- 0 means "not applicable"
        seksjonsnummer = np.nan_to_num(
            col("matrikkelnummeradresse_seksjonsnummer").astype("float64"), nan=0
        ).astype("int64")
        festenummer = np.nan_to_num(
            col("matrikkelnummeradresse_festenummer").astype("float64"), nan=0
        ).astype("int64")
        adresse = col("adressetekstutenadressetilleggsnavn")

        got = len(kommunenummer)
        if got == 0:
            break

        x, y = _decode_points(wkb)
        ok = ~np.isnan(x)
        skipped += int((~ok).sum())

        matrikkelnummer = [
            f"{k}-{g}/{b}/{s}/{f}"
            for k, g, b, s, f in zip(
                kommunenummer[ok], gardsnummer[ok], bruksnummer[ok],
                seksjonsnummer[ok], festenummer[ok],
            )
        ]

        lon, lat = transformer.transform(x[ok], y[ok])
        writer.writerows(
            zip(
                matrikkelnummer,
                adresse[ok].tolist(),
                np.round(lat, 6).tolist(),
                np.round(lon, 6).tolist(),
            )
        )

        written += int(ok.sum())
        offset += got
        print(f"[..] {layer}: {offset:,}/{total:,} read, {written:,} written", flush=True)

    return written, skipped


def adresse_fgdb_to_csv(zip_path: Path = ADRESSE_ZIP, out_csv: Path = ADRESSE_CSV, *,
                         chunk: int = 250_000, limit: int = 0) -> None:
    """Matrikkelen - Adresse FGDB -> CSV of matrikkelnummer + adresse + WGS84 lat/lon.

    matrikkelnummer is formatted as KOMMUNENUMMER-GARDSNUMMER/BRUKSNUMMER/
    SEKSJONSNUMMER/FESTENUMMER. Combines the 'vegadresse' (street addresses)
    and 'matrikkeladresse' (parcel-only addresses) layers.

    seksjonsnummer is essentially never populated in this dataset -- Geonorge's
    own product abstract says address-to-property linkage stops at the
    grunneiendom/festegrunn level, not down to section (0 of 2,566,736
    vegadresse rows and 3 of 35,029 matrikkeladresse rows have one, as of the
    August 2026 extract). Missing seksjonsnummer/festenummer are written as 0,
    the standard matrikkel convention for "not applicable".
    """
    if not os.path.exists(zip_path):
        raise SystemExit(f"[!!] no such file: {zip_path}")

    src = _find_gdb(zip_path)
    print(f"[ok] {src}")

    crs = pyogrio.read_info(src, layer=ADRESSE_LAYERS[0])["crs"]
    transformer = Transformer.from_crs(crs or "EPSG:25833", "EPSG:4326", always_xy=True)

    tmp = str(out_csv) + ".part"
    total_written = total_skipped = 0
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["matrikkelnummer", "adresse", "lat", "lon"])
        for layer in ADRESSE_LAYERS:
            if _stop:
                break
            written, skipped = _adresse_process_layer(src, layer, transformer, w, chunk, limit)
            total_written += written
            total_skipped += skipped

    if _stop:
        print(f"[--] interrupted; partial output left at {tmp}")
        return

    os.replace(tmp, out_csv)
    print(f"[ok] wrote {total_written:,} rows to {out_csv}"
          + (f" ({total_skipped:,} without geometry skipped)" if total_skipped else ""))


# =============================================================================
# 4. Nearest-neighbor match: address -> nearest building's bygningsnummer
# =============================================================================


def _to_utm33(lat, lon):
    t = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    x, y = t.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def match_bygningsnummer(bygning_csv: Path = BYGNING_CSV, adresse_csv: Path = ADRESSE_CSV,
                          out_csv: Path = MATCH_CSV) -> None:
    """Attach a bygningsnummer to every address by nearest-neighbor spatial match.

    matrikkel_bygning.csv (bygningsnummer + lat/lon) and matrikkel_adresse.csv
    (matrikkelnummer + adresse + lat/lon) don't share a join key -- the
    building layer has no gnr/bnr/address fields and the address layer has no
    bygningsnummer. So for each address point we find the closest building
    point nationwide (reprojected to EPSG:25833/UTM33 for accurate metre
    distances) and attach its bygningsnummer.

    match_distance_m is kept on every row so obviously-bad matches (address
    far from any building -- undeveloped lot, address point not yet built on,
    etc.) can be filtered downstream instead of being silently treated as
    correct.
    """
    t0 = time.time()

    print("Loading buildings...", flush=True)
    bygning = pd.read_csv(bygning_csv)
    bx, by = _to_utm33(bygning["lat"].to_numpy(), bygning["lon"].to_numpy())
    print(f"  {len(bygning):,} buildings ({time.time() - t0:.1f}s)", flush=True)

    print("Loading addresses...", flush=True)
    adresse = pd.read_csv(adresse_csv)
    ax, ay = _to_utm33(adresse["lat"].to_numpy(), adresse["lon"].to_numpy())
    print(f"  {len(adresse):,} addresses ({time.time() - t0:.1f}s)", flush=True)

    print("Building KD-tree over building coordinates...", flush=True)
    tree = cKDTree(np.column_stack([bx, by]))
    print(f"  done ({time.time() - t0:.1f}s)", flush=True)

    print("Querying nearest building for each address...", flush=True)
    dist, idx = tree.query(np.column_stack([ax, ay]), k=1, workers=-1)
    print(f"  done ({time.time() - t0:.1f}s)", flush=True)

    out = adresse.copy()
    out["bygningsnummer"] = bygning["bygningsnummer"].to_numpy()[idx]
    out["match_distance_m"] = np.round(dist, 1)

    print("Writing output CSV...", flush=True)
    out.to_csv(out_csv, index=False)
    print(f"Done: {out_csv} ({time.time() - t0:.1f}s total)", flush=True)

    for thresh in (5, 20, 50, 100):
        pct = (out["match_distance_m"] <= thresh).mean() * 100
        print(f"  {pct:5.1f}% of addresses matched within {thresh}m")


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage",
                     choices=["download-bygning", "download-adresse",
                              "bygning-csv", "adresse-csv", "match", "all"],
                     default="all", help="Which pipeline stage to run (default: all)")
    ap.add_argument("--email", default=None,
                     help="Required for download-bygning/download-adresse/all")
    ap.add_argument("--limit", type=int, default=0,
                     help="Cap features read per layer during CSV conversion (0 = all); "
                          "handy for a quick smoke test")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)

    stages = (
        ["download-bygning", "download-adresse", "bygning-csv", "adresse-csv", "match"]
        if args.stage == "all" else [args.stage]
    )

    for stage in stages:
        if _stop:
            break
        print(f"\n=== {stage} ===", flush=True)
        if stage == "download-bygning":
            if not download_dataset(BYGNING_UUID, BYGNING_DATA_DIR, args.email):
                return 1
        elif stage == "download-adresse":
            if not download_dataset(ADRESSE_UUID, ADRESSE_DATA_DIR, args.email):
                return 1
        elif stage == "bygning-csv":
            bygning_fgdb_to_csv(limit=args.limit)
        elif stage == "adresse-csv":
            adresse_fgdb_to_csv(limit=args.limit)
        elif stage == "match":
            match_bygningsnummer()

    if _stop:
        print("[--] Stopped. Re-run the same --stage to resume/retry.")
        return 130
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[--] Forced exit.", file=sys.stderr)
        sys.exit(130)
