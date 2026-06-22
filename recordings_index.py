#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

INDEX_PATH = os.environ.get("RECORDINGS_INDEX_PATH", "docs/recordings_index.json")
PER_PAGE = 100


def _headers(token):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "douyin-recorder-index",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _request_json(url, headers, timeout=30):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_meta_asset(name):
    name = (name or "").lower()
    return "_meta.json" in name or ("_meta." in name and name.endswith(".json"))


def _body_peak_viewers(body):
    try:
        data = json.loads(body or "{}")
        if isinstance(data, dict) and "pv" in data:
            return data.get("pv")
    except Exception:
        return None
    return None


def _asset_data(asset, body_peak):
    data = {
        "id": asset.get("id"),
        "name": asset.get("name", ""),
        "size": asset.get("size", 0),
        "url": asset.get("url", ""),
        "browser_download_url": asset.get("browser_download_url", ""),
        "created_at": asset.get("created_at", ""),
    }
    if _is_meta_asset(data["name"]):
        if body_peak is not None:
            data["peak_viewers"] = body_peak
    return data


def _release_data(release):
    body = release.get("body", "") or ""
    body_peak = _body_peak_viewers(body)
    return {
        "tag_name": release.get("tag_name", ""),
        "name": release.get("name", ""),
        "body": body,
        "created_at": release.get("created_at", "") or release.get("published_at", ""),
        "assets": [_asset_data(a, body_peak) for a in release.get("assets", [])],
    }


def _fetch_releases(repo, token):
    headers = _headers(token)
    releases = []
    page = 1
    while True:
        url = "https://api.github.com/repos/%s/releases?per_page=%d&page=%d" % (repo, PER_PAGE, page)
        try:
            data = _request_json(url, headers, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 422:
                print("GitHub Releases 只允许读取前 1000 条，已在第 %d 页停止" % page)
                break
            raise
        if not data:
            break
        releases.extend([_release_data(r) for r in data if r.get("assets")])
        print("读取 Releases 第 %d 页：%d 条" % (page, len(data)))
        page += 1
    return releases


def _build_index(releases):
    asset_count = sum(len(r.get("assets", [])) for r in releases)
    return {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_count": len(releases),
        "asset_count": asset_count,
        "releases": releases,
    }


def _write_index(index, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


def rebuild():
    repo = os.environ.get("GH_REPO", "")
    token = os.environ.get("GH_TOKEN", "")
    if not repo:
        print("缺少 GH_REPO")
        return 1
    releases = _fetch_releases(repo, token)
    index = _build_index(releases)
    _write_index(index, INDEX_PATH)
    print("已生成 %s：%d 个 Release，%d 个文件" % (INDEX_PATH, index["release_count"], index["asset_count"]))
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rebuild"
    if cmd == "rebuild":
        return rebuild()
    print("用法：python recordings_index.py rebuild")
    return 1


if __name__ == "__main__":
    sys.exit(main())
