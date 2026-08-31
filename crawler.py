# -*- coding: utf-8 -*-
"""爬取模块：关键词搜索 + 网页 URL 提取图片。

说明：关键词搜索默认使用 Bing 图片搜索的 HTML 解析方式（无需 API Key），
不同站点的搜索可替换为带 Key 的素材站接口（见 README）。
"""
import re
import html
import urllib.parse
import requests

BING_IMAGE_URL = "https://www.bing.com/images/search?q={q}&qft=+filterui:photo-photo&form=IRFLTR"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def search_image_urls(keyword: str, limit: int = 20, timeout: int = 15) -> list:
    """按关键词搜索并返回候选图片 URL 列表（去重后）。"""
    query = urllib.parse.quote(keyword)
    url = BING_IMAGE_URL.format(q=query)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    html_text = resp.text

    # Bing 图片结果中的真实媒体地址出现在 murl 字段中
    murls = re.findall(r'murl&quot;:&quot;(.*?)&quot;', html_text)
    if not murls:
        murls = re.findall(r'"murl":"(.*?)"', html_text)

    urls = []
    for m in murls:
        u = html.unescape(m).replace("\\u0026", "&").replace("\\/", "/")
        if u.startswith("http") and u not in urls:
            urls.append(u)
        if len(urls) >= limit:
            break
    return urls


def pixabay_search(keyword: str, limit: int = 20, api_key: str = "", timeout: int = 15) -> list:
    """Pixabay 搜索，返回可直接下载的图片 URL。"""
    params = {
        "key": api_key,
        "q": keyword,
        "per_page": limit,
        "image_type": "photo",
        "safesearch": "true",
        "orientation": "horizontal",
    }
    resp = requests.get("https://pixabay.com/api/", params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    urls = []
    for hit in data.get("hits", []):
        u = hit.get("webformatURL")
        if u and u.startswith("http") and u not in urls:
            urls.append(u)
        if len(urls) >= limit:
            break
    return urls


def pexels_search(keyword: str, limit: int = 20, api_key: str = "", timeout: int = 15) -> list:
    """Pexels 搜索，返回可直接下载的图片 URL。"""
    headers = {**HEADERS, "Authorization": api_key}
    params = {"query": keyword, "per_page": limit}
    resp = requests.get(
        "https://api.pexels.com/v1/search", params=params, headers=headers, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    urls = []
    for ph in data.get("photos", []):
        u = (ph.get("src") or {}).get("medium")
        if u and u.startswith("http") and u not in urls:
            urls.append(u)
        if len(urls) >= limit:
            break
    return urls


def image_dimensions(data: bytes):
    """读取图片尺寸 (width, height)；无法解析时返回 None。"""
    try:
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


def extract_image_urls(page_url: str, timeout: int = 15) -> list:
    """从指定网页中提取 <img> 图片地址（去重后）。"""
    resp = requests.get(page_url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    base = urllib.parse.urljoin(page_url, "/")
    srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', resp.text, re.I)
    urls = []
    for s in srcs:
        s = html.unescape(s)
        u = urllib.parse.urljoin(base, s)
        if u.startswith("http") and u not in urls:
            urls.append(u)
    return urls


def download_image(url: str, max_size_kb: int, allowed_exts: tuple, timeout: int = 20):
    """下载图片，返回 (bytes, ext) 或 (None, None)。

    过滤条件：大小不超过 max_size_kb，扩展名在 allowed_exts 内。
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if ctype and "image" not in ctype:
            return None, None, "非图片内容"
        data = resp.content
    except Exception:
        return None, None, "下载失败"

    if len(data) > max_size_kb * 1024:
        return None, None, "超过大小限制"

    ext = guess_ext(url, ctype)
    if ext not in allowed_exts:
        return None, None, "格式不支持"
    return data, ext, None


def guess_ext(url: str, content_type: str) -> str:
    """根据 URL 或 Content-Type 猜测扩展名。"""
    path = urllib.parse.urlparse(url).path.lower()
    for e in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
        if path.endswith(e):
            return e
    ct = content_type.lower()
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "gif" in ct:
        return ".gif"
    return ".jpg"
