"""Read-only Zotero/Obsidian audit; writes only project TSV/Markdown reports."""

import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(r"D:\Desktop\01_项目与产品\Rove如晤_跨时空交互")
VAULT = Path(r"D:\Desktop\01_项目与产品\3DS软件社团\课堂演示库\Literature")
INV = ROOT / "03_文献笔记" / "Rove_全部论文_来源与阅读状态审计.md"
OUT = ROOT / "03_文献笔记" / "Rove_跨端同步与阅读审计.tsv"


def norm(value):
    return re.sub(r"[^a-z0-9]", "", (value or "").casefold())


def frontmatter(text):
    if not text.startswith("---\n"):
        return {}
    block = text.split("---\n", 2)[1]
    return dict(re.findall(r"(?m)^([A-Za-z_]+):\s*(.*)$", block))


def main():
    selected = json.loads((OUT.parent / "Rove_Zotero45_实时快照.json").read_text(encoding="utf-8-sig"))
    whole_library = json.loads((OUT.parent / "Zotero_整库218条_题名快照.json").read_text(encoding="utf-8-sig"))
    selected_by_key = {x["key"]: x for x in selected}
    items = {x["key"]: selected_by_key.get(x["key"], x) for x in whole_library}
    # shortcut: immutable SQLite is a point-in-time check while Zotero holds a lock; rerun after sync for final imports.
    db = sqlite3.connect(Path(r"C:\Users\30842\Zotero\zotero.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)
    doi_pairs = db.execute("""
        SELECT i.key,v.value FROM items i JOIN itemData d ON i.itemID=d.itemID
        JOIN fields f ON f.fieldID=d.fieldID JOIN itemDataValues v ON v.valueID=d.valueID
        WHERE f.fieldName='DOI'
    """).fetchall()
    for key, doi in doi_pairs:
        if key in items and not items[key].get("DOI"):
            items[key]["DOI"] = doi
    db.close()
    by_doi, by_title = {}, {}
    for key, p in items.items():
        if p.get("DOI"):
            by_doi.setdefault(norm(p["DOI"]), []).append(key)
        by_title.setdefault(norm(p.get("title")), []).append(key)

    cards = {}
    for path in VAULT.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            continue
        fm = frontmatter(text)
        key = fm.get("zoteroKey", "").strip("'\" ")
        doi = fm.get("doi", "").strip("'\" ")
        if key or doi:
            record = {"path": str(path), "fm": fm, "text": text}
            if key:
                cards.setdefault(("key", key), []).append(record)
            if doi:
                cards.setdefault(("doi", norm(doi)), []).append(record)

    selected_audit = []
    for p in selected:
        attachments = [a for a in p.get("attachments", []) if a.get("path") and Path(a["path"]).is_file()]
        hashes = []
        for a in attachments:
            h = hashlib.sha256()
            with Path(a["path"]).open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            hashes.append(h.hexdigest())
        note_paths = [c["path"] for c in cards.get(("key", p["key"]), [])]
        selected_audit.append({"Zotero父条目Key": p["key"], "英文题名": p["title"], "DOI": p.get("DOI", ""), "PDF附件Key": ";".join(a["key"] for a in attachments), "PDF路径": ";".join(a["path"] for a in attachments), "PDF_SHA256": ";".join(hashes), "Zotero Note数": p.get("notes", 0), "Zotero批注数": 0, "Obsidian卡片路径": ";".join(note_paths), "用户已读状态": "未获用户已读确认"})
    with (OUT.parent / "Rove_Zotero45_附件批注与个人阅读.tsv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_audit[0]), dialect="excel-tab")
        writer.writeheader()
        writer.writerows(selected_audit)

    result = []
    for line in INV.read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"^\|\s*([^|]+?)\s*\|\s*\[([^]]+)\]\((https?://[^)]+)\)", line)
        if not m or m.group(1) == "编号":
            continue
        sid, title, url = m.groups()
        doi_match = re.search(r"(?:doi\.org/|doi:)(10\.[^\s)]+)", url, flags=re.I)
        doi = doi_match.group(1) if doi_match else ""
        ids = by_doi.get(norm(doi), []) if doi else []
        match_type = "DOI" if ids else ""
        if not ids:
            ids = by_title.get(norm(title), [])
            match_type = "title" if ids else ""
        # Preserve ambiguity instead of selecting an arbitrary duplicate.
        zotero_keys = ids
        pdfs = [a["path"] for i in ids for a in items[i].get("attachments", []) if a.get("path") and Path(a["path"]).is_file()]
        card_hits = []
        for key in zotero_keys:
            card_hits += cards.get(("key", key), [])
        if not card_hits and doi:
            card_hits = cards.get(("doi", norm(doi)), [])
        card_hits = list({c["path"]: c for c in card_hits}.values())
        statuses = {c["fm"].get("status", "").strip("'\" ").casefold() for c in card_hits}
        user_status = "未读（卡片明确标记）" if "unread" in statuses else "未获用户已读确认"
        ann = []
        comments = []
        result.append({
            "编号": sid, "英文题名": title, "论文链接": url, "DOI（来源页）": doi,
            "Zotero父条目Key": ";".join(zotero_keys), "Zotero匹配依据": match_type,
            "Zotero所属集合": "Rove｜跨空间触觉交互 / 00｜精选总览（45篇）" if any(i in selected_by_key for i in ids) else "整库条目；集合未核对",
            "Zotero本地PDF数": len(pdfs), "Zotero本地PDF路径": ";".join(pdfs),
            "Obsidian卡片数": len(card_hits), "Obsidian卡片路径": ";".join(c["path"] for c in card_hits),
            "用户已读状态": user_status, "Zotero高亮/批注对象数": len(ann),
            "Zotero有文本评论数": len(comments), "批注来源": "Zotero connector：Rove精选45条逐条查询均为0；其他整库条目尚未核对",
            "证据置信度": "高（DOI匹配 Zotero 45）" if match_type == "DOI" and len(ids) == 1 else "中（题名匹配）" if match_type == "title" and len(ids) == 1 else "冲突（多条）" if len(ids) > 1 else "低（仅来源清单；未匹配本地）",
        })
    assert len(result) == 272, len(result)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]), dialect="excel-tab")
        writer.writeheader()
        writer.writerows(result)
    with (OUT.parent / "Rove_P100_题名DOI本地PDF与集合.csv").open(encoding="utf-8-sig", newline="") as f:
        p100 = list(csv.DictReader(f))
    dedupe = []
    for p in p100:
        if not p["本地PDF"]:
            continue
        doi_ids = by_doi.get(norm(p["DOI_来自原站URL或既有清单"]), []) if p["DOI_来自原站URL或既有清单"] else []
        title_ids = by_title.get(norm(p["英文标题"]), [])
        dedupe.append({"P_ID": p["P_ID"], "英文标题": p["英文标题"], "DOI": p["DOI_来自原站URL或既有清单"], "本地PDF": p["本地PDF"], "本地PDF_SHA256": p["本地PDF_SHA256"], "Zotero_DOI精确匹配Key": ";".join(doi_ids), "Zotero_题名精确匹配Key": ";".join(title_ids), "待导入判定": "已有条目—只附PDF/归类" if doi_ids or title_ids else "整库快照未命中—可用skipExisting再核"})
    with (OUT.parent / "Rove_P100_本地PDF_Zotero整库去重.tsv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dedupe[0]), dialect="excel-tab")
        writer.writeheader()
        writer.writerows(dedupe)
    print("p100_local", len(dedupe), "doi_match", sum(bool(x["Zotero_DOI精确匹配Key"]) for x in dedupe), "title_match", sum(bool(x["Zotero_题名精确匹配Key"]) for x in dedupe))
    print("rows", len(result), "zotero_match", sum(bool(x["Zotero父条目Key"]) for x in result), "obsidian_match", sum(bool(x["Obsidian卡片路径"]) for x in result), "pdf_match", sum(bool(x["Zotero本地PDF路径"]) for x in result), "annotations", sum(int(x["Zotero高亮/批注对象数"]) for x in result))


if __name__ == "__main__":
    main()
