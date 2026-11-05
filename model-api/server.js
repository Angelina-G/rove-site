/**
 * Rove Model Studio 本地服务
 * ---------------------------------------------------------------
 * 纯 Node.js 原生 http，无第三方依赖。职责：
 *   1. 提供工作台页面（../rove-model-studio.html）；
 *   2. 扫描 model-generation/output/jobs/* 的真实产物，输出分阶段数据；
 *   3. 扫描 model-generation/test-images/processed-* 的抠图 QC 数据（v2 优先）；
 *   4. 安全地提供产物静态文件（防目录穿越、支持 Range 与下载）；
 *   5. 保留原有的 TripoSR 创建任务 / 查询 / 下载 / 清理接口。
 *
 * 所有接口都直接读取磁盘：磁盘上没有的产物一律返回“缺失”，绝不编造。
 */
import http from 'node:http';
import { randomUUID } from 'node:crypto';
import {
  readFileSync, createReadStream, existsSync, readdirSync, statSync, openSync, readSync, closeSync, cpSync,
} from 'node:fs';
import { writeFile, unlink, mkdir, stat, rm, rename } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

/* ------------------------------------------------------------------ */
/* 环境与路径                                                          */
/* ------------------------------------------------------------------ */

const HERE = path.dirname(fileURLToPath(import.meta.url));
try {
  for (const line of readFileSync(path.join(HERE, '.env'), 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.*)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^['"]|['"]$/g, '');
  }
} catch { /* .env 可选 */ }

const PORT = Number(process.env.PORT || 8790);
const HOST = process.env.HOST || '127.0.0.1';
const SITE_ROOT = path.resolve(HERE, '..');
const GEN_ROOT = path.join(SITE_ROOT, 'model-generation');
const JOBS_DIR = path.join(GEN_ROOT, 'output', 'jobs');
const TEST_IMAGES_DIR = path.join(GEN_ROOT, 'test-images');
const PAGE_FILE = path.join(SITE_ROOT, 'rove-model-studio.html');
const TRIPOSR_SOURCE = process.env.TRIPOSR_SOURCE || '';
const TRIPOSR_PYTHON = process.env.TRIPOSR_PYTHON || '';
const GEN_FROM_SOURCE = TRIPOSR_SOURCE ? path.resolve(TRIPOSR_SOURCE, '..', '..') : GEN_ROOT;

const provider = process.env.MODEL_PROVIDER || (process.env.MESHY_API_KEY ? 'meshy' : 'local-triposr');
const providerLabel = provider === 'local-triposr' ? 'TripoSR（本地）' : 'Meshy';
const allowedOrigin = process.env.ALLOWED_ORIGIN || `http://localhost:${PORT}`;
const tasks = new Map(); // 仅保存在进程内存中；磁盘历史任务从产物目录恢复

/* ------------------------------------------------------------------ */
/* 通用工具                                                            */
/* ------------------------------------------------------------------ */

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
  '.glb': 'model/gltf-binary',
  '.gltf': 'model/gltf+json',
  '.obj': 'text/plain; charset=utf-8',
  '.ply': 'application/octet-stream',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.txt': 'text/plain; charset=utf-8',
  '.log': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
};
const IMG_RE = /\.(png|jpe?g|webp|gif|bmp)$/i;
const MESH_RE = /\.(glb|gltf|obj|ply)$/i;
const JSON_RE = /\.json$/i;

const json = (res, status, body) => {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
  });
  res.end(payload);
};

function isLocalOrigin(origin) {
  if (origin === 'null') return true;
  try {
    const u = new URL(origin);
    return ['localhost', '127.0.0.1', '[::1]', '::1'].includes(u.hostname) && ['http:', 'https:'].includes(u.protocol);
  } catch { return false; }
}

const relPath = (abs) => path.relative(SITE_ROOT, abs).split(path.sep).join('/');
const toPosix = (p) => String(p).split(path.sep).join('/');
const fileUrl = (prefix, rel) => `${prefix}/${rel.split('/').map(encodeURIComponent).join('/')}`;
const jobFileUrl = (id, rel) => fileUrl(`/api/jobs/${encodeURIComponent(id)}/files`, rel);
const qcFileUrl = (version, rel) => fileUrl(`/api/qc/files/${encodeURIComponent(version)}`, rel);
const qcSourceUrl = (name) => `/api/qc/source/${encodeURIComponent(name)}`;

function resolveInside(base, rel) {
  if (typeof rel !== 'string' || !rel || rel.includes('\0')) return null;
  const normalized = rel.replace(/\\/g, '/').replace(/^\/+/, '');
  if (normalized.split('/').some((seg) => seg === '..' || seg === '.' || seg === '')) return null;
  const baseAbs = path.resolve(base);
  const abs = path.resolve(baseAbs, normalized);
  if (abs !== baseAbs && !abs.startsWith(baseAbs + path.sep)) return null;
  return abs;
}

function walkFiles(dir, base = dir, depth = 0, out = []) {
  let entries;
  try { entries = readdirSync(dir, { withFileTypes: true }); } catch { return out; }
  if (depth > 8) return out;
  for (const entry of entries) {
    if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkFiles(abs, base, depth + 1, out);
    } else if (entry.isFile()) {
      try {
        const st = statSync(abs);
        out.push({ rel: toPosix(path.relative(base, abs)), abs, size: st.size, mtimeMs: st.mtimeMs });
      } catch { /* 忽略单个坏文件 */ }
    }
  }
  return out;
}

function readJsonSync(file) {
  try {
    const raw = readFileSync(file, 'utf8');
    return { json: JSON.parse(raw), error: null, bytes: Buffer.byteLength(raw) };
  } catch (e) {
    return { json: null, error: e.message || 'JSON 解析失败', bytes: null };
  }
}

function listTopFiles(dir) {
  try {
    return readdirSync(dir, { withFileTypes: true })
      .filter((e) => e.isFile())
      .map((e) => {
        try {
          const abs = path.join(dir, e.name);
          const st = statSync(abs);
          return { name: e.name, rel: e.name, abs, size: st.size, mtimeMs: st.mtimeMs };
        } catch { return null; }
      })
      .filter(Boolean);
  } catch { return []; }
}

/* ------------------------------------------------------------------ */
/* 图像元数据 / PNG alpha 实测（QC 指标）                               */
/* ------------------------------------------------------------------ */

const IMAGE_CACHE = new Map();
const ALPHA_CACHE = new Map();

function cacheGet(map, key, sig) {
  const hit = map.get(key);
  if (hit && hit.sig === sig) return hit.value;
  return null;
}
function cacheSet(map, key, sig, value) {
  if (map.size > 300) map.clear();
  map.set(key, { sig, value });
}

const PNG_CHANNELS = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };
const JPEG_SOF = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]);

function jpegSize(buf) {
  let off = 2;
  while (off + 9 < buf.length) {
    if (buf[off] !== 0xff) { off += 1; continue; }
    const marker = buf[off + 1];
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) { off += 2; continue; }
    if (marker === 0xd9 || marker === 0xda) break;
    const len = buf.readUInt16BE(off + 2);
    if (!len) break;
    if (JPEG_SOF.has(marker) && off + 9 < buf.length) {
      return { height: buf.readUInt16BE(off + 5), width: buf.readUInt16BE(off + 7) };
    }
    off += 2 + len;
  }
  return null;
}

function imageMeta(file) {
  let st;
  try { st = statSync(file); } catch { return { exists: false }; }
  const sig = `${st.size}:${st.mtimeMs}`;
  const cached = cacheGet(IMAGE_CACHE, file, sig);
  if (cached) return cached;
  const info = { exists: true, width: null, height: null, format: null, hasAlpha: false, bytes: st.size };
  let head = null;
  try {
    const fd = openSync(file, 'r');
    const buf = Buffer.alloc(64);
    const n = readSync(fd, buf, 0, 64, 0);
    closeSync(fd);
    head = buf.subarray(0, n);
  } catch { /* ignore */ }
  if (head && head.length >= 24 && head.readUInt32BE(0) === 0x89504e47) {
    info.format = 'png';
    info.width = head.readUInt32BE(16);
    info.height = head.readUInt32BE(20);
    const colorType = head[25];
    info.hasAlpha = colorType === 4 || colorType === 6 || colorType === 0; // 灰度蒙版把灰度当 alpha
    info.pngBitDepth = head[24];
    info.pngColorType = colorType;
  } else if (head && head.length >= 3 && head[0] === 0xff && head[1] === 0xd8) {
    info.format = 'jpg';
    try {
      const size = jpegSize(readFileSync(file));
      if (size) { info.width = size.width; info.height = size.height; }
    } catch { /* ignore */ }
  } else if (head && head.length >= 30 && head.toString('ascii', 0, 4) === 'RIFF' && head.toString('ascii', 8, 12) === 'WEBP') {
    info.format = 'webp';
  }
  cacheSet(IMAGE_CACHE, file, sig, info);
  return info;
}

function pngAlphaStats(file) {
  let st;
  try { st = statSync(file); } catch { return null; }
  const sig = `${st.size}:${st.mtimeMs}`;
  const cached = cacheGet(ALPHA_CACHE, file, sig);
  if (cached) return cached;
  let result = null;
  try {
    const buf = readFileSync(file);
    if (buf.length < 33 || buf.readUInt32BE(0) !== 0x89504e47) return null;
    let off = 8;
    let width = 0; let height = 0; let bitDepth = 8; let colorType = 6; let interlace = 0;
    const idat = [];
    while (off + 12 <= buf.length) {
      const len = buf.readUInt32BE(off);
      if (len > buf.length) break;
      const type = buf.toString('ascii', off + 4, off + 8);
      const data = buf.subarray(off + 8, off + 8 + len);
      if (type === 'IHDR') {
        width = data.readUInt32BE(0);
        height = data.readUInt32BE(4);
        bitDepth = data[8];
        colorType = data[9];
        interlace = data[12];
      } else if (type === 'IDAT') idat.push(data);
      else if (type === 'IEND') break;
      off += 12 + len;
    }
    if (!width || !height) return null;
    if (bitDepth !== 8 || interlace !== 0 || colorType === 3) {
      // 调色板 / 隔行 / 非 8 位 PNG 不做 alpha 实测，交给报告值。
      result = { supported: false, width, height };
      cacheSet(ALPHA_CACHE, file, sig, result);
      return result;
    }
    const channels = PNG_CHANNELS[colorType] || 4;
    let raw;
    try { raw = zlib.inflateSync(Buffer.concat(idat)); } catch { return null; }
    const stride = width * channels;
    if (raw.length < (stride + 1) * height) return null;
    const out = Buffer.alloc(stride * height);
    let prev = Buffer.alloc(stride);
    for (let y = 0; y < height; y += 1) {
      const filter = raw[y * (stride + 1)];
      const rowStart = y * (stride + 1) + 1;
      const cur = Buffer.alloc(stride);
      for (let x = 0; x < stride; x += 1) {
        const a = x >= channels ? cur[x - channels] : 0;
        const b = prev[x];
        const c = x >= channels ? prev[x - channels] : 0;
        let v = raw[rowStart + x];
        if (filter === 1) v = (v + a) & 0xff;
        else if (filter === 2) v = (v + b) & 0xff;
        else if (filter === 3) v = (v + ((a + b) >> 1)) & 0xff;
        else if (filter === 4) {
          const p = a + b - c;
          const pa = Math.abs(p - a); const pb = Math.abs(p - b); const pc = Math.abs(p - c);
          v = (v + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c)) & 0xff;
        } else if (filter !== 0) return null;
        cur[x] = v;
      }
      cur.copy(out, y * stride);
      prev = cur;
    }
    const alphaAt = (ch) => {
      if (colorType === 6) return out[ch + 3];
      if (colorType === 4) return out[ch + 1];
      if (colorType === 0) return out[ch];
      return 255;
    };
    const band = Math.max(2, Math.round(Math.min(width, height) * 0.01));
    let fg = 0; let soft = 0; let bandFg = 0; let bandPx = 0; let strictFg = 0;
    let minX = width; let minY = height; let maxX = -1; let maxY = -1;
    const total = width * height;
    for (let y = 0; y < height; y += 1) {
      const inYB = y < band || y >= height - band;
      for (let x = 0; x < width; x += 1) {
        const ch = y * stride + x * channels;
        const a = alphaAt(ch);
        const isFg = a >= 128;
        if (isFg) {
          fg += 1;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
        if (a > 0 && a < 255) soft += 1;
        const inBand = inYB || x < band || x >= width - band;
        if (inBand) { bandPx += 1; if (isFg) bandFg += 1; }
        if (x === 0 || y === 0 || x === width - 1 || y === height - 1) {
          if (isFg) strictFg += 1;
        }
      }
    }
    result = {
      supported: true,
      width,
      height,
      colorType,
      totalPixels: total,
      foregroundRatio: fg / total,
      softAlphaRatio: soft / total,
      edgeLeakRatio: bandPx ? bandFg / bandPx : null,
      edgeBandPx: band,
      touchesFrameEdge: strictFg > 0,
      bbox: maxX >= 0 ? [minX, minY, maxX - minX + 1, maxY - minY + 1] : null,
    };
  } catch {
    result = null;
  }
  cacheSet(ALPHA_CACHE, file, sig, result);
  return result;
}

const round4 = (n) => (typeof n === 'number' && Number.isFinite(n) ? Math.round(n * 10000) / 10000 : null);

/* ------------------------------------------------------------------ */
/* GLB 实测：顶点 / 面 / watertight / 骨架 / 形态键                      */
/* ------------------------------------------------------------------ */

const GLB_CACHE = new Map();
const CT_BYTES = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };

function readGlbIndices(gltf, bin, accessorIndex) {
  const acc = gltf.accessors?.[accessorIndex];
  if (!acc) throw new Error('索引 accessor 不存在');
  const bv = gltf.bufferViews?.[acc.bufferView];
  if (!bv || !bin) throw new Error('索引 bufferView 不存在');
  const count = acc.count >>> 0;
  const ctype = acc.componentType;
  const bytes = CT_BYTES[ctype];
  if (!bytes || (ctype !== 5121 && ctype !== 5123 && ctype !== 5125)) throw new Error('不支持的索引类型');
  const base = (bv.byteOffset || 0) + (acc.byteOffset || 0);
  const stride = bv.byteStride || bytes;
  const out = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) {
    const p = base + i * stride;
    if (p + bytes > bin.length) throw new Error('索引越界');
    out[i] = ctype === 5121 ? bin.readUInt8(p) : ctype === 5123 ? bin.readUInt16LE(p) : bin.readUInt32LE(p);
  }
  return out;
}

function analyzeGlb(file) {
  let st;
  try { st = statSync(file); } catch { return { exists: false, ok: false, error: '文件不存在' }; }
  const sig = `${st.size}:${st.mtimeMs}`;
  const cached = cacheGet(GLB_CACHE, file, sig);
  if (cached) return cached;
  const result = { exists: true, ok: false, file, sizeBytes: st.size, error: null };
  try {
    const buf = readFileSync(file);
    if (buf.length < 20 || buf.toString('ascii', 0, 4) !== 'glTF') throw new Error('不是有效的 GLB 文件');
    const version = buf.readUInt32LE(4);
    const totalLength = buf.readUInt32LE(8);
    let off = 12;
    let gltf = null;
    let bin = null;
    while (off + 8 <= buf.length && off + 8 <= (totalLength || buf.length)) {
      const len = buf.readUInt32LE(off);
      const type = buf.toString('ascii', off + 4, off + 8);
      const start = off + 8;
      if (start + len > buf.length) break;
      if (type === 'JSON') {
        const text = buf.toString('utf8', start, start + len).replace(/\u0000+$/g, '');
        gltf = JSON.parse(text);
      } else if (type === 'BIN\0' || type === 'BIN ') {
        bin = buf.subarray(start, start + len);
      }
      off = start + len;
    }
    if (!gltf) throw new Error('GLB 缺少 JSON 数据块');
    result.version = version;
    result.generator = gltf.asset?.generator || null;
    result.meshes = Array.isArray(gltf.meshes) ? gltf.meshes.length : 0;
    result.nodes = Array.isArray(gltf.nodes) ? gltf.nodes.length : 0;
    result.materials = Array.isArray(gltf.materials) ? gltf.materials.length : 0;
    result.textures = Array.isArray(gltf.textures) ? gltf.textures.length : 0;
    result.images = Array.isArray(gltf.images) ? gltf.images.length : 0;
    result.skins = Array.isArray(gltf.skins) ? gltf.skins.length : 0;
    result.animations = Array.isArray(gltf.animations) ? gltf.animations.length : 0;
    result.hasSkeleton = result.skins > 0 || (Array.isArray(gltf.nodes) && gltf.nodes.some((n) => n.skin != null));
    let morphTargets = 0;
    let vertices = 0;
    let faces = 0;
    let primitives = 0;
    let boundaryEdges = 0;
    let nonManifoldEdges = 0;
    let components = 0;
    let vertexColors = false;
    let bbox = null;
    for (const mesh of gltf.meshes || []) {
      for (const prim of mesh.primitives || []) {
        primitives += 1;
        const posIndex = prim.attributes?.POSITION;
        const posAcc = posIndex != null ? gltf.accessors?.[posIndex] : null;
        const vCount = posAcc?.count || 0;
        vertices += vCount;
        if (prim.attributes?.COLOR_0 != null) vertexColors = true;
        if (Array.isArray(prim.targets)) morphTargets += prim.targets.length;
        if (posAcc?.min && posAcc?.max && posAcc.min.length === 3 && posAcc.max.length === 3) {
          if (!bbox) bbox = { min: posAcc.min.slice(), max: posAcc.max.slice() };
          else {
            for (let k = 0; k < 3; k += 1) {
              bbox.min[k] = Math.min(bbox.min[k], posAcc.min[k]);
              bbox.max[k] = Math.max(bbox.max[k], posAcc.max[k]);
            }
          }
        }
        const mode = prim.mode == null ? 4 : prim.mode;
        if (mode !== 4) continue; // 只统计三角面网格的拓扑
        const indices = prim.indices != null ? readGlbIndices(gltf, bin, prim.indices) : null;
        const triCount = indices ? Math.floor(indices.length / 3) : Math.floor(vCount / 3);
        faces += triCount;
        if (triCount === 0) continue;
        const idx = indices || Uint32Array.from({ length: Math.floor(vCount / 3) * 3 }, (_, i) => i);
        const edgeMap = new Map();
        const parent = new Map();
        const find = (x) => {
          let r = x;
          while (parent.get(r) !== r) r = parent.get(r);
          while (parent.get(x) !== r) { const next = parent.get(x); parent.set(x, r); x = next; }
          return r;
        };
        const union = (a, b) => {
          const ra = find(a); const rb = find(b);
          if (ra !== rb) parent.set(ra, rb);
        };
        for (let t = 0; t < triCount; t += 1) {
          const a = idx[t * 3]; const b = idx[t * 3 + 1]; const c = idx[t * 3 + 2];
          if (!parent.has(a)) parent.set(a, a);
          if (!parent.has(b)) parent.set(b, b);
          if (!parent.has(c)) parent.set(c, c);
          union(a, b); union(b, c);
          if (a === b || b === c || a === c) continue;
          const edges = [[a, b], [b, c], [c, a]];
          for (const [u, v] of edges) {
            const lo = u < v ? u : v;
            const hi = u < v ? v : u;
            const key = lo * 4294967296 + hi;
            edgeMap.set(key, (edgeMap.get(key) || 0) + 1);
          }
        }
        for (const count of edgeMap.values()) {
          if (count === 1) boundaryEdges += 1;
          else if (count > 2) nonManifoldEdges += 1;
        }
        const roots = new Set();
        for (const v of parent.keys()) roots.add(find(v));
        components += roots.size;
      }
    }
    result.primitives = primitives;
    result.vertices = vertices;
    result.faces = faces;
    result.triangles = faces;
    result.boundaryEdges = boundaryEdges;
    result.nonManifoldEdges = nonManifoldEdges;
    result.components = components;
    result.watertight = faces > 0 && boundaryEdges === 0 && nonManifoldEdges === 0;
    result.morphTargets = morphTargets;
    result.hasMorphTargets = morphTargets > 0;
    result.vertexColors = vertexColors;
    result.hasAnimations = result.animations > 0;
    result.bbox = bbox;
    result.dimensions = bbox ? [bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1], bbox.max[2] - bbox.min[2]].map((n) => Math.round(n * 1000) / 1000) : null;
    result.ok = true;
  } catch (e) {
    result.error = e.message || 'GLB 解析失败';
  }
  cacheSet(GLB_CACHE, file, sig, result);
  return result;
}

/* ------------------------------------------------------------------ */
/* 报告字段容错提取                                                     */
/* ------------------------------------------------------------------ */

const REPORT_NAMES = ['generation-report.json', 'report.json', 'job-report.json', 'job.json', 'quality-report.json', 'manifest.json'];

function flattenFields(value, out = [], prefix = '', depth = 0) {
  if (out.length > 400 || depth > 5) return out;
  if (value === null || typeof value !== 'object') {
    if (prefix) out.push({ path: prefix, value });
    return out;
  }
  if (Array.isArray(value)) {
    if (!value.length) return out;
    if (value.every((v) => v === null || typeof v !== 'object')) {
      out.push({ path: prefix, value: value.slice(0, 8) });
    } else {
      value.slice(0, 3).forEach((v, i) => flattenFields(v, out, `${prefix}[${i}]`, depth + 1));
    }
    return out;
  }
  for (const [k, v] of Object.entries(value)) flattenFields(v, out, prefix ? `${prefix}.${k}` : k, depth + 1);
  return out;
}

function pickField(fields, pattern) {
  for (const field of fields) {
    if (!pattern.test(field.path)) continue;
    const v = field.value;
    if (typeof v === 'number' || typeof v === 'boolean' || (typeof v === 'string' && v.length < 120)) return field;
  }
  return null;
}

function toMemoryMb(field) {
  if (!field) return null;
  const v = Number(field.value);
  if (!Number.isFinite(v)) return null;
  const p = field.path.toLowerCase();
  if (/gb/.test(p)) return Math.round(v * 1024 * 10) / 10;
  if (/bytes|byte|_b$/.test(p)) return Math.round(v / 1048576 * 10) / 10;
  if (/mb|mib/.test(p)) return Math.round(v * 10) / 10;
  return v > 100000 ? Math.round(v / 1048576 * 10) / 10 : Math.round(v * 10) / 10;
}

function findDate(fields, keys) {
  for (const field of fields) {
    const leaf = (field.path.split('.').pop() || '').toLowerCase();
    if (!keys.some((k) => leaf === k || leaf.endsWith(k))) continue;
    const t = typeof field.value === 'number'
      ? (field.value > 1e12 ? field.value : field.value > 1e9 ? field.value * 1000 : null)
      : Date.parse(String(field.value));
    if (Number.isFinite(t) && t > 0) return t;
  }
  return null;
}

const PARAM_RE = /(octree|resolution|mc_|chunk|seed|step|guidance|threshold|scale|num_inference|num_view|views|device|precision|low_vram|quality|model_name|variant|weights)/i;
const DURATION_RE = /(elapsed|duration|seconds|runtime|took|time_cost|total_time)/i;
const VRAM_RE = /(vram|peak.*mem|max_memory|memory_allocated|gpu.*mem|mem_peak)/i;
const VERTEX_RE = /(num_vertices|vertex_count|vertices|verts)/i;
const FACE_RE = /(num_faces|face_count|faces|triangles)/i;
const WATERTIGHT_RE = /watertight|manifold/i;

function normalizeStatus(value) {
  if (value == null) return null;
  const s = String(value).toUpperCase();
  if (/(SUCCEED|SUCCESS|COMPLETE|DONE|FINISH|READY)/.test(s)) return 'SUCCEEDED';
  if (/(FAIL|ERROR|ABORT)/.test(s)) return 'FAILED';
  if (/(RUN|PROGRESS|PENDING|QUEUE|START)/.test(s)) return 'RUNNING';
  return null;
}

function buildReportInfo(reportEntry) {
  if (!reportEntry) return null;
  const json = reportEntry.json;
  if (json == null) {
    return { name: reportEntry.name, url: reportEntry.url, downloadUrl: `${reportEntry.url}?download=1`, parseError: reportEntry.error, fields: [], summary: {} };
  }
  const fields = flattenFields(json);
  // 优先用整轮耗时，避免抓到 model_load 这种阶段耗时
  const totalDuration = fields.find(
    (f) => /(elapsed_total|elapsed_seconds|total_seconds|total_time)/i.test(f.path) && Number.isFinite(Number(f.value)),
  );
  const durationField = totalDuration || pickField(fields, DURATION_RE);
  const vramField = pickField(fields, VRAM_RE);
  const vertexField = pickField(fields, VERTEX_RE);
  const faceField = pickField(fields, FACE_RE);
  const watertightField = pickField(fields, WATERTIGHT_RE);
  const statusField = pickField(fields, /^(status|state|result|overall_status)$/i) || pickField(fields, /status$/i);
  const modelField = pickField(fields, /^(model|model_name|engine|backend|pipeline|variant)$/i) || pickField(fields, /model_name/i);
  const params = fields
    .filter((f) => PARAM_RE.test(f.path) && !DURATION_RE.test(f.path) && !VRAM_RE.test(f.path) && !VERTEX_RE.test(f.path) && !FACE_RE.test(f.path))
    .filter((f) => typeof f.value === 'number' || typeof f.value === 'boolean' || (typeof f.value === 'string' && f.value.length < 80))
    .slice(0, 10)
    .map((f) => ({ label: f.path.split('.').pop(), path: f.path, value: f.value }));
  const summary = {
    status: statusField ? normalizeStatus(statusField.value) || String(statusField.value) : null,
    model: modelField ? String(modelField.value) : null,
    durationSeconds: durationField && Number.isFinite(Number(durationField.value)) ? Number(durationField.value) : null,
    durationPath: durationField?.path || null,
    vramPeakMb: toMemoryMb(vramField),
    vramPath: vramField?.path || null,
    vertexCount: vertexField && Number.isFinite(Number(vertexField.value)) ? Number(vertexField.value) : null,
    faceCount: faceField && Number.isFinite(Number(faceField.value)) ? Number(faceField.value) : null,
    watertightReported: watertightField && typeof watertightField.value === 'boolean' ? watertightField.value : null,
    params,
  };
  return {
    name: reportEntry.name,
    url: reportEntry.url,
    downloadUrl: `${reportEntry.url}?download=1`,
    parseError: null,
    fields,
    summary,
    json,
    bytes: reportEntry.bytes,
  };
}

/* ------------------------------------------------------------------ */
/* 任务扫描：分类产物                                                   */
/* ------------------------------------------------------------------ */

function angleFromName(name) {
  const n = String(name || '').toLowerCase();
  if (/back|rear|背面|后面|后侧/.test(n)) return 'back';
  if (/left|左侧|左面/.test(n)) return 'left';
  if (/right|右侧|右面/.test(n)) return 'right';
  if (/front|正面|前面|前侧/.test(n)) return 'front';
  if (/side|profile|侧面|侧身/.test(n)) return 'side';
  if (/face|面部|脸/.test(n)) return 'face-detail';
  if (/top|俯视/.test(n)) return 'top';
  return 'unknown';
}

const ANGLE_LABELS = {
  front: '正面', left: '左侧', right: '右侧', back: '背面', side: '侧面', 'face-detail': '面部细节', top: '俯视', unknown: '角度未标注',
};
const VIEW_ORDER = ['front', 'left', 'right', 'back'];

function classifyJobFiles(id, files) {
  const groups = {
    capture: [], segHuman: [], segPortrait: [], segFused: [], segMask: [], segCutout: [],
    input: [], render: [], glb: [], report: [], log: [], other: [], all: files,
  };
  for (const f of files) {
    const relLower = f.rel.toLowerCase();
    const base = path.basename(relLower);
    const dirPart = relLower.includes('/') ? relLower.slice(0, relLower.lastIndexOf('/') + 1) : '';
    if (MESH_RE.test(base)) { groups.glb.push(f); continue; }
    if (JSON_RE.test(base)) { groups.report.push(f); continue; }
    if (/\.(txt|log|md)$/i.test(base)) { groups.log.push(f); continue; }
    if (!IMG_RE.test(base)) { groups.other.push(f); continue; }
    if (/(^|\/)(inputs?|source|sources|views?|captures?|frames?)\//.test(dirPart)) { groups.capture.push(f); continue; }
    if (/mask|matte/.test(base)) {
      if (/human|u2net/.test(base)) groups.segHuman.push(f);
      else if (/portrait|birefnet/.test(base)) groups.segPortrait.push(f);
      else if (/fused|fusion/.test(base)) groups.segFused.push(f);
      else groups.segMask.push(f);
      continue;
    }
    if (/cutout|rgba|alpha/.test(base)) { groups.segCutout.push(f); continue; }
    if (/model[-_]?input|gray|input/.test(base)) { groups.input.push(f); continue; }
    if (/contact[-_]?sheet/.test(base)) { groups.render.push(f); continue; }
    if (/(render|screenshot|snapshot|turntable|orbit|ortho|preview|capture)/.test(base) || /(render|screenshot|turn)/.test(dirPart)) { groups.render.push(f); continue; }
    if (/(front|back|left|right|side|top)/.test(base)) { groups.capture.push(f); continue; }
    groups.other.push(f);
  }
  return groups;
}

function detectModel(id, files, reportInfo) {
  const fromReport = reportInfo?.summary?.model ? String(reportInfo.summary.model).toLowerCase() : '';
  const hay = `${id} ${fromReport} ${files.map((f) => f.rel).join(' ')}`.toLowerCase();
  if (/hunyuan/.test(hay)) return { key: 'hunyuan3d-2mv', label: 'Hunyuan3D-2mv', views: '四视图' };
  if (/visual[-_ ]?hull/.test(hay)) return { key: 'visual-hull', label: 'Visual hull（多视图雕刻）', views: '四视图' };
  if (/triposr|tripo-sr/.test(hay)) return { key: 'triposr', label: 'TripoSR', views: '单视图' };
  if (/meshy/.test(hay)) return { key: 'meshy', label: 'Meshy', views: '单图' };
  return { key: 'unknown', label: '未知模型', views: '未知' };
}

function pickPrimaryReport(reports) {
  for (const name of REPORT_NAMES) {
    const hit = reports.find((r) => r.name.toLowerCase() === name);
    if (hit) return hit;
  }
  return reports[0] || null;
}

const JOB_CACHE = new Map();

function scanJob(id) {
  if (!/^[\w.-]+$/.test(id) || id.startsWith('_')) return null;
  const dir = path.join(JOBS_DIR, id);
  let st;
  try { st = statSync(dir); } catch { return null; }
  if (!st.isDirectory()) return null;
  const files = walkFiles(dir);
  const sig = `${id}|${files.map((f) => `${f.rel}:${f.size}:${f.mtimeMs}`).join('|')}`;
  const cached = JOB_CACHE.get(id);
  if (cached && cached.sig === sig) return cached.value;
  const value = buildJobBase(id, dir, files);
  JOB_CACHE.set(id, { sig, value });
  return value;
}

function buildJobBase(id, dir, files) {
  const reports = files
    .filter((f) => JSON_RE.test(f.rel) && f.size < 2 * 1024 * 1024)
    .map((f) => {
      const info = readJsonSync(f.abs);
      return {
        name: path.basename(f.rel),
        rel: f.rel,
        url: jobFileUrl(id, f.rel),
        bytes: f.size,
        mtimeMs: f.mtimeMs,
        json: info.json,
        error: info.error,
      };
    })
    .sort((a, b) => {
      const ai = REPORT_NAMES.indexOf(a.name.toLowerCase());
      const bi = REPORT_NAMES.indexOf(b.name.toLowerCase());
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
    });
  const primaryReport = pickPrimaryReport(reports);
  const reportInfo = buildReportInfo(primaryReport);
  const classified = classifyJobFiles(id, files);
  const model = detectModel(id, files, reportInfo);
  const glbs = classified.glb.map((f) => {
    const analysis = analyzeGlb(f.abs);
    return {
      rel: f.rel,
      name: path.basename(f.rel),
      size: f.size,
      mtimeMs: f.mtimeMs,
      url: jobFileUrl(id, f.rel),
      downloadUrl: `${jobFileUrl(id, f.rel)}?download=1`,
      analysis,
    };
  });
  const imageFiles = files.filter((f) => IMG_RE.test(f.rel));
  const firstImage = classified.segCutout[0] || classified.input[0] || classified.capture[0] || classified.render[0] || imageFiles[0] || null;
  const thumbnail = firstImage ? jobFileUrl(id, firstImage.rel) : null;
  const live = tasks.get(id);
  const now = Date.now();
  let createdAt = files.length ? files.reduce((min, f) => Math.min(min, f.mtimeMs), now) : now;
  let updatedAt = files.length ? files.reduce((max, f) => Math.max(max, f.mtimeMs), 0) : now;
  if (reportInfo?.json) {
    const fields = reportInfo.fields;
    const start = findDate(fields, ['created_at', 'createdat', 'started_at', 'start_time', 'startedat', 'timestamp', 'date']);
    if (start) createdAt = start;
    const end = findDate(fields, ['finished_at', 'ended_at', 'completed_at', 'updated_at', 'generated_at', 'finishedat']);
    if (end) updatedAt = Math.max(updatedAt, end);
    else if (start && reportInfo.summary?.durationSeconds) updatedAt = Math.max(updatedAt, start + reportInfo.summary.durationSeconds * 1000);
  }
  const reportStatus = reportInfo?.summary?.status || null;
  const bestGlb = glbs.find((g) => g.analysis?.ok) || glbs[0] || null;
  const glbOk = glbs.some((g) => g.analysis?.ok);
  let status = 'EMPTY';
  let statusSource = 'disk';
  if (live) { status = live.status; statusSource = 'live'; }
  else if (reportStatus === 'FAILED') { status = 'FAILED'; statusSource = 'report'; }
  else if (glbOk) { status = 'SUCCEEDED'; statusSource = 'disk'; }
  else if (reportStatus) { status = reportStatus; statusSource = 'report'; }
  else if (files.length) { status = 'PARTIAL'; statusSource = 'disk'; }
  const captureViews = classified.capture
    .filter((f) => IMG_RE.test(f.rel))
    .map((f) => ({ angle: angleFromName(path.basename(f.rel)), url: jobFileUrl(id, f.rel), name: path.basename(f.rel), rel: f.rel }));
  const seenAngles = new Set(captureViews.map((v) => v.angle));
  const missingViews = VIEW_ORDER.filter((a) => !seenAngles.has(a));
  const glbSummary = bestGlb?.analysis?.ok ? {
    url: bestGlb.url,
    downloadUrl: bestGlb.downloadUrl,
    sizeBytes: bestGlb.size,
    vertices: bestGlb.analysis.vertices,
    faces: bestGlb.analysis.faces,
    watertight: bestGlb.analysis.watertight,
    components: bestGlb.analysis.components,
    hasSkeleton: bestGlb.analysis.hasSkeleton,
    hasMorphTargets: bestGlb.analysis.hasMorphTargets,
    materials: bestGlb.analysis.materials,
    textures: bestGlb.analysis.textures,
    images: bestGlb.analysis.images,
    dimensions: bestGlb.analysis.dimensions,
    generator: bestGlb.analysis.generator,
  } : null;
  return {
    id,
    dirRel: relPath(dir),
    model,
    status: live ? live.status : status,
    statusSource: live ? 'live' : statusSource,
    stage: live?.stage || (glbOk ? 'complete' : null),
    createdAt,
    updatedAt,
    artifactCount: files.length,
    sizeBytes: files.reduce((sum, f) => sum + f.size, 0),
    thumbnail,
    hasReport: Boolean(primaryReport),
    reportName: primaryReport?.name || null,
    reportFiles: reports,
    reportInfo,
    glbs,
    glbSummary,
    captureViews,
    missingViews,
    classified,
    files,
    metrics: {
      durationSeconds: reportInfo?.summary?.durationSeconds ?? null,
      vramPeakMb: reportInfo?.summary?.vramPeakMb ?? null,
      vertexCount: glbSummary?.vertices ?? reportInfo?.summary?.vertexCount ?? null,
      faceCount: glbSummary?.faces ?? reportInfo?.summary?.faceCount ?? null,
      watertight: glbSummary?.watertight ?? reportInfo?.summary?.watertightReported ?? null,
      components: glbSummary?.components ?? null,
    },
    error: live?.error || null,
  };
}

function summaryOf(job) {
  return {
    id: job.id,
    dirRel: job.dirRel,
    model: job.model,
    status: job.status,
    statusSource: job.statusSource,
    stage: job.stage,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    artifactCount: job.artifactCount,
    sizeBytes: job.sizeBytes,
    thumbnail: job.thumbnail,
    hasReport: job.hasReport,
    reportName: job.reportName,
    captureViews: job.captureViews,
    missingViews: job.missingViews,
    metrics: job.metrics,
    glb: job.glbSummary,
    error: job.error,
  };
}

function listJobIds() {
  try {
    return readdirSync(JOBS_DIR, { withFileTypes: true })
      .filter((e) => e.isDirectory() && !e.name.startsWith('_') && !e.name.startsWith('.'))
      .map((e) => e.name)
      .sort();
  } catch { return []; }
}

function listJobSummaries() {
  return listJobIds()
    .map((id) => scanJob(id))
    .filter(Boolean)
    .map(summaryOf)
    .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
}

/* ------------------------------------------------------------------ */
/* 抠图 QC：processed-v2 优先                                          */
/* ------------------------------------------------------------------ */

const QC_VERSION_DEFS = [
  { key: 'processed-v3', label: 'v3 手工修版', note: '工作台画笔擦除手持布料后的版本；可一键重跑四视图。', priority: 0 },
  { key: 'processed-v2', label: 'v2 修复版', note: '修复缺失鞋脚后的版本；含 mask-manifest.json 角度建议。', priority: 1 },
  { key: 'processed-approved', label: 'v1 已验收融合', note: 'u2net_human_seg × birefnet-portrait 融合 + 最大连通域清理。', priority: 2 },
  { key: 'processed-u2net-human', label: 'u2net_human_seg 单模型', note: 'U²-Net 人体分割单模型输出，用于对照。', priority: 3 },
  { key: 'processed-birefnet-portrait', label: 'birefnet-portrait 单模型', note: 'BiRefNet 人像边缘单模型输出，用于对照。', priority: 4 },
];

/* ---- 手工修蒙版 + 一键多视图重跑 ---- */
const MV_VIEW_FILES = {
  front: '9ca2f019a28896cbef7fc86359863ddc',
  left: '9e01afc1d631e3f3d42de46bd77545ac',
  right: '52a9f704b1daa15643d7e6d0d13e16d0',
  back: '01139015a4830e26611025860662f5c9',
};
const MV_VIEW_LABELS = { front: '正面', left: '左侧', right: '右侧', back: '背面' };
const MASK_EDIT_VERSION = 'processed-v3';
const HOME_PYTHON = TRIPOSR_PYTHON || path.join(GEN_ROOT, 'envs', 'triposr', 'Scripts', 'python.exe');
const HY_PYTHON = path.join(GEN_ROOT, 'envs', 'hunyuan3d', 'Scripts', 'python.exe');
let multiviewRun = null; // { startedAt, outDir, maskVersion, stage, done, error, log }
let multiviewStarting = false;
let maskSaveBusy = false;

function runPython(scriptPath, args, { cwd = GEN_ROOT, timeoutMs = 120000, env = {}, logPath = null } = {}) {
  // 注意：本机沙箱禁止子进程使用管道（spawn 默认 stdio: 'pipe' 会 EPERM），
  // 因此把 stdout/stderr 直接重定向到日志文件，再用文件内容组装错误信息。
  const logFile = logPath || path.join(GEN_ROOT, 'output', `_py-${Date.now()}-${Math.random().toString(16).slice(2, 8)}.log`);
  return new Promise((resolve, reject) => {
    let fd;
    try { fd = openSync(logFile, 'a'); } catch (e) { reject(new Error(`无法创建日志：${e.message}`)); return; }
    let child;
    try {
      child = spawn(HOME_PYTHON, [scriptPath, ...args], {
        cwd, windowsHide: true, env: { ...process.env, ...env }, stdio: ['ignore', fd, fd],
      });
    } catch (e) {
      try { closeSync(fd); } catch { /* ignore */ }
      reject(e);
      return;
    } finally {
      try { closeSync(fd); } catch { /* ignore */ }
    }
    const timer = setTimeout(() => { try { child.kill(); } catch { /* ignore */ } reject(new Error('python 执行超时')); }, timeoutMs);
    child.on('error', (e) => { clearTimeout(timer); reject(e); });
    child.on('close', (code) => {
      clearTimeout(timer);
      let text = '';
      try { text = readFileSync(logFile, 'utf8'); } catch { /* ignore */ }
      if (code === 0) resolve({ out: text, err: '', logFile });
      else reject(new Error((`python 退出码 ${code}\n` + text.slice(-700)).trim()));
    });
  });
}

function maskEditState() {
  const v3 = path.join(TEST_IMAGES_DIR, MASK_EDIT_VERSION);
  const active = existsSync(v3) ? MASK_EDIT_VERSION : 'processed-v2';
  const views = {};
  for (const [angle, stem] of Object.entries(MV_VIEW_FILES)) {
    views[angle] = {
      angle,
      label: MV_VIEW_LABELS[angle] || angle,
      id: stem,
      sourceUrl: qcSourceUrl(`${stem}.jpg`),
      cutoutUrl: qcFileUrl(active, `${stem}-cutout.png`),
      maskUrl: qcFileUrl(active, `${stem}-mask.png`),
      grayUrl: qcFileUrl(active, `${stem}-gray-preview.png`),
    };
  }
  return {
    ok: true,
    activeVersion: active,
    editableVersion: MASK_EDIT_VERSION,
    baseVersion: 'processed-v2',
    views,
    run: refreshMultiviewStage(multiviewRun),
  };
}

async function startMultiviewRun({ maskVersion, octree }) {
  const outDir = path.join(JOBS_DIR, 'hunyuan3d-2mv-v3');
  const backupDir = path.join(JOBS_DIR, `.hunyuan3d-2mv-v3-previous-${randomUUID()}`);
  let preserved = false;
  if (existsSync(outDir)) {
    await rename(outDir, backupDir);
    preserved = true;
  }
  try {
    await mkdir(outDir, { recursive: true });
    const consoleFile = path.join(outDir, 'console.log');
    const args = [
      'multiview_worker.py', '--preset', 'full', '--octree', String(octree),
      '--steps', '5', '--profile', '4', '--mask-version', maskVersion, '--out-dir', outDir,
    ];
    let fd;
    try { fd = openSync(consoleFile, 'a'); } catch (e) { throw new Error(`无法创建任务日志：${e.message}`); }
    let child;
    try {
      child = spawn(HY_PYTHON, args, {
        cwd: GEN_ROOT, windowsHide: true,
        env: {
          ...process.env,
          HF_HOME: path.join(GEN_ROOT, 'cache', 'huggingface'),
          HY3DGEN_MODELS: path.join(GEN_ROOT, 'weights'),
        },
        stdio: ['ignore', fd, fd],
      });
    } finally {
      try { closeSync(fd); } catch { /* ignore */ }
    }
    const run = { startedAt: Date.now(), outDir, maskVersion, stage: 'starting', done: false, error: null, consoleFile, log: [], backupDir: preserved ? backupDir : null };
    const restorePrevious = async () => {
      if (!preserved) return;
      await rm(outDir, { recursive: true, force: true }).catch(() => {});
      await rename(backupDir, outDir).catch(() => {});
      preserved = false;
    };
    child.on('error', async (e) => {
      await restorePrevious();
      run.done = true; run.error = e.message; run.stage = 'failed';
    });
    child.on('close', async (code) => {
      run.done = true;
      run.stage = code === 0 ? 'complete' : 'failed';
      if (code !== 0) {
        run.error = `python 退出码 ${code}（日志：${path.relative(GEN_ROOT, consoleFile)}）`;
        await restorePrevious();
      } else if (preserved) {
        await rm(backupDir, { recursive: true, force: true }).catch(() => {});
        preserved = false;
      }
      refreshMultiviewStage(run);
      if (multiviewRun === run) setTimeout(() => { if (multiviewRun === run) multiviewRun = null; }, 180000);
    });
    multiviewRun = run;
    return run;
  } catch (e) {
    await rm(outDir, { recursive: true, force: true }).catch(() => {});
    if (preserved) await rename(backupDir, outDir).catch(() => {});
    throw e;
  }
}

async function ensureComparison(targetId) {
  if (!/^[\w.-]+$/.test(targetId)) throw new Error('任务编号无效。');
  const comparisonDir = path.join(GEN_ROOT, 'output', 'comparisons', targetId);
  await mkdir(comparisonDir, { recursive: true });
  for (const name of readdirSync(JOBS_DIR)) {
    const dir = path.join(JOBS_DIR, name);
    if (!existsSync(dir) || !statSync(dir).isDirectory()) continue;
    const glbs = readdirSync(dir).filter((n) => n.toLowerCase().endsWith('.glb'));
    for (const glb of glbs) {
      const preview = path.join(dir, `${path.basename(glb, '.glb')}-preview.png`);
      const stale = existsSync(preview) && statSync(path.join(dir, glb)).mtimeMs > statSync(preview).mtimeMs;
      if (!existsSync(preview) || stale) {
        await runPython('render_glb.py', [path.join(dir, glb), dir, '--size', '420', '--label', name], { timeoutMs: 120000 });
      }
    }
  }
  await runPython('compose_comparison.py', [JOBS_DIR, comparisonDir], { timeoutMs: 120000 });
  const manifest = JSON.parse(readFileSync(path.join(comparisonDir, 'manifest.json'), 'utf8'));
  manifest.contactSheetUrl = `/api/comparisons/${encodeURIComponent(targetId)}/contact-sheet.png?ts=${Date.now()}`;
  manifest.models = (manifest.models || []).map((m) => ({ ...m, previewUrl: `${m.previewUrl}?ts=${Date.now()}` }));
  return manifest;
}

function refreshMultiviewStage(run) {
  if (!run) return run;
  try {
    const stageFile = path.join(run.outDir, 'stage-log.txt');
    if (existsSync(stageFile)) {
      const text = readFileSync(stageFile, 'utf8');
      const matches = [...text.matchAll(/ROVE_STAGE:(\w+)/g)];
      if (matches.length) run.stage = matches[matches.length - 1][1];
      run.log = text.split(/\r?\n/).filter(Boolean).slice(-25);
    }
  } catch { /* ignore */ }
  return run;
}

const QC_CACHE = new Map();
const SOURCE_IMAGE_CACHE = { key: null, value: new Map() };

function listSourceImages() {
  let names = [];
  try { names = readdirSync(TEST_IMAGES_DIR).filter((n) => IMG_RE.test(n)); } catch { names = []; }
  const key = names.map((name) => {
    try { const st = statSync(path.join(TEST_IMAGES_DIR, name)); return `${name}:${st.size}:${st.mtimeMs}`; }
    catch { return `${name}:missing`; }
  }).join('|');
  if (SOURCE_IMAGE_CACHE.key === key) return SOURCE_IMAGE_CACHE.value;
  const map = new Map();
  for (const name of names) {
    try {
      const abs = path.join(TEST_IMAGES_DIR, name);
      const st = statSync(abs);
      map.set(name, { name, abs, size: st.size, mtimeMs: st.mtimeMs, url: qcSourceUrl(name) });
    } catch { /* ignore */ }
  }
  SOURCE_IMAGE_CACHE.key = key;
  SOURCE_IMAGE_CACHE.value = map;
  return map;
}

function sourceStem(name) {
  return String(name || '').replace(/\.[^.]+$/, '');
}

function normalizeAngle(value) {
  if (value == null) return 'unknown';
  const s = String(value).toLowerCase().trim();
  if (!s) return 'unknown';
  if (['front', 'front-view', 'frontview', '正面', '前', '正'].includes(s)) return 'front';
  if (['left', 'left-view', '左侧', '左', '左面'].includes(s)) return 'left';
  if (['right', 'right-view', '右侧', '右', '右面'].includes(s)) return 'right';
  if (['back', 'rear', 'back-view', '背面', '后', '后面'].includes(s)) return 'back';
  if (['side', 'profile', '侧面', '侧身', '左/右侧'].includes(s)) return 'side';
  if (['face', 'face-detail', '面部', '脸部', '面部细节'].includes(s)) return 'face-detail';
  if (['top', '俯视'].includes(s)) return 'top';
  return angleFromName(s);
}

function extractQcRowsFromReport(reportJson, version, filesByName) {
  let arr = null;
  if (Array.isArray(reportJson)) arr = reportJson;
  else if (reportJson && typeof reportJson === 'object') {
    for (const key of ['images', 'items', 'entries', 'masks', 'results', 'samples', 'data', 'rows', 'frames']) {
      if (Array.isArray(reportJson[key])) { arr = reportJson[key]; break; }
    }
    if (!arr) {
      const found = Object.values(reportJson).find((v) => Array.isArray(v) && v.length && typeof v[0] === 'object');
      if (found) arr = found;
    }
  }
  if (!arr) return [];
  const sources = listSourceImages();
  return arr.map((raw, index) => {
    const entry = raw && typeof raw === 'object' ? raw : {};
    const sourceName = path.basename(String(entry.source || entry.source_file || entry.file || entry.input || entry.image || entry.name || ''));
    const stem = sourceStem(sourceName);
    const findFile = (suffixes) => {
      for (const suffix of suffixes) {
        const name = `${stem}${suffix}`;
        if (filesByName.has(name)) return name;
      }
      return null;
    };
    const cutoutName = (entry.cutout && filesByName.has(path.basename(entry.cutout))) ? path.basename(entry.cutout) : findFile(['-cutout.png']);
    const maskName = (entry.mask && filesByName.has(path.basename(entry.mask))) ? path.basename(entry.mask) : findFile(['-mask.png']);
    const grayName = (entry.gray_preview && filesByName.has(path.basename(entry.gray_preview))) ? path.basename(entry.gray_preview) : findFile(['-gray-preview.png', '-gray.png']);
    const rawMaskName = (entry.raw_mask && filesByName.has(path.basename(entry.raw_mask))) ? path.basename(entry.raw_mask) : findFile(['-raw-mask.png']);
    const source = sources.get(sourceName) || sources.get(`${stem}.jpg`) || sources.get(`${stem}.jpeg`) || null;
    const angleRaw = entry.angle ?? entry.view ?? entry.view_angle ?? entry.suggested_angle ?? entry.suggestedAngle
      ?? entry.direction ?? entry.orientation ?? entry.suggested_role ?? entry.role ?? entry.side ?? '';
    const angle = normalizeAngle(angleRaw);
    const reportedFg = [entry.foreground_ratio, entry.foreground_ratio_256, entry.foreground_ratio_512].find((v) => typeof v === 'number') ?? null;
    const reportedEdge = [entry.foreground_on_outer_border_ratio, entry.foreground_on_outer_border_ratio_256, entry.foreground_on_outer_border_ratio_512, entry.edge_leak_ratio].find((v) => typeof v === 'number') ?? null;
    const reportedTouches = typeof entry.touches_frame_edge === 'boolean' ? entry.touches_frame_edge
      : typeof entry.touches_edge === 'boolean' ? entry.touches_edge : null;
    const maskInfo = maskName ? imageMeta(path.join(TEST_IMAGES_DIR, version, maskName)) : null;
    let computed = null;
    let metricsSource = '报告';
    if (reportedFg == null || reportedEdge == null || reportedTouches == null) {
      const probe = maskName || cutoutName;
      if (probe) {
        computed = pngAlphaStats(path.join(TEST_IMAGES_DIR, version, probe));
        if (computed?.supported) metricsSource = '实测';
      }
    }
    const fg = reportedFg ?? computed?.foregroundRatio ?? null;
    const edge = reportedEdge ?? computed?.edgeLeakRatio ?? null;
    const touches = reportedTouches ?? computed?.touchesFrameEdge ?? null;
    const modelChain = entry.model_chain || entry.models || entry.model_chain_desc || entry.model || entry.chain || null;
    const notes = entry.notes || entry.note || entry.remark || entry.remarks || entry.comment || entry.qc_note || null;
    const extra = {};
    for (const [k, v] of Object.entries(entry)) {
      if (['source', 'angle', 'cutout', 'mask', 'gray_preview', 'raw_mask', 'notes'].includes(k)) continue;
      if (Object.keys(extra).length >= 10) break;
      if (['string', 'number', 'boolean'].includes(typeof v)) extra[k] = v;
    }
    return {
      id: stem || `row-${index + 1}`,
      index: index + 1,
      source: sourceName || stem || null,
      sourceUrl: source?.url || null,
      sourceSize: source ? { width: imageMeta(source.abs).width, height: imageMeta(source.abs).height } : null,
      angle,
      angleLabel: ANGLE_LABELS[angle] || '角度未标注',
      role: typeof angleRaw === 'string' && angleRaw.trim() ? angleRaw : (entry.role || null),
      cutoutUrl: cutoutName ? qcFileUrl(version, cutoutName) : null,
      cutoutDownloadUrl: cutoutName ? `${qcFileUrl(version, cutoutName)}?download=1` : null,
      maskUrl: maskName ? qcFileUrl(version, maskName) : null,
      maskDownloadUrl: maskName ? `${qcFileUrl(version, maskName)}?download=1` : null,
      grayUrl: grayName ? qcFileUrl(version, grayName) : null,
      rawMaskUrl: rawMaskName ? qcFileUrl(version, rawMaskName) : null,
      size: maskInfo?.width ? [maskInfo.width, maskInfo.height] : null,
      metrics: {
        foregroundRatio: round4(fg),
        edgeLeakRatio: round4(edge),
        touchesFrameEdge: touches,
        source: metricsSource,
      },
      modelChain,
      notes,
      seconds: typeof entry.seconds === 'number' ? entry.seconds : null,
      extra,
    };
  });
}

function deriveQcRowsFromFiles(version, files) {
  const byName = new Set(files.map((f) => f.name));
  const sources = listSourceImages();
  const stems = new Set();
  for (const name of byName) {
    const m = name.match(/^(.+?)-(cutout|mask|gray-preview|raw-mask)\.png$/i);
    if (m) stems.add(m[1]);
  }
  return [...stems].sort().map((stem, index) => {
    const source = sources.get(`${stem}.jpg`) || sources.get(`${stem}.jpeg`) || sources.get(`${stem}.png`) || null;
    const maskName = byName.has(`${stem}-mask.png`) ? `${stem}-mask.png` : null;
    const computed = maskName ? pngAlphaStats(path.join(TEST_IMAGES_DIR, version, maskName)) : null;
    const angle = angleFromName(stem);
    return {
      id: stem,
      index: index + 1,
      source: source?.name || `${stem}.jpg`,
      sourceUrl: source?.url || null,
      sourceSize: source ? { width: imageMeta(source.abs).width, height: imageMeta(source.abs).height } : null,
      angle,
      angleLabel: ANGLE_LABELS[angle] || '角度未标注',
      role: null,
      cutoutUrl: byName.has(`${stem}-cutout.png`) ? qcFileUrl(version, `${stem}-cutout.png`) : null,
      cutoutDownloadUrl: null,
      maskUrl: maskName ? qcFileUrl(version, maskName) : null,
      maskDownloadUrl: null,
      grayUrl: byName.has(`${stem}-gray-preview.png`) ? qcFileUrl(version, `${stem}-gray-preview.png`) : null,
      rawMaskUrl: byName.has(`${stem}-raw-mask.png`) ? qcFileUrl(version, `${stem}-raw-mask.png`) : null,
      size: computed?.width ? [computed.width, computed.height] : null,
      metrics: {
        foregroundRatio: round4(computed?.foregroundRatio ?? null),
        edgeLeakRatio: round4(computed?.edgeLeakRatio ?? null),
        touchesFrameEdge: computed?.touchesFrameEdge ?? null,
        source: '实测',
      },
      modelChain: null,
      notes: null,
      seconds: null,
      extra: {},
    };
  });
}

function scanQcVersion(key) {
  const def = QC_VERSION_DEFS.find((v) => v.key === key);
  const label = def?.label || key;
  const dir = path.join(TEST_IMAGES_DIR, key);
  if (!existsSync(dir)) {
    return {
      key, label, note: def?.note || '', exists: false, ready: false, entries: [], files: [],
      contactSheetUrl: null, reportName: null, reportUrl: null,
      coverage: { present: [], missing: VIEW_ORDER, counts: {} },
    };
  }
  const files = listTopFiles(dir).sort((a, b) => a.name.localeCompare(b.name));
  const sig = files.map((f) => `${f.name}:${f.size}:${f.mtimeMs}`).join('|');
  const cached = QC_CACHE.get(key);
  if (cached && cached.sig === sig) return cached.value;
  const filesByName = new Map(files.map((f) => [f.name, f]));
  const reportFile = ['mask-manifest.json', 'quality-report.json', 'report.json', 'manifest.json']
    .map((name) => filesByName.get(name))
    .find(Boolean) || null;
  const reportParsed = reportFile ? readJsonSync(reportFile.abs) : { json: null, error: null };
  let entries = extractQcRowsFromReport(reportParsed.json, key, filesByName);
  if (!entries.length) entries = deriveQcRowsFromFiles(key, files);
  const counts = {};
  for (const entry of entries) if (entry.angle && entry.angle !== 'unknown') counts[entry.angle] = (counts[entry.angle] || 0) + 1;
  const present = VIEW_ORDER.filter((a) => counts[a]);
  const missing = VIEW_ORDER.filter((a) => !counts[a]);
  const contact = files.find((f) => /contact[-_]?sheet/i.test(f.name)) || null;
  const value = {
    key,
    label,
    note: def?.note || '',
    dirRel: relPath(dir),
    exists: true,
    ready: entries.length > 0,
    reportName: reportFile?.name || null,
    reportUrl: reportFile ? qcFileUrl(key, reportFile.name) : null,
    reportDownloadUrl: reportFile ? `${qcFileUrl(key, reportFile.name)}?download=1` : null,
    reportParseError: reportParsed.error,
    reportJson: reportParsed.json && JSON.stringify(reportParsed.json).length < 300000 ? reportParsed.json : null,
    contactSheetUrl: contact ? qcFileUrl(key, contact.name) : null,
    files,
    entries,
    coverage: { present, missing, counts },
  };
  QC_CACHE.set(key, { sig, value });
  return value;
}

function qcOverview() {
  const versions = QC_VERSION_DEFS.map((def) => {
    const scanned = scanQcVersion(def.key);
    return {
      key: def.key,
      label: def.label,
      note: def.note,
      exists: scanned.exists,
      ready: scanned.ready,
      entryCount: scanned.entries.length,
      reportName: scanned.reportName || null,
      coverage: scanned.coverage,
      contactSheetUrl: scanned.contactSheetUrl || null,
    };
  });
  const preferred = versions.find((v) => v.ready)?.key || versions[0].key;
  return { preferred, versions };
}

/* ------------------------------------------------------------------ */
/* 详情：分阶段面板数据                                                 */
/* ------------------------------------------------------------------ */

function makeArtifact(jobId, f, extra = {}) {
  const meta = IMG_RE.test(f.rel) ? imageMeta(f.abs) : null;
  return {
    rel: f.rel,
    name: path.basename(f.rel),
    size: f.size,
    mtimeMs: f.mtimeMs,
    url: jobFileUrl(jobId, f.rel),
    downloadUrl: `${jobFileUrl(jobId, f.rel)}?download=1`,
    image: meta,
    ...extra,
  };
}

function segMetricsFromFile(f) {
  if (!f || !IMG_RE.test(f.rel) || !/\.png$/i.test(f.rel)) return null;
  const stats = pngAlphaStats(f.abs);
  if (!stats || !stats.supported) return null;
  return {
    foregroundRatio: round4(stats.foregroundRatio),
    softAlphaRatio: round4(stats.softAlphaRatio),
    edgeLeakRatio: round4(stats.edgeLeakRatio),
    touchesFrameEdge: stats.touchesFrameEdge,
    bbox: stats.bbox,
    width: stats.width,
    height: stats.height,
  };
}

function buildStages(job) {
  const id = job.id;
  const groups = job.classified;
  const reportSummary = job.reportInfo?.summary || null;
  const stages = [];

  /* 1. 素材与视角 */
  const captureArtifacts = groups.capture.map((f) => {
    const angle = angleFromName(path.basename(f.rel));
    return makeArtifact(id, f, { angle, angleLabel: ANGLE_LABELS[angle] });
  });
  stages.push({
    key: 'capture',
    title: '素材与视角',
    status: captureArtifacts.length ? (job.model.key === 'hunyuan3d-2mv' && captureArtifacts.length >= 4 ? 'done' : 'partial') : 'missing',
    summary: captureArtifacts.length
      ? `任务目录内有 ${captureArtifacts.length} 张视图输入；缺失角度：${job.missingViews.length ? job.missingViews.map((a) => ANGLE_LABELS[a]).join('、') : '无'}`
      : '任务目录没有原始照片或视频帧（原始素材按隐私设计不落盘）',
    artifacts: captureArtifacts,
    views: job.captureViews,
    missingViews: job.missingViews,
    notes: captureArtifacts.length === 0
      ? [`${job.model.label} 流水线只在内存中读取原始照片，任务目录只保留处理产物。素材与角度标注请查看下方「批量抠图 QC 素材库」。`]
      : (job.missingViews.length ? [`Hunyuan3D-2mv 需要 front / left / right / back 四张视图，当前缺少：${job.missingViews.map((a) => ANGLE_LABELS[a]).join('、')}。补拍后再跑多视图更可靠。`] : []),
    metrics: captureArtifacts.length ? [
      { label: '视图数量', value: captureArtifacts.length, source: '磁盘文件' },
      { label: '已覆盖角度', value: captureArtifacts.map((a) => a.angleLabel).join('、') || '未标注', source: '文件名' },
      { label: '缺失角度', value: job.missingViews.length ? job.missingViews.map((a) => ANGLE_LABELS[a]).join('、') : '无', source: '文件名' },
    ] : [],
  });

  /* 2. 抠图 */
  const segSlots = [
    { key: 'human', label: 'u2net_human_seg 蒙版', files: groups.segHuman, hint: 'U²-Net 人体分割模型原始输出' },
    { key: 'portrait', label: 'birefnet-portrait 蒙版', files: groups.segPortrait, hint: 'BiRefNet 人像边缘模型原始输出' },
    { key: 'fused', label: '融合蒙版', files: groups.segFused, hint: '两模型蒙版几何平均融合，再保留主体连通区域' },
    { key: 'mask', label: '最终 alpha 蒙版', files: groups.segMask, hint: '最终用于合成中灰底输入的主体蒙版' },
    { key: 'cutout', label: '最终透明抠图', files: groups.segCutout, hint: '去除背景后的 RGBA 抠图' },
  ].map((slot) => {
    const f = slot.files[0] || null;
    return {
      key: slot.key,
      label: slot.label,
      hint: slot.hint,
      present: Boolean(f),
      artifact: f ? makeArtifact(id, f) : null,
      metrics: f ? segMetricsFromFile(f) : null,
    };
  });
  const segPresent = segSlots.filter((s) => s.present).length;
  stages.push({
    key: 'segmentation',
    title: '抠图与蒙版',
    status: segPresent === 0 ? 'missing' : (segPresent >= 5 ? 'done' : 'partial'),
    summary: segPresent ? `任务内已有 ${segPresent}/5 类蒙版与抠图产物；批量 QC 数据集见下方（v2 优先）` : '任务目录没有抠图产物',
    slots: segSlots,
    notes: !job.hasReport ? ['本任务没有写抠图 QC 报告文件；下方指标由服务端从 PNG 的 alpha 通道实测计算。'] : [],
    metrics: [],
  });

  /* 3. 模型输入 */
  const inputCandidates = job.model.key === 'hunyuan3d-2mv'
    ? [...captureArtifacts, ...groups.input.map((f) => makeArtifact(id, f))]
    : [...groups.input.map((f) => makeArtifact(id, f)), ...captureArtifacts];
  const inputArtifacts = inputCandidates.map((a) => ({
    ...a,
    inputFor: job.model.label,
    note: job.model.key === 'hunyuan3d-2mv' ? 'Hunyuan3D-2mv 四视图输入' : 'TripoSR 单视图输入',
  }));
  stages.push({
    key: 'input',
    title: '模型输入',
    status: inputArtifacts.length ? 'done' : 'missing',
    summary: inputArtifacts.length
      ? `${inputArtifacts.length} 张输入将送入 ${job.model.label}（${job.model.views}）`
      : '任务目录没有模型输入图',
    artifacts: inputArtifacts,
    model: job.model,
    notes: job.model.key === 'triposr'
      ? ['TripoSR 只消费 1 张 512 × 512 中灰底主视图；其他视角不会被该模型使用。']
      : (job.model.key === 'hunyuan3d-2mv' ? ['Hunyuan3D-2mv 消费 front / left / right / back 四张视图；缺一不可。'] : []),
    metrics: inputArtifacts.length ? [{ label: '输入张数', value: inputArtifacts.length, source: '磁盘文件' }] : [],
  });

  /* 4. 几何生成 */
  const geometryArtifacts = job.glbs.map((g) => ({
    rel: g.rel,
    name: g.name,
    size: g.size,
    mtimeMs: g.mtimeMs,
    url: g.url,
    downloadUrl: g.downloadUrl,
    analysis: g.analysis,
  }));
  const hasGlb = geometryArtifacts.some((g) => g.analysis?.ok);
  stages.push({
    key: 'geometry',
    title: '几何生成',
    status: hasGlb ? 'done' : (geometryArtifacts.length ? 'failed' : 'missing'),
    summary: hasGlb
      ? `${job.model.label}（${job.model.views}）· ${job.glbSummary.vertices} 顶点 · ${job.glbSummary.faces} 面`
      : '没有可解析的 GLB 产物',
    artifacts: geometryArtifacts,
    report: job.reportInfo ? {
      name: job.reportInfo.name,
      url: job.reportInfo.url,
      downloadUrl: job.reportInfo.downloadUrl,
      summary: reportSummary,
    } : null,
    notes: hasGlb ? [
      '顶点 / 面 / watertight / 连通组件由服务端直接解析 GLB 实测得出。',
      job.hasReport ? `参数、耗时、显存峰值取自 ${job.reportInfo.name}。` : '任务目录没有生成报告，参数、耗时、显存峰值标记为「缺失」。',
    ] : [],
    metrics: [],
  });

  /* 5. 验收 */
  stages.push({
    key: 'acceptance',
    title: '验收检查',
    status: 'partial',
    summary: '自动检查闭合性、连通组件、骨架、形态键、纹理与多角度截图；明确区分已实现与未实现',
    ...buildAcceptance(job),
  });

  return stages;
}

function reportMetaOk(job, key) {
  return job.metrics?.[key] != null;
}

function buildAcceptance(job) {
  const best = job.glbs.find((g) => g.analysis?.ok)?.analysis || null;
  const checks = [];
  const add = (key, label, state, value, detail) => checks.push({ key, label, state, value: value ?? null, detail: detail ?? null });
  if (best) {
    add('glb', 'GLB 可解析', 'pass', `${best.vertices} 顶点 · ${best.faces} 三角面 · ${best.primitives} 个 primitive`, `生成器：${best.generator || '未知'}`);
    add('watertight', '网格闭合（watertight）', best.watertight ? 'pass' : 'fail',
      best.watertight ? '是' : '否',
      `边界边 ${best.boundaryEdges} 条 · 非流形边 ${best.nonManifoldEdges} 条 · 连通组件 ${best.components} 个`);
    add('components', '连通组件', best.components === 1 ? 'pass' : 'warn', `${best.components} 个`, best.components === 1 ? '单一连通主体' : '存在多个互不相连的部件，需人工确认是否为造型的一部分');
    add('skeleton', '骨架（GLB skins）', best.hasSkeleton ? 'pass' : 'unimplemented',
      best.hasSkeleton ? `${best.skins} 个骨架` : '未实现 · GLB 不含骨架',
      best.hasSkeleton ? null : '当前不会生成骨骼，不能直接用于 Rove 动捕；需要后续自动绑定阶段。');
    add('morph', '面部形态键（morph targets）', best.hasMorphTargets ? 'pass' : 'unimplemented',
      best.hasMorphTargets ? `${best.morphTargets} 组形态键` : '未实现 · GLB 不含形态键',
      best.hasMorphTargets ? null : '当前不包含面部 BlendShape，表情捕捉无法直接映射。');
    add('texture', '纹理 / 材质', best.materials > 0 ? 'pass' : 'unimplemented',
      best.materials > 0 ? `${best.materials} 个材质 · ${best.images} 张贴图` : '未实现 · 无材质与贴图',
      best.materials > 0 ? null : (best.vertexColors ? 'GLB 只带顶点颜色（COLOR_0），不是纹理贴图。' : 'GLB 只包含几何。'));
    add('animation', '动画 / 动作', best.hasAnimations ? 'pass' : 'unimplemented',
      best.hasAnimations ? `${best.animations} 段动画` : '未实现 · 无动画', null);
  } else if (job.glbs.length) {
    add('glb', 'GLB 可解析', 'fail', '解析失败', job.glbs.map((g) => g.analysis?.error).filter(Boolean).join('；') || '文件损坏');
    add('watertight', '网格闭合（watertight）', 'missing', null, 'GLB 无法解析，无法判断');
    add('skeleton', '骨架（GLB skins）', 'missing', null, 'GLB 无法解析，无法判断');
    add('morph', '面部形态键（morph targets）', 'missing', null, 'GLB 无法解析，无法判断');
    add('texture', '纹理 / 材质', 'missing', null, 'GLB 无法解析，无法判断');
  } else {
    add('glb', 'GLB 可解析', 'missing', '缺失 · 没有 GLB 文件', '任务目录没有生成网格');
    add('watertight', '网格闭合（watertight）', 'missing', null, '没有 GLB，无法检查');
    add('skeleton', '骨架（GLB skins）', 'missing', null, '没有 GLB，无法检查');
    add('morph', '面部形态键（morph targets）', 'missing', null, '没有 GLB，无法检查');
    add('texture', '纹理 / 材质', 'missing', null, '没有 GLB，无法检查');
  }
  const shots = job.classified.render
    .filter((f) => IMG_RE.test(f.rel))
    .map((f) => {
      const angle = angleFromName(path.basename(f.rel));
      return { angle, angleLabel: ANGLE_LABELS[angle] || '未标注', ...makeArtifact(job.id, f) };
    })
    .filter((f) => f.angle !== 'unknown');
  add('screenshots', '正 / 侧 / 背验收截图', shots.length ? 'pass' : 'missing',
    shots.length ? `${shots.length} 张` : '缺失 · 本任务没有生成模型多角度截图',
    '照片素材里的正/侧/背是拍摄原图，不能当作生成模型的验收截图。');
  const hasDuration = reportMetaOk(job, 'durationSeconds');
  const hasVram = reportMetaOk(job, 'vramPeakMb');
  add('report', '生成报告（参数 / 耗时 / 显存）', hasDuration && hasVram ? 'pass' : (job.hasReport ? 'warn' : 'missing'),
    job.hasReport ? `${job.reportInfo.name}${hasDuration ? ` · 耗时 ${job.metrics.durationSeconds} 秒` : ' · 耗时缺失'}${hasVram ? ` · 显存峰值 ${job.metrics.vramPeakMb} MB` : ' · 显存缺失'}` : '缺失 · 没有 generation-report.json',
    job.hasReport ? '字段缺失时页面显示「缺失」，不会用估计值替代。' : '没有报告文件；耗时、显存、参数无法确认。');
  add('binding', '动捕绑定与 retarget 兼容', 'unimplemented', '未实现 · 尚未接入自动绑定器', '绑定成功也不等于 Rove 动捕兼容，需要单独验收。');
  add('download', 'GLB 下载', best ? 'pass' : 'missing', best ? '可下载' : '缺失', null);
  const summary = {
    pass: checks.filter((c) => c.state === 'pass').length,
    fail: checks.filter((c) => c.state === 'fail').length,
    warn: checks.filter((c) => c.state === 'warn').length,
    missing: checks.filter((c) => c.state === 'missing').length,
    unimplemented: checks.filter((c) => c.state === 'unimplemented').length,
  };
  return { checks, screenshots: shots, checkSummary: summary };
}

function buildJobDetail(id) {
  const job = scanJob(id);
  if (!job) return null;
  const qc = qcOverview();
  const preferred = scanQcVersion(qc.preferred);
  return {
    job: summaryOf(job),
    full: {
      dirRel: job.dirRel,
      reportInfo: job.reportInfo ? {
        name: job.reportInfo.name,
        url: job.reportInfo.url,
        downloadUrl: job.reportInfo.downloadUrl,
        parseError: job.reportInfo.parseError,
        summary: job.reportInfo.summary,
        fields: job.reportInfo.fields.slice(0, 120),
        json: job.reportInfo.bytes && job.reportInfo.bytes < 400000 ? job.reportInfo.json : null,
        bytes: job.reportInfo.bytes,
      } : null,
      fileList: job.files
        .slice()
        .sort((a, b) => a.rel.localeCompare(b.rel))
        .map((f) => ({
          rel: f.rel,
          name: path.basename(f.rel),
          size: f.size,
          mtimeMs: f.mtimeMs,
          url: jobFileUrl(id, f.rel),
          downloadUrl: `${jobFileUrl(id, f.rel)}?download=1`,
        })),
      reportFiles: job.reportFiles.map((r) => ({
        name: r.name,
        rel: r.rel,
        size: r.bytes,
        mtimeMs: r.mtimeMs,
        url: r.url,
        downloadUrl: `${r.url}?download=1`,
        parseError: r.error,
        keys: r.json && typeof r.json === 'object' && !Array.isArray(r.json) ? Object.keys(r.json).slice(0, 40) : null,
      })),
    },
    stages: buildStages(job),
    qc: { preferred, versions: qc.versions, active: preferred },
  };
}

/* ------------------------------------------------------------------ */
/* 静态文件服务                                                        */
/* ------------------------------------------------------------------ */

function sendFile(req, res, file, { download = false } = {}) {
  let st;
  try { st = statSync(file); } catch {
    return json(res, 404, { error: '文件不存在。' });
  }
  if (!st.isFile()) return json(res, 404, { error: '不是文件。' });
  const mime = MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';
  const headers = {
    'content-type': mime,
    'cache-control': 'no-cache',
    'accept-ranges': 'bytes',
    'content-length': st.size,
    'x-content-type-options': 'nosniff',
  };
  if (download) {
    const name = path.basename(file);
    headers['content-disposition'] = `attachment; filename="${name.replace(/[^\w.-]+/g, '_')}"; filename*=UTF-8''${encodeURIComponent(name)}`;
  }
  const range = req.headers.range;
  if (range) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(String(range));
    if (m) {
      let start = m[1] ? Number(m[1]) : 0;
      let end = m[2] ? Number(m[2]) : st.size - 1;
      if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= st.size) {
        res.writeHead(416, { 'content-range': `bytes */${st.size}` });
        return res.end();
      }
      end = Math.min(end, st.size - 1);
      res.writeHead(206, { ...headers, 'content-range': `bytes ${start}-${end}/${st.size}`, 'content-length': end - start + 1 });
      if (req.method === 'HEAD') return res.end();
      return createReadStream(file, { start, end }).pipe(res);
    }
  }
  res.writeHead(200, headers);
  if (req.method === 'HEAD') return res.end();
  return createReadStream(file).pipe(res);
}

/* ------------------------------------------------------------------ */
/* HTTP 处理                                                           */
/* ------------------------------------------------------------------ */

function decodeBase64Strict(value) {
  if (typeof value !== 'string' || !value || value.length % 4 === 1 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value) || (value.includes('=') && !/=+$/.test(value))) return null;
  const bytes = Buffer.from(value, 'base64');
  if (!bytes.length || bytes.toString('base64').replace(/=+$/, '') !== value.replace(/=+$/, '')) return null;
  return bytes;
}

function pngDimensions(bytes) {
  if (!Buffer.isBuffer(bytes) || bytes.length < 24 || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) || bytes.toString('ascii', 12, 16) !== 'IHDR') return null;
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

async function readBody(req, limit = 27_000_000) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new Error('请求体过大。');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function handle(req, res) {
  const origin = req.headers.origin;
  if (origin && origin !== allowedOrigin && !isLocalOrigin(origin)) {
    return json(res, 403, { error: '此来源未获本地服务允许；请从本机 localhost 页面打开。' });
  }
  if (origin) {
    res.setHeader('access-control-allow-origin', origin);
    res.setHeader('vary', 'Origin');
  }
  res.setHeader('access-control-allow-methods', 'GET,POST,DELETE,OPTIONS');
  res.setHeader('access-control-allow-headers', 'content-type');
  res.setHeader('x-content-type-options', 'nosniff');
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }

  const url = new URL(req.url || '/', 'http://localhost');
  const pathname = url.pathname;

  if (req.method === 'GET' && (pathname === '/' || pathname === '/index.html' || pathname === '/rove-model-studio.html')) {
    return sendFile(req, res, PAGE_FILE);
  }
  if (req.method === 'GET' && pathname === '/favicon.ico') { res.writeHead(204); return res.end(); }

  if (req.method === 'GET' && pathname === '/api/config') {
    const configured = provider === 'local-triposr'
      ? Boolean(process.env.TRIPOSR_PYTHON && process.env.TRIPOSR_SOURCE && existsSync(process.env.TRIPOSR_PYTHON))
      : Boolean(process.env.MESHY_API_KEY);
    return json(res, 200, {
      configured,
      provider: providerLabel,
      mode: provider,
      port: PORT,
      page: '/',
      jobsDir: relPath(JOBS_DIR),
      testImagesDir: relPath(TEST_IMAGES_DIR),
      hasJobsDir: existsSync(JOBS_DIR),
      localTriposr: {
        python: TRIPOSR_PYTHON ? relPath(TRIPOSR_PYTHON) : null,
        source: TRIPOSR_SOURCE ? relPath(TRIPOSR_SOURCE) : null,
      },
    });
  }

  if (req.method === 'GET' && pathname === '/api/jobs') {
    const jobs = listJobSummaries();
    const qc = qcOverview();
    const lineup = [
      { key: 'triposr', label: 'TripoSR', views: '单视图', count: jobs.filter((j) => j.model.key === 'triposr').length },
      { key: 'hunyuan3d-2mv', label: 'Hunyuan3D-2mv', views: '四视图', count: jobs.filter((j) => j.model.key === 'hunyuan3d-2mv').length },
      { key: 'visual-hull', label: 'Visual hull', views: '四视图', count: jobs.filter((j) => j.model.key === 'visual-hull').length },
    ];
    return json(res, 200, {
      ok: true,
      generatedAt: new Date().toISOString(),
      provider: { label: providerLabel, mode: provider },
      storage: {
        jobsDir: relPath(JOBS_DIR),
        jobsDirExists: existsSync(JOBS_DIR),
        testImagesDir: relPath(TEST_IMAGES_DIR),
        testImagesExists: existsSync(TEST_IMAGES_DIR),
        page: relPath(PAGE_FILE),
      },
      lineup,
      qc,
      jobs,
    });
  }

  const jobDetailMatch = pathname.match(/^\/api\/jobs\/([\w.-]+)$/);
  if (req.method === 'GET' && jobDetailMatch) {
    const detail = buildJobDetail(decodeURIComponent(jobDetailMatch[1]));
    if (!detail) return json(res, 404, { error: `任务「${jobDetailMatch[1]}」不存在或不是目录。` });
    return json(res, 200, { ok: true, ...detail });
  }

  const comparisonFileMatch = pathname.match(/^\/api\/comparisons\/([\w.-]+)\/(.+)$/);
  if (req.method === 'GET' && comparisonFileMatch) {
    const id = decodeURIComponent(comparisonFileMatch[1]);
    const rel = decodeURIComponent(comparisonFileMatch[2]);
    const abs = resolveInside(path.join(GEN_ROOT, 'output', 'comparisons', id), rel);
    if (!abs || !existsSync(abs)) return json(res, 404, { error: '对比图不存在。' });
    return sendFile(req, res, abs);
  }

  if (req.method === 'GET' && pathname.startsWith('/api/comparison/')) {
    const id = decodeURIComponent(pathname.slice('/api/comparison/'.length));
    try { return json(res, 200, { ok: true, ...(await ensureComparison(id)) }); }
    catch (e) { return json(res, 500, { error: `生成对比图失败：${e.message}` }); }
  }

  if (req.method === 'POST' && pathname === '/api/feedback') {
    let body;
    try { body = JSON.parse(await readBody(req)); } catch { return json(res, 400, { error: '请求格式无效。' }); }
    const id = String(body.id || '').replace(/\.[^.]+$/, '');
    const angle = String(body.angle || 'unknown');
    if (!/^[0-9a-f]{32}$/i.test(id) || !/^(front|left|right|back|unknown)$/.test(angle)) return json(res, 400, { error: '反馈样本标识无效。' });
    const feedbackId = randomUUID();
    const dir = path.join(GEN_ROOT, 'feedback');
    await mkdir(dir, { recursive: true });
    const version = String(body.version || MASK_EDIT_VERSION);
     if (version !== MASK_EDIT_VERSION) return json(res, 400, { error: `只允许记录 ${MASK_EDIT_VERSION} 修正。` });
     const record = { feedbackId, id, angle, version, sourceMaskUrl: `/api/qc/files/processed-v2/${id}-mask.png`, correctedMaskUrl: `/api/qc/files/${version}/${id}-mask.png`, confirmed: body.confirmed !== false, createdAt: new Date().toISOString() };
    await writeFile(path.join(dir, `${feedbackId}.json`), JSON.stringify(record, null, 2), 'utf8');
    return json(res, 201, { ok: true, feedbackId, record });
  }

  const jobFileMatch = pathname.match(/^\/api\/jobs\/([\w.-]+)\/files\/(.+)$/);
  if (req.method === 'GET' && jobFileMatch) {
    const id = decodeURIComponent(jobFileMatch[1]);
    let rel;
    try { rel = decodeURIComponent(jobFileMatch[2]); } catch { return json(res, 400, { error: '文件路径编码无效。' }); }
    if (!/^[\w.-]+$/.test(id)) return json(res, 400, { error: '任务编号无效。' });
    const jobDir = path.join(JOBS_DIR, id);
    const abs = resolveInside(jobDir, rel);
    if (!abs) return json(res, 400, { error: '文件路径非法（疑似目录穿越）。' });
    if (!existsSync(abs)) return json(res, 404, { error: '产物文件不存在。' });
    return sendFile(req, res, abs, { download: url.searchParams.get('download') === '1' });
  }

  if (req.method === 'GET' && pathname === '/api/qc') {
    const requested = url.searchParams.get('version');
    if (requested) {
      if (!QC_VERSION_DEFS.some((v) => v.key === requested)) return json(res, 404, { error: '未知的 QC 版本。' });
      return json(res, 200, { ok: true, version: scanQcVersion(requested), overview: qcOverview() });
    }
    const overview = qcOverview();
    const preferred = scanQcVersion(overview.preferred);
    return json(res, 200, { ok: true, ...overview, detail: preferred });
  }

  const qcFileMatch = pathname.match(/^\/api\/qc\/files\/([\w.-]+)\/(.+)$/);
  if (req.method === 'GET' && qcFileMatch) {
    const version = qcFileMatch[1];
    if (!QC_VERSION_DEFS.some((v) => v.key === version)) return json(res, 404, { error: '未知的 QC 版本。' });
    let rel;
    try { rel = decodeURIComponent(qcFileMatch[2]); } catch { return json(res, 400, { error: '文件路径编码无效。' }); }
    const abs = resolveInside(path.join(TEST_IMAGES_DIR, version), rel);
    if (!abs) return json(res, 400, { error: '文件路径非法（疑似目录穿越）。' });
    if (!existsSync(abs)) return json(res, 404, { error: 'QC 文件不存在。' });
    return sendFile(req, res, abs, { download: url.searchParams.get('download') === '1' });
  }

  const qcSourceMatch = pathname.match(/^\/api\/qc\/source\/([^/]+)$/);
  if (req.method === 'GET' && qcSourceMatch) {
    let name;
    try { name = decodeURIComponent(qcSourceMatch[1]); } catch { return json(res, 400, { error: '文件名编码无效。' }); }
    if (!/^[\w.\-\u4e00-\u9fa5]+\.(png|jpe?g|webp)$/i.test(name)) return json(res, 400, { error: '仅允许读取 test-images 根目录的图片。' });
    const abs = path.join(TEST_IMAGES_DIR, name);
    if (!existsSync(abs)) return json(res, 404, { error: '素材原图不存在。' });
    return sendFile(req, res, abs);
  }

  /* ---- 手工修蒙版 + 一键重跑四视图 ---- */
  if (req.method === 'GET' && pathname === '/api/masks') {
    return json(res, 200, maskEditState());
  }

  if (req.method === 'POST' && pathname === '/api/masks/save') {
    if (maskSaveBusy) return json(res, 409, { error: '已有蒙版保存任务进行中，请稍候。' });
    let body;
    try { body = JSON.parse(await readBody(req)); } catch { return json(res, 400, { error: '请求格式无效。' }); }
    const version = String(body.version || MASK_EDIT_VERSION);
    if (version !== MASK_EDIT_VERSION) return json(res, 400, { error: `只允许写入 ${MASK_EDIT_VERSION}。` });
    const stem = String(body.id || '').replace(/\.[^.]+$/, '');
    if (!/^[0-9a-f]{32}$/i.test(stem)) return json(res, 400, { error: '素材编号无效。' });
    if (!existsSync(path.join(TEST_IMAGES_DIR, `${stem}.jpg`))) return json(res, 404, { error: '素材原图不存在。' });
    const match = String(body.mask || '').match(/^data:image\/png;base64,([\s\S]+)$/);
    if (!match) return json(res, 400, { error: '蒙版必须是 PNG dataURL。' });
    const bytes = decodeBase64Strict(match[1]);
    if (!bytes) return json(res, 400, { error: '蒙版 base64 内容无效。' });
     const dims = pngDimensions(bytes);
     if (!dims || dims.width < 2 || dims.height < 2 || dims.width * dims.height > 25_000_000) return json(res, 400, { error: '蒙版 PNG 无效或尺寸超限。' });
     if (bytes.length > 12 * 1024 * 1024) return json(res, 413, { error: '蒙版文件过大。' });
    maskSaveBusy = true;
    const dstDir = path.join(TEST_IMAGES_DIR, version);
    if (!existsSync(dstDir)) cpSync(path.join(TEST_IMAGES_DIR, 'processed-v2'), dstDir, { recursive: true });
    const tmp = path.join(GEN_ROOT, 'output', `_mask-edit-${randomUUID()}.png`);
    try {
      await writeFile(tmp, bytes);
      const { out } = await runPython('apply_mask_edit.py', [version, stem, tmp], { timeoutMs: 60000 });
      QC_CACHE.delete(version);
      console.log(out.trim());
      return json(res, 200, { ok: true, version, id: stem, state: maskEditState() });
    } catch (e) {
      return json(res, 500, { error: `保存蒙版失败：${e.message}` });
    } finally {
      await unlink(tmp).catch(() => {});
      maskSaveBusy = false;
    }
  }

  if (req.method === 'GET' && pathname === '/api/multiview') {
    return json(res, 200, { ok: true, run: refreshMultiviewStage(multiviewRun), state: maskEditState() });
  }

  if (req.method === 'POST' && pathname === '/api/multiview/run') {
    if (multiviewStarting || (multiviewRun && !multiviewRun.done)) return json(res, 409, { error: '已经有一个多视图任务在跑，请等它结束。' });
    let body;
    try { body = JSON.parse(await readBody(req)); } catch { return json(res, 400, { error: '请求格式无效。' }); }
    const maskVersion = String(body.maskVersion || MASK_EDIT_VERSION);
    if (!['processed-v3', 'processed-v2'].includes(maskVersion)) return json(res, 400, { error: '未知蒙版版本。' });
    if (!existsSync(path.join(TEST_IMAGES_DIR, maskVersion))) return json(res, 400, { error: `蒙版版本 ${maskVersion} 不存在。` });
    const octree = Math.min(384, Math.max(128, Number(body.octree) || 320));
    if (!existsSync(HY_PYTHON)) return json(res, 500, { error: `找不到 Hunyuan 环境：${HY_PYTHON}` });
    multiviewStarting = true;
    try {
      const run = await startMultiviewRun({ maskVersion, octree });
      return json(res, 202, { ok: true, run, jobUrl: '/api/jobs/hunyuan3d-2mv-v3' });
    } finally {
      multiviewStarting = false;
    }
  }

  /* ---- 原有创建任务接口（保留） ---- */
  if (req.method === 'POST' && pathname === '/api/local-models') {
    if (provider !== 'local-triposr') return json(res, 409, { error: '当前服务未选择本地 TripoSR。' });
    let body;
    try { body = JSON.parse(await readBody(req)); } catch (e) { return json(res, 400, { error: e.message || '请求格式无效。' }); }
    const m = String(body.image || '').match(/^data:(image\/(?:jpeg|png|webp));base64,([\s\S]+)$/i);
    if (!m) return json(res, 400, { error: '请上传 JPEG、PNG 或 WebP 图片。' });
    const bytes = Buffer.from(m[2], 'base64');
    if (bytes.length > 20 * 1024 * 1024) return json(res, 413, { error: '图片不能超过 20 MB。' });
    const sig = m[1] === 'image/png'
      ? bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
      : m[1] === 'image/jpeg'
        ? bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255
        : bytes.subarray(0, 4).toString() === 'RIFF' && bytes.subarray(8, 12).toString() === 'WEBP';
    if (!sig) return json(res, 400, { error: '图像内容与文件格式不匹配。' });
    const id = randomUUID();
    const ext = m[1] === 'image/png' ? '.png' : m[1] === 'image/webp' ? '.webp' : '.jpg';
    const input = path.join(GEN_FROM_SOURCE, 'input-temp', `rove-${id}${ext}`);
    const out = path.join(GEN_FROM_SOURCE, 'output', 'jobs', id);
    try {
      await mkdir(path.dirname(input), { recursive: true });
      await mkdir(path.join(out, '0'), { recursive: true });
      await writeFile(input, bytes);
    } catch (e) {
      return json(res, 500, { error: `无法写入任务目录：${e.message}` });
    }
    const task = { id, status: 'RUNNING', stage: 'queued', progress: null, createdAt: Date.now(), output: null, out, error: null };
    tasks.set(id, task);
    const args = [path.join(GEN_FROM_SOURCE, 'worker.py'), input, out];
    const consoleFile = path.join(out, 'console.log');
    let child;
    try {
      // 沙箱禁止管道，子进程输出直接写文件
      const fd = openSync(consoleFile, 'a');
      try {
        child = spawn(TRIPOSR_PYTHON, args, {
          cwd: TRIPOSR_SOURCE || GEN_FROM_SOURCE,
          windowsHide: true,
          env: {
            ...process.env,
            HF_HOME: process.env.HF_HOME || path.join(GEN_FROM_SOURCE, 'cache', 'huggingface'),
            U2NET_HOME: path.join(GEN_FROM_SOURCE, 'cache', 'u2net'),
          },
          stdio: ['ignore', fd, fd],
        });
      } finally {
        try { closeSync(fd); } catch { /* ignore */ }
      }
    } catch (e) {
      task.status = 'FAILED'; task.stage = 'failed'; task.error = `无法启动本地 Python 模型：${e.message}`;
      return json(res, 500, { error: task.error });
    }
    const readConsole = () => {
      try { return readFileSync(consoleFile, 'utf8'); } catch { return ''; }
    };
    child.on('error', async (e) => {
      await unlink(input).catch(() => {});
      task.status = 'FAILED'; task.stage = 'failed'; task.error = `无法启动本地 Python 模型：${e.message}`;
    });
    child.on('close', async (code) => {
      await unlink(input).catch(() => {});
      const text = readConsole();
      const stageMatches = [...text.matchAll(/ROVE_STAGE:(segment|prepare|geometry)/g)];
      if (stageMatches.length) task.stage = stageMatches[stageMatches.length - 1][1];
      if (code !== 0) {
        task.status = 'FAILED'; task.stage = 'failed';
        task.error = text.slice(-1200) || `TripoSR 进程退出（${code}）。`;
        return;
      }
      try {
        const glb = path.join(out, '0', 'mesh.glb');
        await stat(glb);
        task.output = glb; task.stage = 'complete'; task.status = 'SUCCEEDED';
        JOB_CACHE.delete(id);
      } catch {
        task.status = 'FAILED'; task.stage = 'failed'; task.error = '模型程序未产生 GLB 文件。';
      }
    });
    return json(res, 202, { id, status: 'RUNNING', provider: providerLabel, jobUrl: `/api/jobs/${id}` });
  }

  if (req.method === 'POST' && pathname === '/api/models') {
    if (!process.env.MESHY_API_KEY) return json(res, 503, { code: 'WAITING_CONFIGURATION', error: '生成服务尚未配置 API Key。请按本地配置说明设置 Meshy API Key。' });
    let body;
    try { body = JSON.parse(await readBody(req)); } catch (e) { return json(res, 400, { error: e.message || '请求格式无效。' }); }
    const m = String(body.image || '').match(/^data:(image\/(?:jpeg|png|webp));base64,([\s\S]+)$/i);
    if (!m) return json(res, 400, { error: '请上传 JPEG、PNG 或 WebP 图片。' });
    const bytes = Buffer.from(m[2], 'base64');
    if (bytes.length > 20 * 1024 * 1024) return json(res, 413, { error: '图片不能超过 20 MB。' });
    try {
      const upstream = await fetch('https://api.meshy.ai/openapi/v1/image-to-3d', {
        method: 'POST',
        headers: { Authorization: `Bearer ${process.env.MESHY_API_KEY}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: body.image, ai_model: 'latest', target_formats: ['glb'], should_texture: true, enable_pbr: true, pose_mode: 'a-pose' }),
      });
      const data = await upstream.json();
      if (!upstream.ok) return json(res, upstream.status, { error: data.message || data.error || `Meshy 返回 HTTP ${upstream.status}` });
      const id = randomUUID();
      tasks.set(id, { providerId: data.result, createdAt: Date.now(), status: 'PENDING' });
      return json(res, 202, { id, status: 'PENDING', provider: 'Meshy' });
    } catch {
      return json(res, 502, { error: '无法连接 Meshy 服务，请检查网络后重试。' });
    }
  }

  /* ---- 原有任务查询 / 预览 / 下载 / 清理（保留兼容） ---- */
  const preview = pathname.match(/^\/api\/models\/([\w-]+)\/preview\/(cutout|input|mask|human-mask|portrait-mask|fused-mask)$/);
  if (preview && req.method === 'GET' && provider === 'local-triposr') {
    const task = tasks.get(preview[1]);
    if (!task) return json(res, 404, { error: '本地任务记录不存在（服务重启后内存任务会清空，磁盘任务请用 /api/jobs）。' });
    const filename = {
      cutout: 'subject-cutout.png', input: 'model-input.png', mask: 'subject-mask.png',
      'human-mask': 'subject-mask-human.png', 'portrait-mask': 'subject-mask-portrait.png', 'fused-mask': 'subject-mask-fused.png',
    }[preview[2]];
    const file = path.join(task.out, filename);
    if (!existsSync(file)) return json(res, 404, { error: '该步骤尚未生成预览。' });
    return sendFile(req, res, file);
  }

  const modelMatch = pathname.match(/^\/api\/models\/([\w-]+)(?:\/(download))?$/);
  if (modelMatch && req.method === 'GET') {
    const task = tasks.get(modelMatch[1]);
    if (!task) return json(res, 404, { error: '本地任务记录不存在；服务重启后任务记录会清空，历史任务请访问 /api/jobs。' });
    if (provider === 'local-triposr') {
      if (modelMatch[2]) {
        if (task.status !== 'SUCCEEDED') return json(res, 409, { error: '本地模型尚未生成完成。' });
        return sendFile(req, res, task.output, { download: true });
      }
      return json(res, 200, {
        id: task.id,
        provider: providerLabel,
        status: task.status,
        stage: task.stage,
        elapsedSeconds: Math.round((Date.now() - task.createdAt) / 1000),
        error: task.error,
        jobUrl: `/api/jobs/${task.id}`,
        cutoutUrl: `/api/models/${task.id}/preview/cutout`,
        maskUrl: `/api/models/${task.id}/preview/mask`,
        humanMaskUrl: `/api/models/${task.id}/preview/human-mask`,
        portraitMaskUrl: `/api/models/${task.id}/preview/portrait-mask`,
        fusedMaskUrl: `/api/models/${task.id}/preview/fused-mask`,
        inputUrl: `/api/models/${task.id}/preview/input`,
        meshUrl: task.status === 'SUCCEEDED' ? `/api/models/${task.id}/download` : null,
        downloadUrl: task.status === 'SUCCEEDED' ? `/api/models/${task.id}/download` : null,
      });
    }
    if (!process.env.MESHY_API_KEY) return json(res, 503, { error: 'Meshy API Key 未配置。' });
    try {
      const upstream = await fetch(`https://api.meshy.ai/openapi/v1/image-to-3d/${encodeURIComponent(task.providerId)}`, { headers: { Authorization: `Bearer ${process.env.MESHY_API_KEY}` } });
      const data = await upstream.json();
      if (!upstream.ok) return json(res, upstream.status, { error: data.message || `Meshy 返回 HTTP ${upstream.status}` });
      task.status = data.status;
      if (modelMatch[2]) {
        if (data.status !== 'SUCCEEDED' || !data.model_urls?.glb) return json(res, 409, { error: '模型尚未生成完成。' });
        const asset = new URL(data.model_urls.glb);
        if (asset.protocol !== 'https:' || !asset.hostname.endsWith('meshy.ai')) return json(res, 502, { error: '服务商返回的模型地址未通过安全检查。' });
        const file = await fetch(asset);
        if (!file.ok) return json(res, 502, { error: '模型下载暂时失败，请稍后重试。' });
        res.writeHead(200, { 'content-type': 'model/gltf-binary', 'content-disposition': 'attachment; filename="rove-character.glb"', 'cache-control': 'no-store' });
        return res.end(Buffer.from(await file.arrayBuffer()));
      }
      return json(res, 200, { id: modelMatch[1], provider: 'Meshy', status: data.status, progress: data.progress ?? null, error: data.task_error?.message || null, downloadUrl: data.status === 'SUCCEEDED' ? `/api/models/${modelMatch[1]}/download` : null });
    } catch {
      return json(res, 502, { error: '查询 Meshy 任务失败，请稍后继续查询原任务。' });
    }
  }

  if (modelMatch && req.method === 'DELETE') {
    const task = tasks.get(modelMatch[1]);
    if (!task) return json(res, 404, { error: '只能删除本进程内创建的任务；磁盘历史任务不会被删除。' });
    if (task.status === 'RUNNING') return json(res, 409, { error: '任务仍在运行，完成后再删除。' });
    if (provider === 'local-triposr' && task.out) {
      const safe = resolveInside(JOBS_DIR, path.relative(JOBS_DIR, task.out));
      if (safe && existsSync(safe)) await rm(safe, { recursive: true, force: true });
    }
    tasks.delete(modelMatch[1]);
    JOB_CACHE.delete(modelMatch[1]);
    return json(res, 200, { deleted: true });
  }

  return json(res, 404, { error: '未找到该接口。' });
}

const server = http.createServer((req, res) => {
  handle(req, res).catch((err) => {
    if (res.headersSent) { try { res.destroy(); } catch { /* ignore */ } return; }
    json(res, 500, { error: `服务内部错误：${err?.message || err}` });
  });
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`端口 ${PORT} 已被占用。请修改 .env 中的 PORT，或在启动前关闭占用该端口的进程。`);
    process.exit(1);
  }
  throw err;
});

server.listen(PORT, HOST, () => {
  console.log(`Rove Model Studio 已启动：http://localhost:${PORT}/`);
  console.log(`任务产物目录：${JOBS_DIR}`);
  console.log(`页面文件：${PAGE_FILE}`);
});
