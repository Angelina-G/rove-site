import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const chrome = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
].find(fs.existsSync);
assert.ok(chrome, 'Chrome or Edge is required for the smoke check');

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'rove-smoke-'));
const pageUrl = pathToFileURL(path.join(process.cwd(), '01_网页报告', 'Rove_体系交互与研究路线.html')).href;
const browser = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--no-first-run', '--disable-extensions',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, pageUrl,
], { stdio: 'ignore' });

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket;
try {
  const portFile = path.join(profile, 'DevToolsActivePort');
  for (let i = 0; i < 50 && !fs.existsSync(portFile); i++) await wait(100);
  assert.ok(fs.existsSync(portFile), 'browser debugging port did not open');
  const port = Number(fs.readFileSync(portFile, 'utf8').split('\n')[0]);
  const pages = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  const page = pages.find(item => item.url === pageUrl);
  assert.ok(page, 'Rove page did not open');

  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let serial = 0;
  const pending = new Map();
  socket.onmessage = event => {
    const reply = JSON.parse(event.data);
    if (!pending.has(reply.id)) return;
    const { resolve, reject } = pending.get(reply.id);
    pending.delete(reply.id);
    reply.error ? reject(new Error(reply.error.message)) : resolve(reply.result);
  };
  const command = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++serial;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await command('Runtime.evaluate', { expression, returnByValue: true });
    assert.ok(!result.exceptionDetails, result.exceptionDetails?.text);
    return result.result.value;
  };

  let paperCount = 0;
  for (let i = 0; i < 50 && paperCount !== 100; i++) {
    paperCount = await evaluate('document.querySelectorAll(".paper").length');
    if (paperCount !== 100) await wait(100);
  }
  assert.equal(paperCount, 100);
  assert.equal(await evaluate('document.querySelectorAll("#journeyRail button").length'), 8);
  assert.equal(await evaluate('document.querySelector("[data-room-focus=guest]").click(); document.querySelector("[data-room-label=host]").textContent'), 'B′ · 主场 AR · 绿色空间');
  assert.equal(await evaluate('getComputedStyle(document.querySelector("[data-room=host] .room-stage")).backgroundImage === getComputedStyle(document.querySelector("[data-room=guest] .room-stage")).backgroundImage && getComputedStyle(document.querySelector("[data-room=host] .room-stage")).backgroundImage.includes("48, 78, 73")'), true);
  assert.equal(await evaluate('document.querySelector("[data-room-focus=host]").click(); document.querySelector("[data-room-label=host]").textContent'), 'A′ · 主场 AR · 红色空间');
  assert.equal(await evaluate('getComputedStyle(document.querySelector("[data-room=host] .room-stage")).backgroundImage === getComputedStyle(document.querySelector("[data-room=guest] .room-stage")).backgroundImage && getComputedStyle(document.querySelector("[data-room=host] .room-stage")).backgroundImage.includes("87, 62, 58")'), true);
  assert.equal(await evaluate('document.querySelector("#consentQuestion").textContent'), 'A 邀请 B 进入 A′ 红色房间');
  assert.equal(await evaluate('getComputedStyle(document.querySelector("[data-room=host] .person-b")).visibility'), 'hidden');
  assert.equal(await evaluate('document.querySelector("[data-consent=yes]").click(); document.querySelector("#consentStatus").textContent'), 'B 已同意进入');
  assert.equal(await evaluate('getComputedStyle(document.querySelector("[data-room=host] .person-b")).visibility'), 'visible');
  assert.equal(await evaluate('document.querySelector("[data-room-focus=guest]").click(); document.querySelector("#consentQuestion").textContent'), 'B 邀请 A 进入 B′ 绿色房间');
  assert.equal(await evaluate('document.querySelector("#consentStatus").textContent'), '等待 A 确认');
  assert.equal(await evaluate('getComputedStyle(document.querySelector("[data-room=guest] .person-b")).visibility'), 'hidden');
  assert.equal(await evaluate('document.querySelector("[data-consent=no]").click(); document.querySelector("#consentStatus").textContent'), 'A 暂不进入');
  assert.equal(await evaluate('document.querySelector("[data-room=guest] .person-b").closest("#roomPair").classList.contains("consent-accepted")'), false);
  await evaluate('document.querySelector("[data-room-focus=host]").click()');
  assert.equal(await evaluate('document.querySelectorAll(".room-person.person-a").length'), 2);
  assert.equal(await evaluate('document.querySelectorAll(".room-person.person-b").length'), 2);
  assert.equal(await evaluate('document.querySelectorAll(".flow-map svg rect").length'), 6);
  assert.equal(fs.existsSync(path.join(process.cwd(), '01_网页报告', 'rove_future_concept.png')), true);
  assert.equal(await evaluate('document.querySelectorAll(".concept-figure img").length === 1 && document.querySelector(".concept-figure img").naturalWidth > 0'), true);
  assert.equal(await evaluate('document.querySelector("[data-scenario=memory]").click(); document.querySelector("#journeyMain h3").textContent'), '共同决定记录');
  assert.equal(await evaluate('document.querySelector("[data-filter=社交触觉与关系]").click(); document.querySelectorAll(".paper").length > 5'), true);
  assert.equal(await evaluate('document.querySelector("[data-phase=\'3\']").click(); document.querySelector("#roadmapBody").textContent.includes("触碰请求、可触区域和急停")'), true);
  assert.equal(await evaluate('document.querySelectorAll("#roadmapHeader button").length'), 6);
  assert.equal(await evaluate('document.querySelector("[data-layer=\'5\']").click(); document.querySelector("#archDetail h3").textContent'), '让亲密互动始终由两个人决定');
  await command('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth'), true, 'horizontal overflow at 390px');
  console.log('Rove browser smoke check passed: content, interactions, mobile width.');
} finally {
  socket?.close();
  browser.kill();
  // shortcut: temporary browser profile only; retry cleanup if Windows still holds a file.
  if (path.resolve(profile).startsWith(path.resolve(os.tmpdir()) + path.sep) && path.basename(profile).startsWith('rove-smoke-')) {
    for (let i = 0; i < 5; i++) {
      try { fs.rmSync(profile, { recursive: true, force: true }); break; }
      catch { await wait(200); }
    }
  }
}
