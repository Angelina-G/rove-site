const base = process.env.ROVE_BASE_URL || 'http://127.0.0.1:8790';
const badMask = await fetch(`${base}/api/masks/save`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ id: '9ca2f019a28896cbef7fc86359863ddc', version: 'processed-v3', mask: 'data:image/png;base64,!!!!' }) });
if (badMask.status !== 400) throw new Error(`invalid PNG mask expected 400, got ${badMask.status}`);

const feedback = await fetch(`${base}/api/feedback`, {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ id: '9ca2f019a28896cbef7fc86359863ddc', angle: 'front', version: 'processed-v3', confirmed: true }),
});
if (feedback.status !== 201) throw new Error(`feedback expected 201, got ${feedback.status}`);
const comparison = await fetch(`${base}/api/comparison/hunyuan3d-2mv-v3`);
if (comparison.status !== 200) throw new Error(`comparison expected 200, got ${comparison.status}`);
const data = await comparison.json();
if (!data.ok || !Array.isArray(data.models) || data.models.length < 1 || !data.contactSheetUrl) throw new Error('invalid or empty comparison manifest');
if (data.models.some((m) => !m.previewUrl || !m.previewMtimeNs)) throw new Error('comparison model lacks preview provenance');
const sheet = await fetch(`${base}${data.contactSheetUrl}`);
if (sheet.status !== 200 || !(sheet.headers.get('content-type') || '').includes('image/png')) throw new Error('contact sheet is not served as PNG');
console.log('workflow checks passed');
