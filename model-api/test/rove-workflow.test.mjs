const base = process.env.ROVE_BASE_URL || 'http://127.0.0.1:8790';

const feedback = await fetch(`${base}/api/feedback`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    id: '9ca2f019a28896cbef7fc86359863ddc',
    angle: 'front',
    version: 'processed-v3',
    sourceMaskUrl: '/api/qc/files/processed-v2/9ca2f019a28896cbef7fc86359863ddc-mask.png',
    correctedMaskUrl: '/api/qc/files/processed-v3/9ca2f019a28896cbef7fc86359863ddc-mask.png',
    confirmed: true,
  }),
});
if (feedback.status !== 201) throw new Error(`feedback expected 201, got ${feedback.status}`);
const feedbackData = await feedback.json();
if (!feedbackData.ok || !/^[0-9a-f-]{36}$/.test(feedbackData.feedbackId)) throw new Error('invalid feedback result');

const comparison = await fetch(`${base}/api/comparison/hunyuan3d-2mv-v3`);
if (comparison.status !== 200) throw new Error(`comparison expected 200, got ${comparison.status}`);
const data = await comparison.json();
if (!data.ok || !Array.isArray(data.models) || !data.contactSheetUrl) throw new Error('invalid comparison manifest');
const sheet = await fetch(`${base}${data.contactSheetUrl}`);
if (sheet.status !== 200 || !(sheet.headers.get('content-type') || '').includes('image/png')) throw new Error('contact sheet is not served as PNG');
console.log('workflow tests passed');
