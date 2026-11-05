const visionVersion = '0.10.21';
const visionCdnRoots = [
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${visionVersion}`,
  `https://fastly.jsdelivr.net/npm/@mediapipe/tasks-vision@${visionVersion}`,
  `https://registry.npmmirror.com/@mediapipe/tasks-vision/${visionVersion}/files`,
  `https://unpkg.com/@mediapipe/tasks-vision@${visionVersion}`
];
let visionTasks, visionCdn;
const importErrors = [];
const importWithTimeout = async (url) => {
  let timer;
  try {
    return await Promise.race([
      import(url),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('加载超时（10 秒）')), 10000); })
    ]);
  } finally { clearTimeout(timer); }
};
for (const root of visionCdnRoots) {
  try { visionTasks = await importWithTimeout(`${root}/vision_bundle.mjs`); visionCdn = root; break; }
  catch (error) { importErrors.push(`${new URL(root).host}: ${error.message}`); }
}
if (!visionTasks) {
  const message = `追踪运行库下载失败（${importErrors.join('；')}）`;
  window.__roveTrackerModuleError = message;
  const status = document.querySelector('#camera-status');
  const badge = document.querySelector('#tracking-badge');
  if (status) status.textContent = `${message}。请切换网络后刷新，或让管理员把运行库放到本地。`;
  if (badge) badge.textContent = '! 追踪运行库下载失败';
  throw new Error(message);
}
window.__roveTrackerModuleSource = new URL(visionCdn).host;
const { FilesetResolver, PoseLandmarker, HandLandmarker, FaceLandmarker } = visionTasks;
window.__roveTrackerModuleLoaded = true;

const video = document.querySelector('#camera');
const canvas = document.querySelector('#tracking-overlay');
const ctx = canvas.getContext('2d');
const status = document.querySelector('#camera-status');
const poseState = document.querySelector('#pose-state');
const handsState = document.querySelector('#hands-state');
const fpsState = document.querySelector('#fps-state');
const latencyState = document.querySelector('#latency-state');
const trackingBadge = document.querySelector('#tracking-badge');
let avatarFaceMorphCount = window.__roveAvatarCapabilities?.faceMorphCount??0;
window.addEventListener('rove-avatar-capabilities',event=>{avatarFaceMorphCount=event.detail?.faceMorphCount??0;});
let poseTask, handTask, faceTask, running = false, lastVideoTime = -1, frameCount = 0, fpsStart = 0;
let filteredPose = null, poseMisses = Array(33).fill(0), filteredFace = new Map(), filteredHands = new Map(), filteredHandWorlds = new Map();
const moduleErrors = new Map();
const modelBuffers = new Map();
let lowerBodyMessage = '站到画面中央，确保头顶和双脚都入镜', legOverlapFrames = 0;
const legThresholdControl=document.querySelector('#leg-threshold');
const savedLegThreshold=Number(localStorage.getItem('rove-leg-threshold')??legThresholdControl.value);
let legThreshold=(Number.isFinite(savedLegThreshold)?Math.min(60,Math.max(25,savedLegThreshold)):32)/100;
legThresholdControl.value=String(Math.round(legThreshold*100));
document.querySelector('#leg-threshold-value').textContent=`${Math.round(legThreshold*100)}%`;
legThresholdControl.addEventListener('input',()=>{
  legThreshold=Number(legThresholdControl.value)/100;
  document.querySelector('#leg-threshold-value').textContent=`${legThresholdControl.value}%`;
  localStorage.setItem('rove-leg-threshold',legThresholdControl.value);
});

const poseConnections = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[28,30],[30,32],[15,17],[15,19],[15,21],[16,18],[16,20],[16,22]];
const handConnections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

async function setup() {
  if (poseTask && handTask && faceTask) return;
  status.textContent = '正在加载身体、手部与面部模型…首次加载需要网络';
  trackingBadge.textContent = '◌ 正在加载追踪模型';
  let vision, wasmError;
  const wasmRoots = [...new Set([visionCdn, ...visionCdnRoots])];
  for (const root of wasmRoots) {
    try { vision = await FilesetResolver.forVisionTasks(`${root}/wasm`); break; }
    catch (error) { wasmError = `${new URL(root).host}: ${error.message}`; }
  }
  if (!vision) throw new Error(`MediaPipe WASM 运行文件下载失败（${wasmError}）`);
  const fetchModel = async (url) => {
    if (modelBuffers.has(url)) return modelBuffers.get(url);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(url, { signal: controller.signal, cache: 'force-cache' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const buffer = new Uint8Array(await response.arrayBuffer());
      modelBuffers.set(url, buffer);
      return buffer;
    } finally { clearTimeout(timeout); }
  };
  const makeTask = async (name, Task, modelAssetPaths, options) => {
    const errors = [];
    for (const modelAssetPath of modelAssetPaths) {
      let modelAssetBuffer;
      try { modelAssetBuffer = await fetchModel(modelAssetPath); }
      catch (error) { errors.push(`${new URL(modelAssetPath, location.href).host || '本地'} 下载超时/失败: ${error.message}`); continue; }
      for (const delegate of ['GPU', 'CPU']) {
        try {
          const task = await Task.createFromOptions(vision, { ...options, baseOptions: { modelAssetBuffer, delegate } });
          return { task, error: '' };
        } catch (error) { errors.push(`${new URL(modelAssetPath, location.href).host || '本地'} 模型 / ${delegate}: ${error.message}`); }
      }
    }
    const error = `${name}模型下载或初始化失败（${errors.join('；')}）`;
    moduleErrors.set(name, error);
    return { task: null, error };
  };
  const asset = (file, officialUrl, mirrorUrl) => [
    new URL(`./models/vision/${file}`, location.href).href,
    officialUrl,
    ...(mirrorUrl ? [mirrorUrl] : [])
  ];
  const [poseResult, handResult, faceResult] = await Promise.all([
    poseTask ? Promise.resolve({ task: poseTask }) : makeTask('身体', PoseLandmarker, asset('pose_landmarker_lite.task', 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'), { runningMode: 'VIDEO', numPoses: 1 }),
    handTask ? Promise.resolve({ task: handTask }) : makeTask('手部', HandLandmarker, asset('hand_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/hand_landmarker.task'), { runningMode: 'VIDEO', numHands: 2, minHandDetectionConfidence: .4, minHandPresenceConfidence: .4, minTrackingConfidence: .45 }),
    faceTask ? Promise.resolve({ task: faceTask }) : makeTask('面部', FaceLandmarker, asset('face_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/face_landmarker.task'), { runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true })
  ]);
  poseTask = poseResult.task; handTask = handResult.task; faceTask = faceResult.task;
  if (!poseTask && !handTask && !faceTask) throw new Error([...moduleErrors.values()].join('；'));
}

function line(a, b, color, width = 3) {
  const pa=canvasPoint(a),pb=canvasPoint(b);
  ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y);
  ctx.strokeStyle = color; ctx.lineWidth = width * devicePixelRatio; ctx.lineCap = 'round'; ctx.stroke();
}
function point(p, color, radius = 4) {
  const mapped=canvasPoint(p);
  ctx.beginPath(); ctx.arc(mapped.x,mapped.y,radius * devicePixelRatio,0,Math.PI * 2); ctx.fillStyle = color; ctx.fill();
}
function canvasPoint(p) {
  if(!video.videoWidth||!video.videoHeight)return {x:p.x*canvas.width,y:p.y*canvas.height};
  const scale=Math.max(canvas.width/video.videoWidth,canvas.height/video.videoHeight);
  const width=video.videoWidth*scale,height=video.videoHeight*scale;
  return {x:(canvas.width-width)/2+p.x*width,y:(canvas.height-height)/2+p.y*height};
}
function drawLinks(points, links, color, label) {
  for (const [a, b] of links) if (points[a] && points[b] && pointVisible(points,a) && pointVisible(points,b)) line(points[a], points[b], color);
  for (let i=0;i<points.length;i++) if (pointVisible(points,i)) point(points[i], color, 3);
  const anchor = points[0];
  if (anchor) { const mapped=canvasPoint(anchor); ctx.font = `${11 * devicePixelRatio}px sans-serif`; ctx.fillStyle = color; ctx.fillText(label,mapped.x+8*devicePixelRatio,mapped.y-8*devicePixelRatio); }
}

function conf(p) { return Math.min(p?.visibility ?? 1, p?.presence ?? 1); }
function pointVisible(points,index) { return conf(points[index]) >= (points.length===33&&index>=23?legThreshold:.35); }
function stabilizePose(input) {
  if (!input) return null;
  const result = input.map((p,i) => {
    const threshold = i >= 23 ? legThreshold : .38;
    const previous = filteredPose?.[i];
    if (conf(p) < threshold) {
      poseMisses[i]++;
      return previous && conf(previous) >= threshold && poseMisses[i] <= 4 ? {...previous, visibility: Math.max(threshold,.4), presence:Math.max(threshold,.4)} : {...p, visibility: 0};
    }
    if (!previous) { poseMisses[i]=0; return {...p,visibility:conf(p)}; }
    const alpha = i>=23 ? .18 : .3;
    const jump = Math.hypot(p.x - previous.x, p.y - previous.y);
    if (jump > (i >= 23 ? .22 : .28)) {
      poseMisses[i]++;
      if (poseMisses[i] <= 3) return {...previous, visibility: Math.max(threshold,.4), presence:Math.max(threshold,.4)};
    }
    poseMisses[i] = 0;
    return {...p, visibility:conf(p), x:previous.x+(p.x-previous.x)*alpha, y:previous.y+(p.y-previous.y)*alpha, z:previous.z+(p.z-previous.z)*alpha};
  });
  // Reject implausible segments and suppress the less reliable leg when both legs collapse to the same image points.
  for (const [hip,knee,ankle] of [[23,25,27],[24,26,28]]) {
    const upper=Math.hypot(result[hip].x-result[knee].x,result[hip].y-result[knee].y);
    const lower=Math.hypot(result[knee].x-result[ankle].x,result[knee].y-result[ankle].y);
    if (result[knee].visibility !== .2 && result[ankle].visibility !== .2 && (upper < .035 || lower < .035 || upper > .62 || lower > .62)) {
      result[knee].visibility=0; result[ankle].visibility=0;
    }
  }
  const visible = i => pointVisible(result,i);
  const hipWidth = Math.hypot(result[23].x-result[24].x,result[23].y-result[24].y);
  const kneeGap = Math.hypot(result[25].x-result[26].x,result[25].y-result[26].y);
  const ankleGap = Math.hypot(result[27].x-result[28].x,result[27].y-result[28].y);
  const overlapCandidate = [23,24,25,26,27,28].every(visible) && hipWidth > .025 && kneeGap < Math.max(.008,hipWidth*.1) && ankleGap < Math.max(.01,hipWidth*.12);
  legOverlapFrames=overlapCandidate?legOverlapFrames+1:0;
  if (overlapCandidate&&legOverlapFrames>=3) {
    const leftScore = [23,25,27].reduce((sum,i)=>sum+conf(result[i]),0);
    const rightScore = [24,26,28].reduce((sum,i)=>sum+conf(result[i]),0);
    if(Math.abs(leftScore-rightScore)>.24){
      const hide = leftScore > rightScore ? [26,28,30,32] : [25,27,29,31];
      for (const i of hide) result[i].visibility = 0;
      lowerBodyMessage = '腿部点几乎完全重合：已隐藏置信度较低的一侧，请把双脚分开';
    } else lowerBodyMessage = '双腿距离很近：暂时保留两侧追踪，请把双脚分开一点';
  } else if(overlapCandidate) {
    lowerBodyMessage = '双腿关键点接近，正在确认是否重叠';
  } else if (![27,28].every(visible)) {
    lowerBodyMessage = '脚踝没有稳定入镜：后退一点并把完整双脚放进画面';
  } else if (![25,26].every(visible)) {
    lowerBodyMessage = '膝盖被遮挡或置信度较低：请避免交叉腿，稍微分开双腿';
  } else {
    lowerBodyMessage = '双腿清晰 · 脚踝可见';
  }
  filteredPose=result;
  return result;
}
function handSide(hand, pose, fallback = '') {
  if (hand?.[0] && pose?.[15] && pose?.[16]) {
    const left = Math.hypot(hand[0].x-pose[15].x,hand[0].y-pose[15].y);
    const right = Math.hypot(hand[0].x-pose[16].x,hand[0].y-pose[16].y);
    if (Math.abs(left-right) > .015) return left < right ? 'left' : 'right';
  }
  return fallback.toLowerCase();
}
function stabilizeHands(hands, handedness, pose) {
  const next=new Map();
  const stable=hands.map((hand,index)=>{
    const key=handSide(hand,pose,handedness?.[index]?.[0]?.categoryName??String(index)),previous=filteredHands.get(key);
    const points=hand.map((p,i)=>{
      const confidence=Math.min(p.visibility??1,p.presence??1),old=previous?.[i];
      return {...p,visibility:confidence,...(old&&confidence>.25?{x:old.x+(p.x-old.x)*.55,y:old.y+(p.y-old.y)*.55,z:(old.z??0)+((p.z??0)-(old.z??0))*.55}:{})};
    });
    next.set(key,points);return points;
  });
  filteredHands=next;
  return stable;
}
function stabilizeHandWorlds(hands, worldHands, handedness, pose) {
  const next=new Map();
  const stable=worldHands.map((hand,index)=>{
    const key=handSide(hands[index],pose,handedness?.[index]?.[0]?.categoryName??String(index)),previous=filteredHandWorlds.get(key);
    const points=hand.map((p,i)=>{const old=previous?.[i];return old?{x:old.x+(p.x-old.x)*.46,y:old.y+(p.y-old.y)*.46,z:old.z+(p.z-old.z)*.46}:p});
    next.set(key,points);return points;
  });
  filteredHandWorlds=next;
  return stable;
}
function stabilizeFace(categories) {
  const next=new Map();
  for(const item of categories){const old=filteredFace.get(item.categoryName)??item.score;next.set(item.categoryName,old*.55+item.score*.45)}
  filteredFace=next;
  return categories.map(item=>({...item,score:next.get(item.categoryName)}));
}

function detectModule(name, task, video, timestamp) {
  try {
    const result = task.detectForVideo(video, timestamp);
    moduleErrors.delete(name);
    return result;
  } catch (error) {
    moduleErrors.set(name, error?.message ?? String(error));
    return null;
  }
}

function tick() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    const start = performance.now();
    canvas.width = Math.round(video.clientWidth * devicePixelRatio);
    canvas.height = Math.round(video.clientHeight * devicePixelRatio);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    try {
      const t = performance.now();
      const pose = poseTask ? detectModule('身体', poseTask, video, t) : {landmarks:[],worldLandmarks:[]};
      const hands = handTask ? detectModule('手部', handTask, video, t) : {landmarks:[],worldLandmarks:[],handedness:[]};
      const face = faceTask ? detectModule('面部', faceTask, video, t) : {faceBlendshapes:[],faceLandmarks:[]};
      const stablePose=stabilizePose(pose.landmarks?.[0]??null);
      if (stablePose) {
        drawLinks(stablePose, poseConnections, '#c5e36d', '身体骨架');
        const visiblePoints=stablePose.filter((p,i)=>pointVisible(stablePose,i)).length;
        poseState.textContent = `${visiblePoints}/33 点可见`;
      }
      else poseState.textContent = '未检测到';
      if (hands.landmarks?.length) {
        const stableHands=stabilizeHands(hands.landmarks,hands.handedness,stablePose);
        hands.worldLandmarks=stabilizeHandWorlds(stableHands,hands.worldLandmarks??[],hands.handedness,stablePose);
        const handSummaries=[];
        stableHands.forEach((points, index) => {
          const side=handSide(points,stablePose,hands.handedness?.[index]?.[0]?.categoryName??'');
          const name=side?`${side==='left'?'左':'右'}手`:`手${index+1}`;
          const tips=[4,8,12,16,20].filter(i=>(points[i]?.visibility??1)>.45).length;
          drawLinks(points, handConnections, '#ff9b76', `${name} · 21点`);
          handSummaries.push(`${name} · ${tips}/5指尖`);
        });
        handsState.textContent = handSummaries.join(' / ');
        hands.landmarks=stableHands;
      } else { handsState.textContent = '未检测到'; hands.landmarks=[]; }
      const bodyHint = document.querySelector('#body-hint');
      if (bodyHint) {
        bodyHint.textContent = stablePose ? lowerBodyMessage : '没有稳定识别到全身：请站远一点，并让头顶、膝盖和双脚完整入镜';
        bodyHint.classList.toggle('good', !!stablePose && lowerBodyMessage === '双腿清晰 · 脚踝可见');
        bodyHint.classList.toggle('warn', !stablePose || lowerBodyMessage !== '双腿清晰 · 脚踝可见');
      }
      const feetOk = stablePose && [27,28].every(i => pointVisible(stablePose,i));
      const legMetric = document.querySelector('#legs-state');
      if (legMetric) legMetric.textContent = !stablePose ? '未识别' : lowerBodyMessage.includes('正在确认') ? '确认中' : lowerBodyMessage.includes('重合') || lowerBodyMessage.includes('距离很近') ? '腿点重叠' : feetOk ? '清晰' : '不完整';
      const faceCategories = stabilizeFace(face.faceBlendshapes?.[0]?.categories ?? []);
      document.querySelector('#face-state').textContent = faceCategories.length ? '已追踪' : '未检测到';
      if (face.faceLandmarks?.[0]) { const marks=face.faceLandmarks[0]; for (let i=0;i<marks.length;i+=3) point(marks[i],'#75d5cf',2.2); }
      renderExpressions(faceCategories);
      window.dispatchEvent(new CustomEvent('rove-motion', { detail: { pose: stablePose, worldPose: pose.worldLandmarks?.[0] ?? [], hands: hands.landmarks ?? [], handsWorld: hands.worldLandmarks ?? [], handedness: hands.handedness?.map(item => item[0]?.categoryName ?? '') ?? [], face: faceCategories, faceLandmarks: face.faceLandmarks?.[0] ?? [], legThreshold } }));
      const found = [stablePose && '身体', hands.landmarks?.length && '手指', faceCategories.length && '表情'].filter(Boolean);
      trackingBadge.textContent = found.length ? `● 正在追踪：${found.join(' · ')}` : '○ 模型运行中 · 暂未检测到人体';
      if (moduleErrors.size) status.textContent = `部分追踪模块暂不可用：${[...moduleErrors].map(([name,error])=>`${name}（${error}）`).join('；')}`;
      else if (running) status.textContent = `模型已加载 · ${found.length ? `当前检测到${found.join('、')}` : '正在等待人物进入画面'}`;
      latencyState.textContent = `${Math.round(performance.now() - start)} ms`;
      frameCount++;
      if (!fpsStart) fpsStart = performance.now();
      if (performance.now() - fpsStart >= 1000) { fpsState.textContent = `${frameCount} fps`; frameCount = 0; fpsStart = performance.now(); }
    } catch (error) { status.textContent = `追踪错误：${error.message}`; stopTracking(); return; }
    lastVideoTime = video.currentTime;
  }
  requestAnimationFrame(tick);
}

function stopTracking() {
  running = false; ctx.clearRect(0, 0, canvas.width, canvas.height);
  filteredPose=null; poseMisses.fill(0); filteredFace=new Map(); filteredHands=new Map(); filteredHandWorlds=new Map(); legOverlapFrames=0;
  trackingBadge.textContent = '● 追踪已关闭';
  poseState.textContent = handsState.textContent = document.querySelector('#face-state').textContent = document.querySelector('#legs-state').textContent = '等待'; fpsState.textContent = latencyState.textContent = '—';
  document.querySelector('#body-hint').textContent = '等待摄像头追踪后，这里会告诉你目前识别到的部位或遮挡原因。';
  renderExpressions([]);
  window.dispatchEvent(new CustomEvent('rove-motion', { detail: { pose: null, hands: [], face: [] } }));
}

function renderExpressions(categories) {
  const panel = document.querySelector('#expression-bars');
  const values = new Map(categories.map(item => [item.categoryName, item.score]));
  const signals = {
    blink:Math.max(values.get('eyeBlinkLeft')??0,values.get('eyeBlinkRight')??0),
    jaw:values.get('jawOpen')??0,
    smile:((values.get('mouthSmileLeft')??0)+(values.get('mouthSmileRight')??0))/2,
    frown:((values.get('mouthFrownLeft')??0)+(values.get('mouthFrownRight')??0))/2,
    brow:values.get('browInnerUp')??0
  };
  panel.querySelector('.expression-title').textContent=categories.length?(avatarFaceMorphCount?`面部信号 · 已连接角色（${avatarFaceMorphCount} 个形态键）`:'面部信号 · 当前角色没有可映射形态键'):'面部信号 · 尚未检测到面部';
  panel.querySelectorAll('.expression-row').forEach(row=>{
    const value=signals[row.dataset.expression]??0;
    row.querySelector('i').style.width=`${Math.round(value*100)}%`;
    row.querySelector('b').textContent=value.toFixed(2);
  });
}

async function startTrackingWhenCameraReady() {
  for (let i = 0; i < 150 && !video.srcObject; i++) await new Promise(resolve => setTimeout(resolve, 200));
  if (!video.srcObject || running) return;
  try { await setup(); running = true; fpsStart = 0; frameCount = 0; status.textContent = '模型已加载，正在分析摄像头画面'; trackingBadge.textContent = '◌ 模型已加载 · 正在识别'; tick(); }
  catch (error) { status.textContent = `追踪启动失败：${error.message}`; trackingBadge.textContent = '! 追踪启动失败'; }
}
document.querySelector('#camera-btn').addEventListener('click', startTrackingWhenCameraReady);
// The camera permission handler can run before this module finishes loading from the CDN.
if (video.srcObject) startTrackingWhenCameraReady();
document.querySelector('#stop-btn').addEventListener('click', stopTracking);
window.addEventListener('pagehide', stopTracking);
