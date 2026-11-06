import * as THREE from 'three';
import { GLTFLoader } from './node_modules/three/examples/jsm/loaders/GLTFLoader.js';

const canvas = document.querySelector('#avatar-canvas');
const empty = document.querySelector('#empty');
const modelView = document.querySelector('#model');
const state = document.querySelector('#model-state');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(26, 1, .05, 50);
camera.position.set(0, .86, 4.0);
camera.lookAt(0, .86, 0);
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
scene.add(new THREE.HemisphereLight(0xeaf4ff, 0x685b53, 2.0));
const key = new THREE.DirectionalLight(0xffe4cb, 3.2); key.position.set(-2.5, 4, 4); scene.add(key);
const rim = new THREE.DirectionalLight(0xdce9ff, 2.1); rim.position.set(2, 2.2, -2.5); scene.add(rim);

let avatarRoot, restRig = new Map(), boneMap = new Map(), loaded = false, modelShoulderWidth = .42, lastBodyScale = 1.8;
let limbLengths = {};
let faceBindings = [];
let faceTest = null, faceOverride = null, lastFaceCategories = [];
let lastHipCenter = null, calibratedHip = null, lastHeadAngles = { yaw: 0, pitch: 0, roll: 0 }, calibratedHead = { yaw: 0, pitch: 0, roll: 0 };
let defaultModelReport = '';
const tempQ = new THREE.Quaternion(), parentQ = new THREE.Quaternion(), deltaQ = new THREE.Quaternion();
const tempDir = new THREE.Vector3();
const hipAnchor = new THREE.Vector3(), avatarRootBase = new THREE.Vector3();
let renderWidth = 0, renderHeight = 0;
let activeModelIsExternal = false, modelLoadSequence = 0;
function canonicalBoneName(raw) {
  const name=raw.toLowerCase().replace(/^mixamorig:?/,'').replace(/^armature/,'').replace(/[^a-z0-9]/g,'');
  const direct=name.match(/^(clavicle|upperarm|lowerarm|hand|thigh|calf|foot|ball)([lr])$/);
  if(direct)return `${direct[1]}_${direct[2]}`;
  const existingFinger=name.match(/^(thumb|index|middle|ring|pinky|little)(0?[123])([lr])$/);
  if(existingFinger)return `${existingFinger[1]==='little'?'pinky':existingFinger[1]}_0${existingFinger[2].replace(/^0/,'')}_${existingFinger[3]}`;
  const finger=name.match(/^(left|right)hand(thumb|index|middle|ring|pinky|little)([123])$/);
  if(finger)return `${finger[2]==='little'?'pinky':finger[2]}_0${finger[3]}_${finger[1]==='left'?'l':'r'}`;
  const biped=name.match(/^jbip([lr])(thumb|index|middle|ring|little|small|pinky)([123])$/);
  if(biped)return `${biped[2]==='small'||biped[2]==='little'?'pinky':biped[2]}_0${biped[3]}_${biped[1]}`;
  const ccFinger=name.match(/^ccbase([lr])(thumb|index|middle|mid|ring|little|pinky)([123])$/);
  if(ccFinger)return `${ccFinger[2]==='mid'?'middle':ccFinger[2]==='little'?'pinky':ccFinger[2]}_0${ccFinger[3]}_${ccFinger[1]}`;
  const aliases={
    hips:'pelvis',hip:'pelvis',pelvis:'pelvis',defhips:'pelvis',ccbasehip:'pelvis',ccbasehips:'pelvis',jbipchips:'pelvis',
    leftshoulder:'clavicle_l',rightshoulder:'clavicle_r',ccbaselclavicle:'clavicle_l',ccbaserclavicle:'clavicle_r',jbiplshoulder:'clavicle_l',jbiprshoulder:'clavicle_r',
    leftarm:'upperarm_l',rightarm:'upperarm_r',ccbaselupperarm:'upperarm_l',ccbaserupperarm:'upperarm_r',jbiplupperarm:'upperarm_l',jbiprupperarm:'upperarm_r',
    leftforearm:'lowerarm_l',rightforearm:'lowerarm_r',ccbaselforearm:'lowerarm_l',ccbaserforearm:'lowerarm_r',jbipllowerarm:'lowerarm_l',jbiprlowerarm:'lowerarm_r',
    lefthand:'hand_l',righthand:'hand_r',ccbaselhand:'hand_l',ccbaserhand:'hand_r',jbiplhand:'hand_l',jbiprhand:'hand_r',
    leftupleg:'thigh_l',rightupleg:'thigh_r',ccbaselthigh:'thigh_l',ccbaserthigh:'thigh_r',jbiplupperleg:'thigh_l',jbiprupperleg:'thigh_r',
    leftleg:'calf_l',rightleg:'calf_r',ccbaselcalf:'calf_l',ccbasercalf:'calf_r',jbipllowerleg:'calf_l',jbiprlowerleg:'calf_r',
    leftfoot:'foot_l',rightfoot:'foot_r',ccbaselfoot:'foot_l',ccbaserfoot:'foot_r',jbiplfoot:'foot_l',jbiprfoot:'foot_r',
    lefttoebase:'ball_l',righttoebase:'ball_r',head:'head',ccbasehead:'head',jbipchead:'head',
    spine:'spine_01',defspine:'spine_01',spine1:'spine_02',defspine001:'spine_02',spine2:'spine_03',defspine002:'spine_03',spine01:'spine_01',spine02:'spine_02',spine03:'spine_03',
    ccbasespine01:'spine_01',ccbasespine02:'spine_02',ccbasespine03:'spine_03',jbipcspine:'spine_01',jbipcchest:'spine_02',jbipcupperchest:'spine_03',
    neck:'neck_01',defneck:'neck_01',neck01:'neck_01',ccbasenecktwist01:'neck_01',jbipcneck:'neck_01'
  };
  const rigify=name.match(/^def(.+?)([lr])$/);
  if(rigify){
    const role=rigify[1],side=rigify[2];
    const limb={shoulder:'clavicle',clavicle:'clavicle',upperarm:'upperarm',forearm:'lowerarm',lowerarm:'lowerarm',hand:'hand',thigh:'thigh',upperleg:'thigh',shin:'calf',calf:'calf',lowerleg:'calf',foot:'foot'}[role];
    if(limb)return `${limb}_${side}`;
    const rf=role.match(/^f(thumb|index|middle|ring|pinky|little)(0?[123])$/);
    if(rf)return `${rf[1]==='little'?'pinky':rf[1]}_0${rf[2].replace(/^0/,'')}_${side}`;
  }
  return aliases[name]??raw;
}
function rememberRig(root) {
  root.updateMatrixWorld(true);
  root.traverse(object => {
    if (!object.isBone) return;
    const worldPosition = object.getWorldPosition(new THREE.Vector3());
    const worldQuaternion = object.getWorldQuaternion(new THREE.Quaternion());
    const childBone = object.children.find(child => child.isBone);
    let direction = new THREE.Vector3(0, 1, 0).applyQuaternion(worldQuaternion);
    if (childBone) direction = childBone.getWorldPosition(new THREE.Vector3()).sub(worldPosition).normalize();
    else if (object.parent?.isBone) direction = worldPosition.clone().sub(object.parent.getWorldPosition(new THREE.Vector3())).normalize();
    const name=canonicalBoneName(object.name);
    restRig.set(name, { bone: object, worldPosition, worldQuaternion, direction, localQuaternion: object.quaternion.clone(), localPosition: object.position.clone() });
    boneMap.set(name, object);
  });
  const pelvis = boneMap.get('pelvis');
  if (pelvis) pelvis.getWorldPosition(hipAnchor);
  const leftArm = boneMap.get('upperarm_l'), rightArm = boneMap.get('upperarm_r');
  if (leftArm && rightArm) modelShoulderWidth = leftArm.getWorldPosition(new THREE.Vector3()).distanceTo(rightArm.getWorldPosition(new THREE.Vector3()));
  const measure = (a, b) => {
    const A = boneMap.get(a), B = boneMap.get(b);
    if (!A || !B) return 0;
    return A.getWorldPosition(new THREE.Vector3()).distanceTo(B.getWorldPosition(new THREE.Vector3()));
  };
  limbLengths = {
    arm_l: { upper: measure('upperarm_l', 'lowerarm_l'), lower: measure('lowerarm_l', 'hand_l') },
    arm_r: { upper: measure('upperarm_r', 'lowerarm_r'), lower: measure('lowerarm_r', 'hand_r') },
    leg_l: { upper: measure('thigh_l', 'calf_l'), lower: measure('calf_l', 'foot_l') },
    leg_r: { upper: measure('thigh_r', 'calf_r'), lower: measure('calf_r', 'foot_r') }
  };
}

function describeDefaultModel(gltf) {
  const joints = [...boneMap.keys()];
  const fingers = joints.filter(name => /(thumb|index|middle|ring|little|pinky|finger)/i.test(name));
  const bodyRoles=['pelvis','spine_01','spine_02','neck_01','head','clavicle_l','upperarm_l','lowerarm_l','hand_l','clavicle_r','upperarm_r','lowerarm_r','hand_r','thigh_l','calf_l','foot_l','thigh_r','calf_r','foot_r'];
  const fingerRoles=['thumb','index','middle','ring','pinky'].flatMap(finger=>['l','r'].flatMap(side=>[1,2,3].map(segment=>`${finger}_0${segment}_${side}`)));
  const mappedBody=bodyRoles.filter(name=>boneMap.has(name)).length,mappedFingers=fingerRoles.filter(name=>boneMap.has(name)).length;
  const morphSet = new Set();
  gltf.scene.traverse(object => { if (object.isMesh) for (const name of Object.keys(object.morphTargetDictionary ?? {})) morphSet.add(name); });
  const morphs = [...morphSet];
  const expressions = morphs.filter(name => /(blink|jaw|mouth|smile|frown|brow|eye)/i.test(name));
  const channels = new Set(faceBindings.map(binding => binding.category)).size;
  defaultModelReport = [
    `身体骨架：检测到 ${joints.length} 个骨骼`,
    `身体映射覆盖：${mappedBody}/${bodyRoles.length} 个关键骨段`,
    `手指骨：识别到 ${fingers.length} 个命名关节；标准指节映射 ${mappedFingers}/${fingerRoles.length}`, 
    `面部形态键：${morphs.length ? `检测到 ${morphs.length} 个唯一目标${expressions.length ? `，其中 ${expressions.length} 个是表情` : ''}` : '未检测到；摄像头只显示表情信号，模型面部不会变形'}`,
    `表情实时映射：${channels} 个 ARKit 通道（共 ${faceBindings.length} 条网格通道）`,
    `动画片段：${gltf.animations?.length ?? 0} 条（实时追踪时不播放）`
  ].join('\n');
  document.querySelector('#rig-check').textContent = defaultModelReport;
}

function bindFaceMorphs(root) {
  faceBindings = [];
  const arkitNames = new Set([
    'browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight',
    'cheekPuff','cheekSquintLeft','cheekSquintRight','eyeBlinkLeft','eyeBlinkRight',
    'eyeLookDownLeft','eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight',
    'eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight',
    'jawForward','jawLeft','jawOpen','jawRight','mouthClose','mouthDimpleLeft','mouthDimpleRight',
    'mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft','mouthLowerDownRight',
    'mouthPressLeft','mouthPressRight','mouthPucker','mouthRight','mouthRollLower','mouthRollUpper',
    'mouthShrugLower','mouthShrugUpper','mouthSmileLeft','mouthSmileRight','mouthStretchLeft','mouthStretchRight',
    'mouthUpperUpLeft','mouthUpperUpRight','mouthUpperUp','noseSneerLeft','noseSneerRight','tongueOut'
  ]);
  const aliases = new Map([
    ['blink_l','eyeBlinkLeft'], ['eyeblink_l','eyeBlinkLeft'], ['blink_r','eyeBlinkRight'], ['eyeblink_r','eyeBlinkRight'],
    ['jaw_lower','jawOpen'], ['jawopen','jawOpen'], ['mouth_open','jawOpen'],
    ['mouth_smile_l','mouthSmileLeft'], ['smile_l','mouthSmileLeft'], ['mouth_smile_r','mouthSmileRight'], ['smile_r','mouthSmileRight'],
    ['mouth_smile','mouthSmile'], ['mouthfrown','mouthFrown'], ['mouth_pucker','mouthPucker'], ['mouthpress','mouthPress'], ['mouthstretch','mouthStretch'],
    ['mouth_right_up_smile','mouthSmileRight'], ['mouth_left_up_smile','mouthSmileLeft'],
    ['mouth_right_down','mouthFrownRight'], ['mouth_left_down','mouthFrownLeft'],
    ['mouth_right_wide','mouthStretchRight'], ['mouth_left_wide','mouthStretchLeft'],
    ['mouth_right_squeeze','mouthPressRight'], ['mouth_left_squeeze','mouthPressLeft'],
    ['mouth up','jawOpen'], ['top lip out','mouthFunnel'], ['lip','mouthPucker'],
    ['mad right','browDownRight'], ['mad left','browDownLeft'], ['shy right','eyeSquintRight'], ['shy left','eyeSquintLeft'],
    ['oout right','eyeWideRight'], ['out left','eyeWideLeft'],
    ['eyewide_l','eyeWideLeft'], ['eyewide_r','eyeWideRight'], ['eyesquint_l','eyeSquintLeft'], ['eyesquint_r','eyeSquintRight'],
    ['brow_inner_up','browInnerUp'], ['browup','browInnerUp'], ['browdown_l','browDownLeft'], ['browdown_r','browDownRight']
  ]);
  root.traverse(object => {
    for (const [name, index] of Object.entries(object.morphTargetDictionary ?? {})) {
      const normalized = name.toLowerCase().replace(/[^a-z0-9]/g, '');
      const alias = [...aliases].find(([key]) => key.replace(/[^a-z0-9]/gi, '').toLowerCase() === normalized)?.[1];
      const direct = [...arkitNames].find(value => value.toLowerCase().replace(/[^a-z0-9]/g,'') === normalized);
      const category = alias ?? [...aliases.values()].find(value => value.toLowerCase() === normalized) ?? direct ?? null;
      if (category) faceBindings.push({ mesh: object, index, category });
    }
  });
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h || (w === renderWidth && h === renderHeight)) return;
  renderWidth = w; renderHeight = h;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function bodyFrame(pose) {
  const visible = p => !!p && (p.visibility ?? p.presence ?? 1) > .22;
  const mid = (a, b) => ({ x:(a.x+b.x)/2, y:(a.y+b.y)/2, z:((a.z??0)+(b.z??0))/2 });
  const shoulders = visible(pose?.[11]) && visible(pose?.[12]) ? mid(pose[11],pose[12]) : null;
  const hips = visible(pose?.[23]) && visible(pose?.[24]) ? mid(pose[23],pose[24]) : null;
  if (shoulders) {
    const width=Math.hypot(pose[11].x-pose[12].x,pose[11].y-pose[12].y);
    if(width>.18) lastBodyScale=THREE.MathUtils.lerp(lastBodyScale,THREE.MathUtils.clamp(modelShoulderWidth/width,.9,2.1),.08);
  }
  const origin=hips??shoulders??{x:.5,y:.55,z:0}, scale=lastBodyScale;
  const cameraPoint=p=>{
    return new THREE.Vector3((p.x-origin.x)*scale,hipAnchor.y+(origin.y-p.y)*scale,-((p.z??0)-(origin.z??0))*scale);
  };
  const point=id=>visible(pose?.[id])?cameraPoint(pose[id]):null;
  const target=new Map();
  for(let id=0;id<33;id++) target.set(id,point(id));
  if(hips) target.set('hips',new THREE.Vector3(0,hipAnchor.y,0));
  if(shoulders) target.set('shoulders',point(11).clone().add(point(12)).multiplyScalar(.5));
  if(hips&&shoulders) target.set('midtorso',target.get('hips').clone().lerp(target.get('shoulders'),.5));
  const nose=point(0), shoulderCenter=target.get('shoulders');
  if(nose&&shoulderCenter) target.set('neck',shoulderCenter.clone().lerp(nose,.18));
  target.set('nose',nose);
  return {target,hips,scale,origin,cameraPoint};
}

function setBoneToward(name, from, to, strength = .28, maxAngle = Math.PI) {
  const rest = restRig.get(name);
  if (!rest || !from || !to) return;
  tempDir.copy(to).sub(from);
  if (tempDir.lengthSq() < 1e-6) return;
  tempDir.normalize();
  deltaQ.setFromUnitVectors(rest.direction, tempDir);
  const angle = deltaQ.angleTo(new THREE.Quaternion());
  if (angle > maxAngle) deltaQ.slerp(new THREE.Quaternion(), maxAngle / angle);
  tempQ.copy(deltaQ).multiply(rest.worldQuaternion);
  const parent = rest.bone.parent;
  if (parent) {
    parent.updateWorldMatrix(true, false);
    parent.getWorldQuaternion(parentQ).invert();
    tempQ.premultiply(parentQ);
  }
  rest.bone.quaternion.slerp(tempQ, strength);
  rest.bone.updateWorldMatrix(true, true);
}

function setBoneWorldQuaternion(name, worldQuaternion, strength = .28) {
  const rest=restRig.get(name);
  if(!rest)return;
  const local=worldQuaternion.clone();
  const parent=rest.bone.parent;
  if(parent){parent.updateWorldMatrix(true,false);parent.getWorldQuaternion(parentQ).invert();local.premultiply(parentQ);}
  rest.bone.quaternion.slerp(local,strength);
  rest.bone.updateWorldMatrix(true,true);
}

function orientationFrame(longAxis, acrossAxis) {
  if(longAxis.lengthSq()<1e-6||acrossAxis.lengthSq()<1e-6)return null;
  const y=longAxis.clone().normalize();
  const x=acrossAxis.clone().addScaledVector(y,-acrossAxis.dot(y)).normalize();
  if(x.lengthSq()<1e-6)return null;
  const z=x.clone().cross(y).normalize();
  x.crossVectors(y,z).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x,y,z));
}

function worldPoint(name) {
  const bone = boneMap.get(name);
  return bone ? bone.getWorldPosition(new THREE.Vector3()) : null;
}

function aimBone(name, from, to, strength, maxAngle) {
  if (!from || !to) return;
  setBoneToward(name, from, to, strength, maxAngle);
  boneMap.get(name)?.updateWorldMatrix(true, true);
}

// Two-bone IK keeps the limb's real bone lengths, so the avatar bends instead of
// stretching when the tracked wrist or ankle is closer than the arm can reach.
function solveTwoBone(root, end, l1, l2, poleHint) {
  if (!(l1 > 0) || !(l2 > 0) || !end) return null;
  const delta = end.clone().sub(root);
  const distance = delta.length();
  if (distance < 1e-4) return null;
  const direction = delta.multiplyScalar(1 / distance);
  const reach = THREE.MathUtils.clamp(distance, Math.abs(l1 - l2) + .01, l1 + l2 - .01);
  const goal = root.clone().addScaledVector(direction, reach);
  const along = (l1 * l1 - l2 * l2 + reach * reach) / (2 * reach);
  const height = Math.sqrt(Math.max(0, l1 * l1 - along * along));
  const pole = poleHint.clone().addScaledVector(direction, -poleHint.dot(direction));
  if (pole.lengthSq() < 1e-6) pole.set(0, 0, 1).addScaledVector(direction, -direction.z);
  pole.normalize();
  const joint = root.clone().addScaledVector(direction, along).addScaledVector(pole, height);
  return { joint, goal };
}

function applyBody(pose) {
  if(!pose?.length) return;
  const {target,hips,scale,origin}=bodyFrame(pose);
  if(hips){
    lastHipCenter={x:hips.x,y:hips.y};
    const anchor=calibratedHip??{x:.5,y:.55},rootTarget=avatarRootBase.clone();
    rootTarget.x+=THREE.MathUtils.clamp((hips.x-anchor.x)*scale*.6,-.18,.18);
    // Keep the pelvis height fixed so hip-estimation noise does not lift both feet off the floor.
    avatarRoot.position.lerp(rootTarget,.12);avatarRoot.updateMatrixWorld(true);
  }
  const get=key=>target.get(key);
  const shoulder=get('shoulders'), hip=get('hips'), torso=get('midtorso'), neck=get('neck'), nose=get('nose');
  // Keep the torso as a stable human volume.  A monocular camera does not give
  // reliable depth for the spine, so letting every spine segment chase noisy
  // 2-D landmarks creates the characteristic "broken waist" twist.  The
  // pelvis remains anchored and only a very small correction is allowed here;
  // shoulders/arms still follow the detected pose independently.
  const spinePairs=[
    ['spine_01',hip,torso,.035,.04],['spine_02',torso,shoulder,.04,.05],['spine_03',shoulder,neck,.05,.08],
    ['clavicle_l',shoulder,get(11),.08,.12],['clavicle_r',shoulder,get(12),.08,.12],
    ['neck_01',shoulder,nose,.06,.10]
  ];
  for(const [bone,from,to,strength,maxAngle] of spinePairs) if(from&&to) setBoneToward(bone,from,to,strength,maxAngle);

  // Arms: IK to the tracked wrist with an outward/down elbow pole.
  for (const side of ['l','r']) {
    const lengths=limbLengths[`arm_${side}`];
    const upper=boneMap.get(`upperarm_${side}`);
    const elbow=get(side==='l'?13:14), wrist=get(side==='l'?15:16), mcp=get(side==='l'?17:18);
    if(!upper||!lengths?.upper||!lengths?.lower||!wrist) continue;
    upper.updateWorldMatrix(true,false);
    const root=upper.getWorldPosition(new THREE.Vector3());
    const outward=Math.sign(root.x||1);
    const poleHint=elbow?elbow.clone().sub(root):new THREE.Vector3(outward*.55,-1,-.55);
    const ik=solveTwoBone(root,wrist,lengths.upper,lengths.lower,poleHint);
    if(!ik) continue;
    aimBone(`upperarm_${side}`,root,ik.joint,.32,1.9);
    aimBone(`lowerarm_${side}`,worldPoint(`lowerarm_${side}`),ik.goal,.32,2.2);
    const handHead=worldPoint(`hand_${side}`);
    if(handHead&&mcp) setBoneToward(`hand_${side}`,handHead,mcp,.18,.8);
  }

  // Legs: IK only when knee and ankle are confidently visible; otherwise keep the
  // rest legs, which looks far better than following half-occluded points.
  for (const side of ['l','r']) {
    const kneePose=pose[side==='l'?25:26], anklePose=pose[side==='l'?27:28];
    if((kneePose?.visibility??1)<.4||(anklePose?.visibility??1)<.4) continue;
    const lengths=limbLengths[`leg_${side}`];
    const thigh=boneMap.get(`thigh_${side}`);
    const ankle=get(side==='l'?27:28);
    if(!thigh||!lengths?.upper||!lengths?.lower||!ankle) continue;
    thigh.updateWorldMatrix(true,false);
    const root=thigh.getWorldPosition(new THREE.Vector3());
    const ik=solveTwoBone(root,ankle,lengths.upper,lengths.lower,new THREE.Vector3(0,.15,1));
    if(!ik) continue;
    aimBone(`thigh_${side}`,root,ik.joint,.3,1.9);
    aimBone(`calf_${side}`,worldPoint(`calf_${side}`),ik.goal,.3,2.3);
    const toe=get(side==='l'?31:32), footHead=worldPoint(`foot_${side}`);
    if(footHead&&toe) setBoneToward(`foot_${side}`,footHead,toe,.18,.7);
  }
  return {scale,origin};
}

function applyHands(hands, handedness, pose, handsWorld, scale, origin) {
  if (!hands.length) return;
  const fingers = [['index', 5], ['middle', 9], ['ring', 13], ['pinky', 17]];
  for (let handIndex = 0; handIndex < hands.length; handIndex++) {
    const hand = hands[handIndex];
    if (!hand?.length || !hand[0]) continue;
    const label = handedness?.[handIndex]?.toLowerCase();
    const leftWrist = pose[15]?.visibility > .22 ? pose[15] : null;
    const rightWrist = pose[16]?.visibility > .22 ? pose[16] : null;
    const leftDistance = leftWrist ? Math.hypot(hand[0].x - leftWrist.x, hand[0].y - leftWrist.y) : Infinity;
    const rightDistance = rightWrist ? Math.hypot(hand[0].x - rightWrist.x, hand[0].y - rightWrist.y) : Infinity;
    const side = leftDistance === rightDistance ? (label === 'left' ? 'l' : 'r') : leftDistance < rightDistance ? 'l' : 'r';
    const points = handsWorld?.[handIndex]?.length === 21 ? handsWorld[handIndex] : hand;
    // MediaPipe hand world landmarks are already expressed in the camera frame, so
    // converting the axis (x right, y up, z toward viewer) gives the true 3D finger
    // direction. The previous hand-frame alignment introduced the twisted fingers.
    const direction = (a, b) => {
      const A = points[a], B = points[b];
      if (!A || !B) return null;
      const vector = new THREE.Vector3(B.x - A.x, -(B.y - A.y), -(B.z - A.z));
      return vector.lengthSq() < 1e-6 ? null : vector.normalize();
    };
    const apply = (name, a, b, strength, maxAngle) => {
      const head = worldPoint(`${name}_${side}`);
      const vec = direction(a, b);
      if (!head || !vec) return;
      setBoneToward(`${name}_${side}`, head, head.clone().add(vec), strength, maxAngle);
    };
    const handRest=restRig.get(`hand_${side}`),middleRest=restRig.get(`middle_01_${side}`),indexRest=restRig.get(`index_01_${side}`),pinkyRest=restRig.get(`pinky_01_${side}`);
    const sourceLong=handRest&&middleRest?middleRest.worldPosition.clone().sub(handRest.worldPosition):null;
    const sourceAcross=indexRest&&pinkyRest?indexRest.worldPosition.clone().sub(pinkyRest.worldPosition):null;
    const targetLong=direction(0,9),targetAcross=direction(17,5);
    const sourceFrame=sourceLong&&sourceAcross?orientationFrame(sourceLong,sourceAcross):null;
    const targetFrame=targetLong&&targetAcross?orientationFrame(targetLong,targetAcross):null;
    if(handRest&&sourceFrame&&targetFrame){
      const delta=targetFrame.multiply(sourceFrame.invert());
      setBoneWorldQuaternion(`hand_${side}`,delta.multiply(handRest.worldQuaternion),.38);
    }
    apply('thumb_01', 1, 2, .5, 1.1); apply('thumb_02', 2, 3, .5, 1.0); apply('thumb_03', 3, 4, .5, .9);
    for (const [name, start] of fingers) {
      apply(`${name}_01`, start, start + 1, .5, 1.3);
      apply(`${name}_02`, start + 1, start + 2, .5, 1.5);
      apply(`${name}_03`, start + 2, start + 3, .5, 1.2);
    }
  }
}

function applyFace(categories) {
  const map = new Map((categories ?? []).map(item => [item.categoryName, item.score]));
  const pick = (...names) => {
    for (const name of names) {
      const value = map.get(name);
      if (typeof value === 'number') return value;
    }
    return 0;
  };
  const average = (left, right) => (left + right) * .5;
  const smileLeft = pick('mouthSmileLeft'), smileRight = pick('mouthSmileRight');
  const frownLeft = pick('mouthFrownLeft'), frownRight = pick('mouthFrownRight');
  const pressLeft = pick('mouthPressLeft'), pressRight = pick('mouthPressRight');
  const scores = {
    eyeBlinkLeft: pick('eyeBlinkLeft'), eyeBlinkRight: pick('eyeBlinkRight'),
    jawOpen: pick('jawOpen', 'mouthOpen', 'jawLower'),
    mouthSmileLeft: smileLeft, mouthSmileRight: smileRight, mouthSmile: average(smileLeft, smileRight),
    mouthFrownLeft: frownLeft, mouthFrownRight: frownRight, mouthFrown: average(frownLeft, frownRight),
    mouthPucker: pick('mouthPucker'), mouthClose: pick('mouthClose'),
    mouthPressLeft: pressLeft, mouthPressRight: pressRight, mouthPress: Math.max(pressLeft, pressRight),
    mouthStretchLeft: pick('mouthStretchLeft'), mouthStretchRight: pick('mouthStretchRight'),
    mouthUpperUpLeft: pick('mouthUpperUpLeft'), mouthUpperUpRight: pick('mouthUpperUpRight'),
    mouthLowerDownLeft: pick('mouthLowerDownLeft'), mouthLowerDownRight: pick('mouthLowerDownRight'),
    mouthFunnel: pick('mouthFunnel'), cheekPuff: pick('cheekPuff'),
    cheekSquintLeft: pick('cheekSquintLeft'), cheekSquintRight: pick('cheekSquintRight'),
    noseSneerLeft: pick('noseSneerLeft'), noseSneerRight: pick('noseSneerRight'),
    browDownLeft: pick('browDownLeft'), browDownRight: pick('browDownRight'), browInnerUp: pick('browInnerUp'),
    browOuterUpLeft: pick('browOuterUpLeft'), browOuterUpRight: pick('browOuterUpRight'),
    eyeWideLeft: pick('eyeWideLeft'), eyeWideRight: pick('eyeWideRight'),
    eyeSquintLeft: pick('eyeSquintLeft'), eyeSquintRight: pick('eyeSquintRight'),
    eyeLookInLeft: pick('eyeLookInLeft'), eyeLookOutLeft: pick('eyeLookOutLeft'),
    eyeLookInRight: pick('eyeLookInRight'), eyeLookOutRight: pick('eyeLookOutRight'),
    tongueOut: pick('tongueOut')
  };
  for (const binding of faceBindings) {
    const override = faceOverride ? faceOverride[binding.category] : undefined;
    const target = override ?? scores[binding.category] ?? map.get(binding.category) ?? 0;
    const influences = binding.mesh.morphTargetInfluences;
    influences[binding.index] = THREE.MathUtils.lerp(influences[binding.index] ?? 0, THREE.MathUtils.clamp(target, 0, 1), .55);
  }
}

const FACE_TEST_STEPS = [
  { label: '眨眼', channels: { eyeBlinkLeft: 1, eyeBlinkRight: 1 } },
  { label: '张嘴', channels: { jawOpen: .8 } },
  { label: '微笑', channels: { mouthSmileLeft: .9, mouthSmileRight: .9, cheekSquintLeft: .35, cheekSquintRight: .35 } },
  { label: '抬眉并睁大眼', channels: { browInnerUp: .85, browOuterUpLeft: .5, browOuterUpRight: .5, eyeWideLeft: .7, eyeWideRight: .7 } },
  { label: '皱眉', channels: { browDownLeft: .85, browDownRight: .85, mouthPressLeft: .4, mouthPressRight: .4 } }
];

window.addEventListener('rove-face-test', () => {
  if (!faceBindings.length) {
    state.textContent = '此角色没有可识别的面部形态键，无法做表情自检';
    return;
  }
  if (faceTest) clearInterval(faceTest);
  let step = 0;
  faceOverride = { ...FACE_TEST_STEPS[0].channels };
  state.textContent = `表情自检：${FACE_TEST_STEPS[0].label}`;
  faceTest = setInterval(() => {
    step += 1;
    if (step >= FACE_TEST_STEPS.length) {
      clearInterval(faceTest);
      faceTest = null;
      faceOverride = null;
      state.textContent = '表情自检完成 · 等待追踪';
      return;
    }
    faceOverride = { ...FACE_TEST_STEPS[step].channels };
    state.textContent = `表情自检：${FACE_TEST_STEPS[step].label}`;
  }, 1100);
});

function applyHead(landmarks) {
  const rest = restRig.get('head');
  if (!rest) return;
  let yaw = 0, pitch = 0, roll = 0;
  if (landmarks?.[33] && landmarks?.[263] && landmarks?.[1]) {
    const a = landmarks[33], b = landmarks[263], nose = landmarks[1];
    const left = a.x <= b.x ? a : b, right = a.x <= b.x ? b : a;
    const width = Math.max(.04, right.x - left.x);
    yaw = THREE.MathUtils.clamp((nose.x - (left.x + right.x) / 2) / width * 1.5, -.45, .45);
    roll = THREE.MathUtils.clamp(-Math.atan2(right.y - left.y, right.x - left.x), -.4, .4);
    if(landmarks[10]&&landmarks[152]){
      const point=p=>new THREE.Vector3(p.x,-p.y,-(p.z??0));
      const eyeAxis=point(right).sub(point(left)).normalize();
      const faceUp=point(landmarks[10]).sub(point(landmarks[152])).normalize();
      const normal=eyeAxis.cross(faceUp).normalize();
      if(normal.z<0)normal.negate();
      pitch=THREE.MathUtils.clamp(-Math.atan2(normal.y,Math.max(.1,normal.z)),-.45,.45);
      yaw=THREE.MathUtils.clamp(Math.atan2(normal.x,Math.max(.1,normal.z)),-.45,.45);
    }
  }
  lastHeadAngles = { yaw, pitch, roll };
  yaw -= calibratedHead.yaw; pitch -= calibratedHead.pitch; roll -= calibratedHead.roll;
  const target = rest.localQuaternion.clone().multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, roll, 'XYZ')));
  rest.bone.quaternion.slerp(target, landmarks?.length ? .16 : .08);
  rest.bone.updateWorldMatrix(true, true);
}

function resetRig() {
  for (const rest of restRig.values()) {
    rest.bone.quaternion.copy(rest.localQuaternion);
    rest.bone.position.copy(rest.localPosition);
  }
  if (avatarRoot) avatarRoot.position.copy(avatarRootBase);
  avatarRoot?.updateMatrixWorld(true);
}

function installAvatar(gltf, external, name='') {
  if(avatarRoot){scene.remove(avatarRoot);avatarRoot.traverse(object=>{object.geometry?.dispose();for(const material of (Array.isArray(object.material)?object.material:[object.material])){material?.map?.dispose();material?.dispose();}});}
  restRig.clear();boneMap.clear();faceBindings=[];
  avatarRoot=gltf.scene;
  if(external){
    const box=new THREE.Box3().setFromObject(avatarRoot),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());
    const scale=THREE.MathUtils.clamp(1.72/Math.max(.1,size.y),.01,20);
    avatarRoot.scale.setScalar(scale);avatarRoot.position.set(-center.x*scale,-box.min.y*scale,-center.z*scale);
  }
  avatarRootBase.copy(avatarRoot.position);scene.add(avatarRoot);rememberRig(avatarRoot);bindFaceMorphs(avatarRoot);describeDefaultModel(gltf);
  loaded=true;activeModelIsExternal=external;window.__roveExternalModelActive=external;
  window.__roveAvatarCapabilities={faceMorphCount:new Set(faceBindings.map(binding=>binding.category)).size,faceBindingCount:faceBindings.length};
  window.dispatchEvent(new CustomEvent('rove-avatar-capabilities',{detail:window.__roveAvatarCapabilities}));
  document.querySelector('#asset-label').textContent=external?`人物资产 · ${name}`:'人物资产 · CC0 女性角色 · 含表情';
  state.textContent=external?(faceBindings.length?`已加载并映射 ${window.__roveAvatarCapabilities.faceMorphCount} 个表情通道`:'已加载 · 此模型无可识别面部键'):'CC0 女性角色 · 等待追踪';
  canvas.style.display='block';modelView.style.display='none';empty.style.display='none';resize();
}

function loadAvatar(url, external=false, name='') {
  const sequence=++modelLoadSequence;
  state.textContent=external?'正在载入并检查骨骼/表情…':'正在载入 CC0 女性角色…';
  new GLTFLoader().load(url,gltf=>{if(sequence===modelLoadSequence)installAvatar(gltf,external,name);},undefined,error=>{
    if(sequence!==modelLoadSequence)return;
    window.__roveExternalModelActive=activeModelIsExternal;state.textContent=`人物模型加载失败：${error.message}`;
    if(!loaded)document.querySelector('#rig-check').textContent=`人物模型加载失败：${error.message}`;
  });
}

const defaultAvatarUrl='./models/rove-avatar-cc0.glb';
loadAvatar(defaultAvatarUrl);
window.addEventListener('rove-load-avatar',event=>loadAvatar(event.detail.url,true,event.detail.name??'自定义角色'));

window.addEventListener('rove-restore-avatar', () => {
  if(activeModelIsExternal){loadAvatar(defaultAvatarUrl,false);return;}
  canvas.style.display = loaded ? 'block' : 'none';
  modelView.style.display = 'none';
  empty.style.display = loaded ? 'none' : 'grid';
  if (loaded) state.textContent = 'CC0 女性角色 · 等待追踪';
  if (defaultModelReport) document.querySelector('#rig-check').textContent = defaultModelReport;
});

window.addEventListener('rove-calibrate', () => {
  if (!lastHipCenter) {
    document.querySelector('#body-hint').textContent = '还没有稳定身体追踪；先站到全身入镜并等待“身体追踪”出现，再校准。';
    return;
  }
  calibratedHip = { ...lastHipCenter };
  calibratedHead = { ...lastHeadAngles };
  document.querySelector('#body-hint').textContent = '中立位置已校准；之后的左右移动、身体高低和头部转向会相对这个位置映射。';
  state.textContent = '中立位置已校准';
});

window.addEventListener('rove-motion', event => {
  const { pose, hands = [], handedness = [], face = [], faceLandmarks = [] } = event.detail;
  window.__roveHands = { all: hands, labels: handedness };
  let frame = {scale:lastBodyScale,origin:{x:.5,y:.55}};
  if (loaded && pose) frame = applyBody(pose) ?? frame;
  else lastHipCenter = null;
  if (loaded) applyHands(hands, handedness, pose ?? [], event.detail.handsWorld ?? [], frame.scale, frame.origin);
  if (loaded) applyHead(faceLandmarks);
  lastFaceCategories = face;
  applyFace(face);
  if (loaded) {
    canvas.style.display = 'block'; empty.style.display = 'none';
    const active = [pose && '身体骨骼', hands.length && '手指骨骼', face.length && (faceBindings.length ? '面部表情' : '面部信号')].filter(Boolean);
    state.textContent = active.length ? `实时映射 · ${active.join(' / ')}` : '等待检测到人物';
  }
});

document.querySelector('#camera-btn').addEventListener('click', () => {
  canvas.style.display = loaded ? 'block' : 'none';
  modelView.style.display = 'none';
  empty.style.display = loaded ? 'none' : 'grid';
});
document.querySelector('#stop-btn').addEventListener('click', () => {
  resetRig();
  canvas.style.display = loaded ? 'block' : 'none';
  modelView.style.display = 'none';
  empty.style.display = loaded ? 'none' : 'grid';
  if (loaded) state.textContent = 'CC0 女性角色 · 等待追踪';
});

window.addEventListener('resize', resize);
function render() { resize(); applyFace(lastFaceCategories); renderer.render(scene, camera); requestAnimationFrame(render); }
render();
