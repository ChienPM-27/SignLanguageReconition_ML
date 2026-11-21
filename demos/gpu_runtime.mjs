import { FilesetResolver, HandLandmarker, FaceLandmarker, PoseLandmarker } from "/vision_bundle.mjs";

const tasks = {};
const hardware = {};

function gpuCanvas(name) {
  const canvas = new OffscreenCanvas(1, 1);
  const gl = canvas.getContext("webgl2", { powerPreference: "high-performance" });
  if (!gl) throw new Error("WebGL2 hardware context is unavailable");
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  if (!renderer || /swiftshader|llvmpipe|software|basic render|softpipe/i.test(renderer)) {
    throw new Error(`Hardware GPU is required; renderer was ${renderer}`);
  }
  const info = { renderer, delegate: "GPU", drawCalls: 0 };
  for (const method of ["drawArrays", "drawElements", "drawArraysInstanced", "drawElementsInstanced"]) {
    const original = gl[method].bind(gl);
    gl[method] = (...args) => { info.drawCalls++; return original(...args); };
  }
  hardware[name] = info;
  return canvas;
}

window.initializeGpuLandmarks = async (options) => {
  const fileset = await FilesetResolver.forVisionTasks("/wasm");
  const common = (name, path) => ({
    baseOptions: { modelAssetPath: path, delegate: "GPU" },
    canvas: gpuCanvas(name), runningMode: "IMAGE"
  });
  tasks.hands = await HandLandmarker.createFromOptions(fileset, {
    ...common("hands", "/models/hand_landmarker.task"), numHands: 2,
    minHandDetectionConfidence: options.detectionConfidence,
    minHandPresenceConfidence: options.detectionConfidence,
    minTrackingConfidence: options.trackingConfidence
  });
  tasks.face = await FaceLandmarker.createFromOptions(fileset, {
    ...common("face", "/models/face_landmarker.task"), numFaces: 1,
    minFaceDetectionConfidence: options.detectionConfidence,
    minFacePresenceConfidence: options.detectionConfidence,
    minTrackingConfidence: options.trackingConfidence,
    outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false
  });
  tasks.pose = await PoseLandmarker.createFromOptions(fileset, {
    ...common("pose", "/models/pose_landmarker_lite.task"), numPoses: 1,
    minPoseDetectionConfidence: options.detectionConfidence,
    minPosePresenceConfidence: options.detectionConfidence,
    minTrackingConfidence: options.trackingConfidence,
    outputSegmentationMasks: false
  });
  return { sdk: "0.10.35", hardware };
};

async function decodeFrame({ png, imagePath }) {
  if (imagePath) {
    const response = await fetch(imagePath, { cache: "no-store" });
    if (!response.ok) throw new Error("Local frame transfer failed");
    return createImageBitmap(await response.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  }
  let data;
  if (Uint8Array.fromBase64) data = Uint8Array.fromBase64(png);
  else {
    const binary = atob(png);
    data = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) data[index] = binary.charCodeAt(index);
  }
  return createImageBitmap(new Blob([data], { type: "image/png" }), {
    colorSpaceConversion: "none", premultiplyAlpha: "none"
  });
}

window.extractGpuLandmarks = async (frame) => {
  const image = await decodeFrame(frame);
  try {
  const started = performance.now();
  const before = Object.fromEntries(Object.entries(hardware).map(([name, info]) => [name, info.drawCalls]));
  const hands = tasks.hands.detect(image);
  const face = tasks.face.detect(image);
  const pose = tasks.pose.detect(image);
  const coords = (points) => points.map(({ x, y, z }) => [x, y, z]);
  return {
    hands: hands.landmarks.map(coords),
    handedness: hands.handedness.map((categories) => categories[0]?.categoryName ?? "Unknown"),
    face: face.faceLandmarks.length ? coords(face.faceLandmarks[0].slice(0, 468)) : null,
    pose: pose.landmarks.length ? coords(pose.landmarks[0]) : null,
    inferenceMs: performance.now() - started,
    drawCalls: Object.fromEntries(Object.entries(hardware).map(([name, info]) => [name, info.drawCalls - before[name]]))
  };
  } finally { image.close(); }
};

window.extractGpuBatch = async (frames) => {
  const results = [];
  for (const frame of frames) results.push(await window.extractGpuLandmarks(frame));
  return results;
};

window.verifyPixelTransfer = async (frame) => {
  const image = await decodeFrame(frame);
  try {
    const canvas = new OffscreenCanvas(image.width, image.height);
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, image.width, image.height).data;
    const hash = await crypto.subtle.digest("SHA-256", pixels);
    return Array.from(new Uint8Array(hash), value => value.toString(16).padStart(2, "0")).join("");
  } finally { image.close(); }
};

window.closeGpuLandmarks = () => {
  for (const task of Object.values(tasks)) task.close();
};
window.gpuRuntimeLoaded = true;
