// app.js

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as depthEstimation from "@tensorflow-models/depth-estimation";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import TRIANGULATION from "./TRIANGULATION.json";

// Define outer ring landmark indices
const OUTER_RING_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109,
];

async function init() {
  try {
    // Load models
    const [detector, depthEstimator] = await loadModels();

    // Load image
    const img = await loadImage("inputImage");

    // Process image
    const { depthImage, predictions } = await processImage(
      img,
      detector,
      depthEstimator
    );

    if (predictions.length === 0) {
      console.log("No faces detected.");
      return;
    }

    const keypoints = predictions[0].keypoints;

    // Draw overlays
    drawOverlays(img, keypoints, depthImage);

    // Set up Three.js scenes
    setupThreeJSScenes(img, keypoints);
  } catch (error) {
    console.error("An error occurred:", error);
  }
}

async function loadModels() {
  const detector = await faceLandmarksDetection.createDetector(
    faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
    {
      runtime: "tfjs",
      maxFaces: 1,
      refineLandmarks: false,
    }
  );

  const depthEstimator = await depthEstimation.createEstimator(
    depthEstimation.SupportedModels.ARPortraitDepth,
    {
      outputDepthRange: [0, 1],
    }
  );

  return [detector, depthEstimator];
}

function loadImage(imageId) {
  return new Promise((resolve, reject) => {
    const img = document.getElementById(imageId);
    if (img.complete && img.naturalHeight !== 0) {
      resolve(img);
    } else {
      img.onload = () => resolve(img);
      img.onerror = reject;
    }
  });
}

async function processImage(img, detector, depthEstimator) {
  const depthMap = await depthEstimator.estimateDepth(img, {
    minDepth: 0,
    maxDepth: 1,
  });

  const depthImage = await depthMap.toCanvasImageSource();

  const predictions = await detector.estimateFaces(img, {
    flipHorizontal: false,
  });

  return { depthImage, predictions };
}

function drawOverlays(img, keypoints, depthImage) {
  // Get canvas contexts
  const outputCanvas = document.getElementById("outputCanvas");
  const outputCtx = outputCanvas.getContext("2d");
  const outerRingCanvas = document.getElementById("outerRingCanvas");
  const outerRingCtx = outerRingCanvas.getContext("2d");
  const maskedDepthCanvas = document.getElementById("maskedDepthCanvas");
  const maskedDepthCtx = maskedDepthCanvas.getContext("2d");
  const invertedDepthCanvas = document.getElementById("invertedDepthCanvas");
  const invertedDepthCtx = invertedDepthCanvas.getContext("2d");
  const invertedMaskedDepthCanvas = document.getElementById(
    "invertedMaskedDepthCanvas"
  );
  const invertedMaskedDepthCtx = invertedMaskedDepthCanvas.getContext("2d");
  const triangulationCanvas = document.getElementById("triangulationCanvas");
  const triangulationCtx = triangulationCanvas.getContext("2d");
  const combinedOverlayCanvas = document.getElementById(
    "combinedOverlayCanvas"
  );
  const combinedOverlayCtx = combinedOverlayCanvas.getContext("2d");
  const depthCanvas = document.getElementById("depthCanvas");
  const depthCtx = depthCanvas.getContext("2d");

  // Draw the original image on the output canvas
  outputCtx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);

  // Draw keypoints
  drawKeypoints(outputCtx, keypoints);

  // Draw outer ring
  drawOuterRing(outerRingCtx, img, keypoints);

  // Draw depth map
  drawDepthMap(depthCtx, depthImage);

  // Draw masked depth map
  drawMaskedDepthMap(maskedDepthCtx, depthImage, keypoints);

  // Invert depth map
  invertCanvasImage(invertedDepthCtx, depthImage);

  // Invert masked depth map
  invertCanvasImage(invertedMaskedDepthCtx, maskedDepthCanvas);

  // Draw triangulation overlay
  drawTriangulation(triangulationCtx, img, keypoints);

  // Draw combined overlay
  drawCombinedOverlay(combinedOverlayCtx, img, keypoints);
}

function drawKeypoints(ctx, keypoints) {
  ctx.fillStyle = "red";
  keypoints.forEach(({ x, y }) => {
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function drawOuterRing(ctx, img, keypoints) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(img, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = "blue";
  ctx.lineWidth = 2;
  ctx.beginPath();

  const startIdx = OUTER_RING_INDICES[0];
  const startPoint = keypoints[startIdx];
  ctx.moveTo(startPoint.x, startPoint.y);

  OUTER_RING_INDICES.slice(1).forEach((idx) => {
    const point = keypoints[idx];
    ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  ctx.stroke();
}

function drawDepthMap(ctx, depthImage) {
  ctx.drawImage(depthImage, 0, 0, ctx.canvas.width, ctx.canvas.height);
}

function drawMaskedDepthMap(ctx, depthImage, keypoints) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.save();
  ctx.beginPath();

  const startIdx = OUTER_RING_INDICES[0];
  const startPoint = keypoints[startIdx];
  ctx.moveTo(startPoint.x, startPoint.y);

  OUTER_RING_INDICES.slice(1).forEach((idx) => {
    const point = keypoints[idx];
    ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  ctx.clip();

  ctx.drawImage(depthImage, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();
}

function invertCanvasImage(ctx, source) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(source, 0, 0, ctx.canvas.width, ctx.canvas.height);
  const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 255 - data[i]; // Red
    data[i + 1] = 255 - data[i + 1]; // Green
    data[i + 2] = 255 - data[i + 2]; // Blue
  }
  ctx.putImageData(imageData, 0, 0);
}

function drawTriangulation(ctx, img, keypoints) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(img, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = "green";
  ctx.lineWidth = 1;

  for (let i = 0; i < TRIANGULATION.length; i += 3) {
    const point1 = keypoints[TRIANGULATION[i]];
    const point2 = keypoints[TRIANGULATION[i + 1]];
    const point3 = keypoints[TRIANGULATION[i + 2]];

    ctx.beginPath();
    ctx.moveTo(point1.x, point1.y);
    ctx.lineTo(point2.x, point2.y);
    ctx.lineTo(point3.x, point3.y);
    ctx.closePath();
    ctx.stroke();
  }
}

function drawCombinedOverlay(ctx, img, keypoints) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(img, 0, 0, ctx.canvas.width, ctx.canvas.height);

  // Draw triangulation
  ctx.strokeStyle = "green";
  ctx.lineWidth = 1;
  for (let i = 0; i < TRIANGULATION.length; i += 3) {
    const point1 = keypoints[TRIANGULATION[i]];
    const point2 = keypoints[TRIANGULATION[i + 1]];
    const point3 = keypoints[TRIANGULATION[i + 2]];

    ctx.beginPath();
    ctx.moveTo(point1.x, point1.y);
    ctx.lineTo(point2.x, point2.y);
    ctx.lineTo(point3.x, point3.y);
    ctx.closePath();
    ctx.stroke();
  }

  // Draw outer ring
  ctx.strokeStyle = "blue";
  ctx.lineWidth = 2;
  ctx.beginPath();
  const startIdx = OUTER_RING_INDICES[0];
  const startPoint = keypoints[startIdx];
  ctx.moveTo(startPoint.x, startPoint.y);

  OUTER_RING_INDICES.slice(1).forEach((idx) => {
    const point = keypoints[idx];
    ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  ctx.stroke();

  // Draw keypoints
  drawKeypoints(ctx, keypoints);
}

function setupThreeJSScenes(img, keypoints) {
  const verticesData = getVerticesData(img, keypoints);

  // Flat wireframe scene
  const flatWireframeScene = createThreeJSScene(
    "flatWireframeContainer",
    verticesData.flatVertices,
    verticesData.uvCoordinates,
    verticesData.indices,
    { color: 0xff0000, wireframe: true },
    true
  );

  // 3D wireframe scene
  const wireframeScene = createThreeJSScene(
    "wireframeContainer",
    verticesData.vertices,
    verticesData.uvCoordinates,
    verticesData.indices,
    { color: 0x00ff00, wireframe: true },
    true
  );

  // Textured 3D scene
  const texture = new THREE.Texture(img);
  texture.needsUpdate = true;

  const texturedScene = createThreeJSScene(
    "texturedContainer",
    verticesData.vertices,
    verticesData.uvCoordinates,
    verticesData.indices,
    { map: texture, side: THREE.DoubleSide },
    false
  );

  // Animate scenes
  animateScenes([flatWireframeScene, wireframeScene, texturedScene]);
}

function getVerticesData(img, keypoints) {
  const vertices = [];
  const flatVertices = [];
  const uvCoordinates = [];
  const depthScale = 1;

  keypoints.forEach(({ x, y, z }) => {
    vertices.push(x - img.width / 2, -y + img.height / 2, z * depthScale);
    flatVertices.push(x - img.width / 2, -y + img.height / 2, 0);
    uvCoordinates.push(x / img.width, 1 - y / img.height);
  });

  const indices = [];
  for (let i = 0; i < TRIANGULATION.length; i += 3) {
    indices.push(TRIANGULATION[i], TRIANGULATION[i + 1], TRIANGULATION[i + 2]);
  }

  return { vertices, flatVertices, uvCoordinates, indices };
}

function createThreeJSScene(
  containerId,
  positions,
  uvs,
  indices,
  materialOptions,
  isWireframe
) {
  const container = document.getElementById(containerId);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
  camera.position.z = 500;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(400, 400);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.25;
  controls.enableZoom = true;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);

  let material;
  if (materialOptions.map) {
    material = new THREE.MeshBasicMaterial(materialOptions);
  } else {
    material = new THREE.MeshBasicMaterial(materialOptions);
  }

  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  return { scene, camera, renderer, controls };
}

function animateScenes(scenes) {
  function animate() {
    requestAnimationFrame(animate);
    scenes.forEach(({ renderer, scene, camera, controls }) => {
      controls.update();
      renderer.render(scene, camera);
    });
  }
  animate();
}

init();
