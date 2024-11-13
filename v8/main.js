// app.js

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as depthEstimation from "@tensorflow-models/depth-estimation";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import TRIANGULATION from "./TRIANGULATION.json";

/**
 * Outer ring landmark indices for facial landmarks.
 * @constant {number[]}
 */
const OUTER_RING_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109,
];

/**
 * Initializes the application.
 */
async function init() {
  try {
    // Load models
    const [detector, depthEstimator] = await loadModels();

    // Load image
    const image = await loadImage("inputImage");

    // Process image
    const { depthImage, predictions } = await processImage(
      image,
      detector,
      depthEstimator
    );

    if (predictions.length === 0) {
      console.log("No faces detected.");
      return;
    }

    const keypoints = predictions[0].keypoints;

    // Draw overlays
    drawOverlays(image, keypoints, depthImage);

    // Set up Three.js scenes
    setupThreeJSScenes(image, keypoints);
  } catch (error) {
    console.error("An error occurred:", error);
  }
}

/**
 * Loads the face detection and depth estimation models.
 * @returns {Promise<[faceLandmarksDetection.FaceLandmarksDetector, depthEstimation.DepthEstimator]>}
 */
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

/**
 * Loads an image element by its ID.
 * @param {string} imageId - The ID of the image element.
 * @returns {Promise<HTMLImageElement>}
 */
function loadImage(imageId) {
  return new Promise((resolve, reject) => {
    const img = document.getElementById(imageId);
    if (!img) {
      reject(new Error(`Image element with ID "${imageId}" not found.`));
    } else if (img.complete && img.naturalHeight !== 0) {
      resolve(img);
    } else {
      img.onload = () => resolve(img);
      img.onerror = reject;
    }
  });
}

/**
 * Processes the image to obtain depth information and facial landmarks.
 * @param {HTMLImageElement} img - The image to process.
 * @param {faceLandmarksDetection.FaceLandmarksDetector} detector - The face landmarks detector.
 * @param {depthEstimation.DepthEstimator} depthEstimator - The depth estimator.
 * @returns {Promise<{ depthImage: CanvasImageSource, predictions: Array }>}
 */
async function processImage(img, detector, depthEstimator) {
  if (!img || !detector || !depthEstimator) {
    throw new Error("Invalid arguments passed to processImage.");
  }

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

/**
 * Retrieves canvas contexts for the specified canvas element IDs.
 * @param {string[]} canvasIds - Array of canvas element IDs.
 * @returns {Object} - An object mapping canvas IDs to their 2D contexts.
 */
function getCanvasContexts(canvasIds) {
  const contexts = {};
  canvasIds.forEach((id) => {
    const canvas = document.getElementById(id);
    if (canvas) {
      contexts[id] = canvas.getContext("2d");
    } else {
      console.warn(`Canvas element with ID "${id}" not found.`);
    }
  });
  return contexts;
}

/**
 * Draws various overlays on the image using the provided keypoints and depth image.
 * @param {HTMLImageElement} image - The original image.
 * @param {Array} keypoints - Array of facial landmark keypoints.
 * @param {CanvasImageSource} depthImage - The depth image.
 */
function drawOverlays(image, keypoints, depthImage) {
  const canvasIds = [
    "outputCanvas",
    "outerRingCanvas",
    "maskedDepthCanvas",
    "invertedDepthCanvas",
    "invertedMaskedDepthCanvas",
    "triangulationCanvas",
    "combinedOverlayCanvas",
    "depthCanvas",
  ];

  const contexts = getCanvasContexts(canvasIds);

  // Draw the original image on the output canvas
  contexts.outputCanvas.drawImage(
    image,
    0,
    0,
    contexts.outputCanvas.canvas.width,
    contexts.outputCanvas.canvas.height
  );

  // Draw keypoints
  drawKeypoints(contexts.outputCanvas, keypoints);

  // Draw outer ring
  drawOuterRing(contexts.outerRingCanvas, image, keypoints);

  // Draw depth map
  drawDepthMap(contexts.depthCanvas, depthImage);

  // Draw masked depth map
  drawMaskedDepthMap(contexts.maskedDepthCanvas, depthImage, keypoints);

  // Invert depth map
  invertCanvasImage(contexts.invertedDepthCanvas, depthImage);

  // Invert masked depth map
  invertCanvasImage(
    contexts.invertedMaskedDepthCanvas,
    contexts.maskedDepthCanvas.canvas
  );

  // Draw triangulation overlay
  drawTriangulation(contexts.triangulationCanvas, image, keypoints);

  // Draw combined overlay
  drawCombinedOverlay(contexts.combinedOverlayCanvas, image, keypoints);
}

/**
 * Draws facial keypoints on a canvas context.
 * @param {CanvasRenderingContext2D} ctx - The canvas context.
 * @param {Array} keypoints - Array of keypoint objects with x and y properties.
 * @param {string} [color='red'] - The color of the keypoints.
 * @param {number} [radius=2] - The radius of each keypoint.
 */
function drawKeypoints(ctx, keypoints, color = "red", radius = 2) {
  ctx.fillStyle = color;
  keypoints.forEach(({ x, y }) => {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
  });
}

/**
 * Creates the outer ring path on a canvas context based on keypoints.
 * @param {CanvasRenderingContext2D} ctx - The canvas context.
 * @param {Array} keypoints - Array of facial landmark keypoints.
 */
function createOuterRingPath(ctx, keypoints) {
  const startIdx = OUTER_RING_INDICES[0];
  const startPoint = keypoints[startIdx];
  ctx.beginPath();
  ctx.moveTo(startPoint.x, startPoint.y);

  OUTER_RING_INDICES.slice(1).forEach((idx) => {
    const point = keypoints[idx];
    ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
}

/**
 * Draws the outer ring of facial landmarks on a canvas context.
 * @param {CanvasRenderingContext2D} ctx - The canvas context.
 * @param {HTMLImageElement} image - The original image.
 * @param {Array} keypoints - Array of facial landmark keypoints.
 * @param {string} [strokeStyle='blue'] - The stroke color.
 * @param {number} [lineWidth=2] - The line width.
 * @param {boolean} [drawImageFirst=true] - Whether to draw the image before drawing the outer ring.
 */
function drawOuterRing(
  ctx,
  image,
  keypoints,
  strokeStyle = "blue",
  lineWidth = 2,
  drawImageFirst = true
) {
  if (drawImageFirst) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(image, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;

  createOuterRingPath(ctx, keypoints);
  ctx.stroke();
}

/**
 * Draws the depth map onto a canvas context.
 * @param {CanvasRenderingContext2D} ctx - The canvas context.
 * @param {CanvasImageSource} depthImage - The depth image.
 */
function drawDepthMap(ctx, depthImage) {
  ctx.drawImage(depthImage, 0, 0, ctx.canvas.width, ctx.canvas.height);
}

/**
 * Draws the masked depth map using the outer ring path as a clipping mask.
 * @param {CanvasRenderingContext2D} ctx - The canvas context.
 * @param {CanvasImageSource} depthImage - The depth image.
 * @param {Array} keypoints - Array of facial landmark keypoints.
 */
function drawMaskedDepthMap(ctx, depthImage, keypoints) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.save();

  createOuterRingPath(ctx, keypoints);
  ctx.clip();

  ctx.drawImage(depthImage, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();
}

/**
 * Inverts the colors of an image or canvas source and draws it onto a canvas context.
 * @param {CanvasRenderingContext2D} ctx - The canvas context.
 * @param {CanvasImageSource} source - The source image or canvas.
 */
function invertCanvasImage(ctx, source) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(source, 0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.globalCompositeOperation = "difference";
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.globalCompositeOperation = "source-over";
}

/**
 * Draws the facial triangulation overlay on a canvas context.
 * @param {CanvasRenderingContext2D} ctx - The canvas context.
 * @param {HTMLImageElement} image - The original image.
 * @param {Array} keypoints - Array of facial landmark keypoints.
 * @param {string} [strokeStyle='green'] - The stroke color.
 * @param {number} [lineWidth=1] - The line width.
 * @param {boolean} [drawImageFirst=true] - Whether to draw the image before drawing the triangulation.
 */
function drawTriangulation(
  ctx,
  image,
  keypoints,
  strokeStyle = "green",
  lineWidth = 1,
  drawImageFirst = true
) {
  if (drawImageFirst) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(image, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;

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

/**
 * Draws a combined overlay of the triangulation, outer ring, and keypoints on a canvas context.
 * @param {CanvasRenderingContext2D} ctx - The canvas context.
 * @param {HTMLImageElement} image - The original image.
 * @param {Array} keypoints - Array of facial landmark keypoints.
 */
function drawCombinedOverlay(ctx, image, keypoints) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(image, 0, 0, ctx.canvas.width, ctx.canvas.height);

  // Draw triangulation without redrawing the image
  drawTriangulation(ctx, image, keypoints, "green", 1, false);

  // Draw outer ring without redrawing the image
  drawOuterRing(ctx, image, keypoints, "blue", 2, false);

  // Draw keypoints
  drawKeypoints(ctx, keypoints, "red", 2);
}

/**
 * Sets up the Three.js scenes for rendering 3D models.
 * @param {HTMLImageElement} image - The original image.
 * @param {Array} keypoints - Array of facial landmark keypoints.
 */
function setupThreeJSScenes(image, keypoints) {
  const verticesData = getVerticesData(image, keypoints);

  // 3D wireframe scene
  const wireframeScene = createThreeJSScene(
    "wireframeContainer",
    verticesData.vertices,
    verticesData.uvCoordinates,
    verticesData.indices,
    { color: 0x00ff00, wireframe: true }
  );

  // Textured 3D scene
  const texture = new THREE.Texture(image);
  texture.needsUpdate = true;

  const texturedScene = createThreeJSScene(
    "texturedContainer",
    verticesData.vertices,
    verticesData.uvCoordinates,
    verticesData.indices,
    { map: texture, side: THREE.DoubleSide }
  );

  // Point Cloud Scene
  const pointCloudScene = createPointCloudScene(
    "pointCloudContainer",
    verticesData.vertices
  );

  // Frame Rim Scene
  const frameRimScene = createFrameRimScene(
    "frameRimContainer",
    verticesData.vertices,
    OUTER_RING_INDICES
  );

  // Animate scenes
  animateScenes([
    wireframeScene,
    texturedScene,
    pointCloudScene,
    frameRimScene,
  ]);
}

/**
 * Generates vertex data for 3D rendering based on keypoints.
 * @param {HTMLImageElement} image - The original image.
 * @param {Array} keypoints - Array of facial landmark keypoints.
 * @param {number} [depthScale=1] - Scale factor for the depth (z-axis).
 * @returns {Object} - An object containing vertices, flatVertices, uvCoordinates, and indices.
 */
function getVerticesData(image, keypoints, depthScale = 1) {
  const vertices = [];
  const flatVertices = [];
  const uvCoordinates = [];

  keypoints.forEach(({ x, y, z }) => {
    vertices.push(x - image.width / 2, -y + image.height / 2, z * depthScale);
    flatVertices.push(x - image.width / 2, -y + image.height / 2, 0);
    uvCoordinates.push(x / image.width, 1 - y / image.height);
  });

  const indices = [];
  for (let i = 0; i < TRIANGULATION.length; i += 3) {
    indices.push(TRIANGULATION[i], TRIANGULATION[i + 1], TRIANGULATION[i + 2]);
  }

  return { vertices, flatVertices, uvCoordinates, indices };
}

/**
 * Creates a Three.js scene with a mesh constructed from provided geometry and material options.
 * @param {string} containerId - The ID of the container element.
 * @param {number[]} positions - Vertex positions.
 * @param {number[]} uvs - UV texture coordinates.
 * @param {number[]} indices - Indices for the mesh.
 * @param {Object} materialOptions - Options for the material.
 * @returns {Object} - An object containing the scene, camera, renderer, and controls.
 */
function createThreeJSScene(
  containerId,
  positions,
  uvs,
  indices,
  materialOptions
) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.warn(`Container element with ID "${containerId}" not found.`);
    return null;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
  camera.position.z = 500;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(400, 400);
  container.appendChild(renderer.domElement);
  renderer.setClearColor(0xffffff, 0);
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

  const material = new THREE.MeshBasicMaterial(materialOptions);

  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  return { scene, camera, renderer, controls };
}

/**
 * Creates a Three.js scene displaying a 3D point cloud.
 * @param {string} containerId - The ID of the container element.
 * @param {number[]} positions - Vertex positions.
 * @returns {Object} - An object containing the scene, camera, renderer, and controls.
 */
function createPointCloudScene(containerId, positions) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.warn(`Container element with ID "${containerId}" not found.`);
    return null;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
  camera.position.z = 500;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(400, 400);
  renderer.setClearColor(0xffffff, 0);
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

  // Increased point size from 2 to 5
  const material = new THREE.PointsMaterial({ color: 0xff0000, size: 5 });

  // Optional: Disable size attenuation if you want consistent point sizes
  // const material = new THREE.PointsMaterial({ color: 0xff0000, size: 5, sizeAttenuation: false });

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  return { scene, camera, renderer, controls };
}

function createFrameRimScene(containerId, positions, ringIndices) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.warn(`Container element with ID "${containerId}" not found.`);
    return null;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
  camera.position.z = 500;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(400, 400);
  container.appendChild(renderer.domElement);
  renderer.setClearColor(0xffffff, 0);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.25;
  controls.enableZoom = true;

  // Extract the positions for the outer ring
  const ringPositions = [];
  ringIndices.forEach((idx) => {
    ringPositions.push(
      positions[idx * 3],
      positions[idx * 3 + 1],
      positions[idx * 3 + 2]
    );
  });

  // Create a closed loop by adding the first point at the end
  ringPositions.push(ringPositions[0], ringPositions[1], ringPositions[2]);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(ringPositions, 3)
  );

  // Change the color from yellow (0xffff00) to blue (0x0000ff)
  const material = new THREE.LineBasicMaterial({ color: 0x0000ff });
  const line = new THREE.LineLoop(geometry, material);
  scene.add(line);

  return { scene, camera, renderer, controls };
}

/**
 * Animates multiple Three.js scenes.
 * @param {Array} scenes - Array of scene objects containing renderer, scene, camera, and controls.
 */
function animateScenes(scenes) {
  function animate() {
    requestAnimationFrame(animate);
    scenes.forEach((sceneObj) => {
      if (sceneObj) {
        const { renderer, scene, camera, controls } = sceneObj;
        controls.update();
        renderer.render(scene, camera);
      }
    });
  }
  animate();
}

init();
