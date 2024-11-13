// app.js

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as depthEstimation from "@tensorflow-models/depth-estimation";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import TRIANGULATION from "./TRIANGULATION.json";

/**
 * Generates the configuration for application models, scenes, and overlays based on image dimensions.
 * @param {Object} imageInputDimensions - The dimensions of the input image.
 * @param {number} imageInputDimensions.width - The width of the input image.
 * @param {number} imageInputDimensions.height - The height of the input image.
 * @returns {Object} The complete configuration object.
 */
const chartConfiguration = (imageInputDimensions) => ({
  modelConfig: {
    detector: {
      modelType: faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
      runtime: "tfjs",
      maxFaces: 1,
      refineLandmarks: false,
    },
    depthEstimator: {
      modelType: depthEstimation.SupportedModels.ARPortraitDepth,
      outputDepthRange: [0, 1],
    },
    depthEstimationRange: {
      minDepth: 0,
      maxDepth: 1,
    },
  },
  overlayStyles: {
    keypoint: {
      color: "red",
      radius: 2,
    },
    outerRing: {
      color: "blue",
      lineWidth: 2,
    },
    depthMap: {
      inverted: {
        color: "white",
      },
    },
    triangulation: {
      color: "green",
      lineWidth: 1,
    },
  },
  threeJSConfig: {
    camera: {
      fieldOfView: 75,
      aspectRatio: imageInputDimensions.width / imageInputDimensions.height,
      nearClip: 0.1,
      farClip: 1000,
      positionZ: 500,
    },
    renderer: {
      width: imageInputDimensions.width,
      height: imageInputDimensions.height,
      backgroundColor: 0xffffff,
    },
    controls: {
      enableDamping: true,
      dampingFactor: 0.25,
      enableZoom: true,
    },
    pointCloud: {
      color: 0xff0000,
      size: 5,
    },
    frameRim: {
      color: 0x0000ff,
    },
  },
  landmarkIndices: {
    outerRing: [
      10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
      378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
      162, 21, 54, 103, 67, 109,
    ],
  },
  canvasIds: [
    "outputCanvas",
    "outerRingCanvas",
    "maskedDepthCanvas",
    "invertedDepthCanvas",
    "invertedMaskedDepthCanvas",
    "triangulationCanvas",
    "combinedOverlayCanvas",
    "depthCanvas",
  ],
});

/**
 * Initializes the application by loading models, processing images, drawing overlays, and setting up 3D scenes.
 * @async
 * @function init
 * @returns {Promise<void>}
 */
async function init() {
  try {
    const image = await loadImage("inputImage");
    const imageInputDimensions = {
      width: image.width,
      height: image.height,
    };
    const config = chartConfiguration(imageInputDimensions);

    const [detector, depthEstimator] = await loadModels(config);

    const { depthImage, predictions } = await processImage(
      image,
      detector,
      depthEstimator,
      config
    );

    if (predictions.length === 0) {
      console.log("No faces detected.");
      return;
    }

    const keypoints = predictions[0].keypoints;

    drawOverlays(image, keypoints, depthImage, config);

    setupThreeJSScenes(image, keypoints, config);
  } catch (error) {
    console.error("An error occurred:", error);
  }
}

/**
 * Loads the face detection and depth estimation models based on the provided configuration.
 * @async
 * @function loadModels
 * @param {Object} config - The configuration object for models.
 * @returns {Promise<Array>} An array containing the face detector and depth estimator models.
 */
async function loadModels(config) {
  const detector = await faceLandmarksDetection.createDetector(
    config.modelConfig.detector.modelType,
    config.modelConfig.detector
  );

  const depthEstimator = await depthEstimation.createEstimator(
    config.modelConfig.depthEstimator.modelType,
    config.modelConfig.depthEstimator
  );

  return [detector, depthEstimator];
}

/**
 * Loads an image element by its ID.
 * @function loadImage
 * @param {string} imageId - The ID of the image element to load.
 * @returns {Promise<HTMLImageElement>} A promise that resolves to the loaded image element.
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
 * Processes the input image using the detector and depth estimator models.
 * @async
 * @function processImage
 * @param {HTMLImageElement} img - The image to process.
 * @param {Object} detector - The face detector model.
 * @param {Object} depthEstimator - The depth estimator model.
 * @param {Object} config - The configuration object.
 * @returns {Promise<Object>} An object containing the depth image and face predictions.
 * @throws Will throw an error if invalid arguments are passed.
 */
async function processImage(img, detector, depthEstimator, config) {
  if (!img || !detector || !depthEstimator) {
    throw new Error("Invalid arguments passed to processImage.");
  }

  const depthMap = await depthEstimator.estimateDepth(
    img,
    config.modelConfig.depthEstimationRange
  );

  const depthImage = await depthMap.toCanvasImageSource();

  const predictions = await detector.estimateFaces(img, {
    flipHorizontal: false,
  });

  return { depthImage, predictions };
}

/**
 * Retrieves the 2D drawing contexts for a list of canvas elements.
 * @function getCanvasContexts
 * @param {Array<string>} canvasIds - An array of canvas element IDs.
 * @param {number} width - The width to set for each canvas.
 * @param {number} height - The height to set for each canvas.
 * @returns {Object} An object mapping canvas IDs to their 2D drawing contexts.
 */
function getCanvasContexts(canvasIds, width, height) {
  const contexts = {};
  canvasIds.forEach((id) => {
    const canvas = document.getElementById(id);
    if (canvas) {
      // Set canvas dimensions to match the image
      canvas.width = width;
      canvas.height = height;
      contexts[id] = canvas.getContext("2d");
    } else {
      console.warn(`Canvas element with ID "${id}" not found.`);
    }
  });
  return contexts;
}

/**
 * Draws various overlays on the image such as keypoints, outer ring, depth maps, and triangulation.
 * @function drawOverlays
 * @param {HTMLImageElement} image - The image on which to draw overlays.
 * @param {Array<Object>} keypoints - The facial keypoints.
 * @param {CanvasImageSource} depthImage - The depth image.
 * @param {Object} config - The configuration object.
 */
function drawOverlays(image, keypoints, depthImage, config) {
  const contexts = getCanvasContexts(
    config.canvasIds,
    image.width,
    image.height
  );

  // Draw the image correctly on each canvas to avoid zoom or distortion
  config.canvasIds.forEach((id) => {
    if (contexts[id]) {
      // Clear and set the background image on each canvas
      contexts[id].clearRect(
        0,
        0,
        contexts[id].canvas.width,
        contexts[id].canvas.height
      );
      contexts[id].drawImage(image, 0, 0, image.width, image.height);
    }
  });

  drawKeypoints(contexts.outputCanvas, keypoints, config);
  drawOuterRing(contexts.outerRingCanvas, image, keypoints, config);
  drawDepthMap(contexts.depthCanvas, depthImage, config);
  drawMaskedDepthMap(contexts.maskedDepthCanvas, depthImage, keypoints, config);
  invertCanvasImage(contexts.invertedDepthCanvas, depthImage, config);
  invertCanvasImage(
    contexts.invertedMaskedDepthCanvas,
    contexts.maskedDepthCanvas.canvas,
    config
  );
  drawTriangulation(contexts.triangulationCanvas, image, keypoints, config);
  drawCombinedOverlay(contexts.combinedOverlayCanvas, image, keypoints, config);
}

/**
 * Draws facial keypoints on the given canvas context.
 * @function drawKeypoints
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {Array<Object>} keypoints - The facial keypoints.
 * @param {Object} config - The configuration object.
 */
function drawKeypoints(ctx, keypoints, config) {
  const { color, radius } = config.overlayStyles.keypoint;
  ctx.fillStyle = color;
  keypoints.forEach(({ x, y }) => {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
  });
}

/**
 * Creates a path for the outer ring based on keypoints.
 * @function createOuterRingPath
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {Array<Object>} keypoints - The facial keypoints.
 * @param {Object} config - The configuration object.
 */
function createOuterRingPath(ctx, keypoints, config) {
  const startIdx = config.landmarkIndices.outerRing[0];
  const startPoint = keypoints[startIdx];
  ctx.beginPath();
  ctx.moveTo(startPoint.x, startPoint.y);

  config.landmarkIndices.outerRing.slice(1).forEach((idx) => {
    const point = keypoints[idx];
    ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
}

/**
 * Draws the outer ring around the face on the given canvas context.
 * @function drawOuterRing
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {HTMLImageElement} image - The image on which to draw.
 * @param {Array<Object>} keypoints - The facial keypoints.
 * @param {Object} config - The configuration object.
 */
function drawOuterRing(ctx, image, keypoints, config) {
  const { color, lineWidth } = config.overlayStyles.outerRing;

  // Clear the canvas and draw the image background first
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(image, 0, 0, image.width, image.height);

  // Set stroke style for the outer ring
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;

  // Draw the outer ring based on the configured indices
  ctx.beginPath();
  const startIdx = config.landmarkIndices.outerRing[0];
  const startPoint = keypoints[startIdx];
  ctx.moveTo(startPoint.x, startPoint.y);

  config.landmarkIndices.outerRing.slice(1).forEach((idx) => {
    const point = keypoints[idx];
    ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  ctx.stroke();
}

/**
 * Draws the depth map on the given canvas context.
 * @function drawDepthMap
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {CanvasImageSource} depthImage - The depth image.
 * @param {Object} config - The configuration object.
 */
function drawDepthMap(ctx, depthImage, config) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(depthImage, 0, 0, ctx.canvas.width, ctx.canvas.height);
}

/**
 * Draws the masked depth map within the outer ring on the given canvas context.
 * @function drawMaskedDepthMap
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {CanvasImageSource} depthImage - The depth image.
 * @param {Array<Object>} keypoints - The facial keypoints.
 * @param {Object} config - The configuration object.
 */
function drawMaskedDepthMap(ctx, depthImage, keypoints, config) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.save();

  createOuterRingPath(ctx, keypoints, config);
  ctx.clip();

  ctx.drawImage(depthImage, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();
}

/**
 * Inverts the colors of the image drawn on the canvas context.
 * @function invertCanvasImage
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {CanvasImageSource} source - The source image or canvas.
 * @param {Object} config - The configuration object.
 */
function invertCanvasImage(ctx, source, config) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(source, 0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.globalCompositeOperation = "difference";
  ctx.fillStyle = config.overlayStyles.depthMap.inverted.color;
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.globalCompositeOperation = "source-over";
}

/**
 * Draws the triangulation overlay on the given canvas context.
 * @function drawTriangulation
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {HTMLImageElement} image - The image on which to draw.
 * @param {Array<Object>} keypoints - The facial keypoints.
 * @param {Object} config - The configuration object.
 */
function drawTriangulation(ctx, image, keypoints, config) {
  const { color, lineWidth } = config.overlayStyles.triangulation;

  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(image, 0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.strokeStyle = color;
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
 * Draws a combined overlay of triangulation, outer ring, and keypoints on the given canvas context.
 * @function drawCombinedOverlay
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {HTMLImageElement} image - The image on which to draw.
 * @param {Array<Object>} keypoints - The facial keypoints.
 * @param {Object} config - The configuration object.
 */
function drawCombinedOverlay(ctx, image, keypoints, config) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Draw the background image first to prevent zooming issues
  ctx.drawImage(image, 0, 0, image.width, image.height);

  // Draw the triangulation overlay
  drawTriangulation(ctx, image, keypoints, config);

  // Draw the outer ring after triangulation to ensure correct layering
  drawOuterRing(ctx, image, keypoints, config);

  // Draw keypoints last to make sure they’re on top of other elements
  drawKeypoints(ctx, keypoints, config);
}

/**
 * Sets up the Three.js 3D scenes for wireframe, textured mesh, point cloud, and frame rim.
 * @function setupThreeJSScenes
 * @param {HTMLImageElement} image - The input image.
 * @param {Array<Object>} keypoints - The facial keypoints.
 * @param {Object} config - The configuration object.
 */
function setupThreeJSScenes(image, keypoints, config) {
  const verticesData = getVerticesData(image, keypoints);

  const wireframeScene = createThreeJSScene(
    "wireframeContainer",
    verticesData.vertices,
    verticesData.uvCoordinates,
    verticesData.indices,
    { color: config.overlayStyles.triangulation.color, wireframe: true },
    config
  );

  const texture = new THREE.Texture(image);
  texture.needsUpdate = true;

  const texturedScene = createThreeJSScene(
    "texturedContainer",
    verticesData.vertices,
    verticesData.uvCoordinates,
    verticesData.indices,
    { map: texture, side: THREE.DoubleSide },
    config
  );

  const pointCloudScene = createPointCloudScene(
    "pointCloudContainer",
    verticesData.vertices,
    config
  );

  const frameRimScene = createFrameRimScene(
    "frameRimContainer",
    verticesData.vertices,
    config.landmarkIndices.outerRing,
    config
  );

  animateScenes([
    wireframeScene,
    texturedScene,
    pointCloudScene,
    frameRimScene,
  ]);
}

/**
 * Generates vertices, UV coordinates, and indices for 3D rendering based on keypoints.
 * @function getVerticesData
 * @param {HTMLImageElement} image - The input image.
 * @param {Array<Object>} keypoints - The facial keypoints.
 * @param {number} [depthScale=1] - The scaling factor for depth (z-axis).
 * @returns {Object} An object containing vertices, flat vertices, UV coordinates, and indices.
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
 * Creates a Three.js scene with a mesh based on provided vertices and indices.
 * @function createThreeJSScene
 * @param {string} containerId - The ID of the HTML container element.
 * @param {Array<number>} positions - The vertex positions.
 * @param {Array<number>} uvs - The UV coordinates.
 * @param {Array<number>} indices - The vertex indices.
 * @param {Object} materialOptions - Options for the mesh material.
 * @param {Object} config - The configuration object.
 * @returns {Object|null} An object containing the scene, camera, renderer, and controls, or null if container not found.
 */
function createThreeJSScene(
  containerId,
  positions,
  uvs,
  indices,
  materialOptions,
  config
) {
  const container = document.getElementById(containerId);
  if (!container) return null;

  const scene = new THREE.Scene();
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(
    config.threeJSConfig.renderer.width,
    config.threeJSConfig.renderer.height
  );
  renderer.setClearColor(config.threeJSConfig.renderer.backgroundColor, 0);
  container.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(
    config.threeJSConfig.camera.fieldOfView,
    config.threeJSConfig.camera.aspectRatio,
    config.threeJSConfig.camera.nearClip,
    config.threeJSConfig.camera.farClip
  );
  camera.position.z = config.threeJSConfig.camera.positionZ;

  const controls = new OrbitControls(camera, renderer.domElement);
  Object.assign(controls, config.threeJSConfig.controls);

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
 * Creates a Three.js scene displaying a point cloud of the facial keypoints.
 * @function createPointCloudScene
 * @param {string} containerId - The ID of the HTML container element.
 * @param {Array<number>} positions - The vertex positions.
 * @param {Object} config - The configuration object.
 * @returns {Object|null} An object containing the scene, camera, renderer, and controls, or null if container not found.
 */
function createPointCloudScene(containerId, positions, config) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.warn(`Container element with ID "${containerId}" not found.`);
    return null;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    config.threeJSConfig.camera.fieldOfView,
    config.threeJSConfig.camera.aspectRatio,
    config.threeJSConfig.camera.nearClip,
    config.threeJSConfig.camera.farClip
  );
  camera.position.z = config.threeJSConfig.camera.positionZ;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(
    config.threeJSConfig.renderer.width,
    config.threeJSConfig.renderer.height
  );
  renderer.setClearColor(config.threeJSConfig.renderer.backgroundColor, 0);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  Object.assign(controls, config.threeJSConfig.controls);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );

  const material = new THREE.PointsMaterial(config.threeJSConfig.pointCloud);
  const points = new THREE.Points(geometry, material);
  scene.add(points);

  return { scene, camera, renderer, controls };
}

/**
 * Creates a Three.js scene displaying a frame rim around the face based on the outer ring keypoints.
 * @function createFrameRimScene
 * @param {string} containerId - The ID of the HTML container element.
 * @param {Array<number>} positions - The vertex positions.
 * @param {Array<number>} ringIndices - The indices for the outer ring keypoints.
 * @param {Object} config - The configuration object.
 * @returns {Object|null} An object containing the scene, camera, renderer, and controls, or null if container not found.
 */
function createFrameRimScene(containerId, positions, ringIndices, config) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.warn(`Container element with ID "${containerId}" not found.`);
    return null;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    config.threeJSConfig.camera.fieldOfView,
    config.threeJSConfig.camera.aspectRatio,
    config.threeJSConfig.camera.nearClip,
    config.threeJSConfig.camera.farClip
  );
  camera.position.z = config.threeJSConfig.camera.positionZ;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(
    config.threeJSConfig.renderer.width,
    config.threeJSConfig.renderer.height
  );
  container.appendChild(renderer.domElement);
  renderer.setClearColor(config.threeJSConfig.renderer.backgroundColor, 0);

  const controls = new OrbitControls(camera, renderer.domElement);
  Object.assign(controls, config.threeJSConfig.controls);

  const ringPositions = [];
  ringIndices.forEach((idx) => {
    ringPositions.push(
      positions[idx * 3],
      positions[idx * 3 + 1],
      positions[idx * 3 + 2]
    );
  });

  ringPositions.push(ringPositions[0], ringPositions[1], ringPositions[2]);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(ringPositions, 3)
  );

  const material = new THREE.LineBasicMaterial(config.threeJSConfig.frameRim);
  const line = new THREE.LineLoop(geometry, material);
  scene.add(line);

  return { scene, camera, renderer, controls };
}

/**
 * Animates the provided Three.js scenes by updating controls and rendering frames.
 * @function animateScenes
 * @param {Array<Object>} scenes - An array of scene objects containing scene, camera, renderer, and controls.
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

// Initialize the application
init();
