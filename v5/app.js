// app.js

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-converter";
import "@tensorflow/tfjs-backend-webgl";
import * as depthEstimation from "@tensorflow-models/depth-estimation";

/**
 * Application state to keep track of the estimator, scene, and current image.
 */
const appState = {
  estimator: null,
  threeDScene: null,
  currentImage: null,
};

/**
 * Initializes the application by loading the depth estimator and setting up event listeners.
 */
async function initializeApp() {
  appState.estimator = await loadDepthEstimator();
  appState.threeDScene = new ThreeDScene();

  document
    .getElementById("imageUpload")
    .addEventListener("change", (event) => handleImageUpload(event));
  document
    .getElementById("toneRangeSlider")
    .addEventListener("input", () => handleToneRangeChange());
  document
    .getElementById("depthSlider")
    .addEventListener("input", (event) => handleDepthSliderChange(event));
}

/**
 * Loads the depth estimation model using TensorFlow.js.
 * @returns {Promise<Object>} The depth estimator model.
 */
async function loadDepthEstimator() {
  await tf.setBackend("webgl");
  tf.env().set("WEBGL_CPU_FORWARD", false);
  const model = depthEstimation.SupportedModels.ARPortraitDepth;
  const estimatorConfig = { outputDepthRange: [0, 1] };
  return await depthEstimation.createEstimator(model, estimatorConfig);
}

/**
 * Handles the image upload event.
 * @param {Event} event - The file input change event.
 */
function handleImageUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const image = new Image();
  image.src = URL.createObjectURL(file);
  image.onload = async () => {
    document.getElementById("imagePreview").src = image.src;
    appState.currentImage = image;
    await updateSceneWithImage(image);
  };
}

/**
 * Updates the scene with the current image and depth texture.
 * @param {HTMLImageElement} image - The uploaded image.
 */
async function updateSceneWithImage(image) {
  const toneRange = parseInt(document.getElementById("toneRangeSlider").value);
  const depthTexture = await generateDepthTexture(
    image,
    appState.estimator,
    toneRange
  );
  await appState.threeDScene.updateMesh(
    depthTexture,
    image.width,
    image.height,
    image
  );
}

/**
 * Handles the tone range slider input event.
 */
async function handleToneRangeChange() {
  const image = appState.currentImage;
  if (!image) return;
  await updateSceneWithImage(image);
}

/**
 * Handles the depth slider input event.
 * @param {Event} event - The slider input event.
 */
function handleDepthSliderChange(event) {
  const depthScale = parseFloat(event.target.value);
  appState.threeDScene.updateDisplacementScale(depthScale);
}

/**
 * Generates a depth texture from the image and estimator.
 * @param {HTMLImageElement} image - The uploaded image.
 * @param {Object} estimator - The depth estimator model.
 * @param {number} toneRange - The tone range value from the slider.
 * @returns {Promise<THREE.CanvasTexture>} The depth texture.
 */
async function generateDepthTexture(image, estimator, toneRange) {
  const depthMap = await estimator.estimateDepth(image, {
    minDepth: 0,
    maxDepth: 1,
  });
  return await createDepthTexture(
    depthMap,
    image.width,
    image.height,
    toneRange
  );
}

/**
 * Creates a depth texture by processing the depth map.
 * @param {Object} depthMap - The depth map estimated from the image.
 * @param {number} width - The width of the image.
 * @param {number} height - The height of the image.
 * @param {number} toneRange - The tone range value from the slider.
 * @returns {Promise<THREE.CanvasTexture>} The processed depth texture.
 */
async function createDepthTexture(depthMap, width, height, toneRange) {
  const depthCanvas = document.createElement("canvas");
  const ctx = depthCanvas.getContext("2d");

  depthCanvas.width = width;
  depthCanvas.height = height;

  // Await the Promise returned by toCanvasImageSource()
  const depthImage = await depthMap.toCanvasImageSource();
  ctx.drawImage(depthImage, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const depthThreshold = toneRange * 3;

  for (let i = 0; i < data.length; i += 4) {
    const brightness = data[i] + data[i + 1] + data[i + 2];
    data[i + 3] = brightness < depthThreshold ? 0 : 255;
  }

  ctx.putImageData(imageData, 0, 0);

  // Optional: Display the depth map on the canvas for debugging
  const depthMapCanvas = document.getElementById("depthMapCanvas");
  if (depthMapCanvas) {
    const depthMapCtx = depthMapCanvas.getContext("2d");
    depthMapCanvas.width = width;
    depthMapCanvas.height = height;
    depthMapCtx.drawImage(depthCanvas, 0, 0);
  }

  return new THREE.CanvasTexture(depthCanvas);
}

/**
 * Class to manage the Three.js scene, including camera, controls, and rendering.
 */
class ThreeDScene {
  constructor() {
    this.renderer = new THREE.WebGLRenderer({ alpha: true });
    this.renderer.setClearColor(0x000000, 0); // Transparent background
    this.scene = new THREE.Scene();
    this.camera = null;
    this.controls = null;
    this.mesh = null;
    this.currentUpdateId = 0; // Add this property
    this.initializeScene();
  }

  /**
   * Initializes the Three.js scene, camera, and controls.
   */
  initializeScene() {
    const container = document.getElementById("threeContainer");
    const width = container.clientWidth || 500;
    const height = container.clientHeight || 500;
    this.renderer.setSize(width, height);
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    this.camera.position.z = 2; // Adjusted for better initial view

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(0, 1, 1).normalize();
    this.scene.add(directionalLight);

    this.animate();
  }

  /**
   * Animation loop to render the scene.
   */
  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Updates the mesh in the scene with the new depth texture.
   * @param {THREE.CanvasTexture} depthTexture - The depth texture.
   * @param {number} width - The image width.
   * @param {number} height - The image height.
   * @param {HTMLImageElement} image - The original image.
   */
  async updateMesh(depthTexture, width, height, image) {
    const updateId = ++this.currentUpdateId; // Increment and capture the update ID

    const aspectRatio = width / height;
    const planeWidth = 1.2;
    const planeHeight = planeWidth / aspectRatio;
    const segments = 256; // Increased for smoother mesh

    // Remove the existing mesh
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.mesh = null;
    }

    const geometry = new THREE.PlaneGeometry(
      planeWidth,
      planeHeight,
      segments,
      segments
    );

    const textureLoader = new THREE.TextureLoader();

    // Use loadAsync to await the texture loading
    const texture = await textureLoader.loadAsync(image.src);

    // After async operations, check if this update is still the latest
    if (updateId !== this.currentUpdateId) {
      // This update has been superseded by a newer one, so exit
      geometry.dispose();
      texture.dispose();
      return;
    }

    const material = new THREE.MeshStandardMaterial({
      map: texture,
      displacementMap: depthTexture,
      displacementScale: -0.2, // Adjusted for better depth effect
      alphaMap: depthTexture,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.mesh);
  }

  /**
   * Updates the displacement scale of the mesh.
   * @param {number} scale - The new displacement scale value.
   */
  updateDisplacementScale(scale) {
    if (this.mesh) {
      this.mesh.material.displacementScale = -scale * 0.2; // Adjusted scale factor
      this.mesh.material.needsUpdate = true;
    }
  }
}

// Start the application
initializeApp();
