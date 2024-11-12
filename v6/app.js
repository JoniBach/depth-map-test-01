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
  await generateDepthMapCanvas(image, appState.estimator);
  const depthTexture = getDepthTextureFromCanvas();
  await appState.threeDScene.updateMesh(
    depthTexture,
    image.width,
    image.height,
    image
  );
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
 * Generates the depth map canvas and initializes painting tools.
 * @param {HTMLImageElement} image - The uploaded image.
 * @param {Object} estimator - The depth estimator model.
 */
async function generateDepthMapCanvas(image, estimator) {
  const depthMap = await estimator.estimateDepth(image, {
    minDepth: 0,
    maxDepth: 1,
  });
  const depthMapCanvas = document.getElementById("depthMapCanvas");
  const depthMapCtx = depthMapCanvas.getContext("2d");
  depthMapCanvas.width = image.width;
  depthMapCanvas.height = image.height;

  // Draw the estimated depth map onto the canvas
  const depthImage = await depthMap.toCanvasImageSource();
  depthMapCtx.drawImage(depthImage, 0, 0, image.width, image.height);

  // Initialize painting tools
  initializePaintingTools();
}

/**
 * Creates a depth texture from the depth map canvas.
 * @returns {THREE.CanvasTexture} The depth texture.
 */
function getDepthTextureFromCanvas() {
  const depthMapCanvas = document.getElementById("depthMapCanvas");
  const depthTexture = new THREE.CanvasTexture(depthMapCanvas);
  depthTexture.minFilter = THREE.LinearFilter;
  depthTexture.magFilter = THREE.LinearFilter;
  return depthTexture;
}

/**
 * Initializes the painting tools on the depth map canvas.
 */
// In the initializePaintingTools function
function initializePaintingTools() {
  const depthMapCanvas = document.getElementById("depthMapCanvas");
  const depthMapCtx = depthMapCanvas.getContext("2d");
  let painting = false;
  let tool = document.getElementById("toolSelect").value;
  let brushSize = parseInt(document.getElementById("brushSize").value);
  let opacity = parseFloat(document.getElementById("brushOpacity").value);
  let feather = parseInt(document.getElementById("brushFeather").value);

  // Update tool, brush size, opacity, and feather based on user input
  document.getElementById("toolSelect").addEventListener("change", function () {
    tool = this.value;
  });

  document.getElementById("brushSize").addEventListener("input", function () {
    brushSize = parseInt(this.value);
  });

  document
    .getElementById("brushOpacity")
    .addEventListener("input", function () {
      opacity = parseFloat(this.value);
    });

  document
    .getElementById("brushFeather")
    .addEventListener("input", function () {
      feather = parseInt(this.value);
    });

  // Event listeners for painting
  depthMapCanvas.addEventListener("mousedown", function (e) {
    painting = true;
    draw(e);
  });

  depthMapCanvas.addEventListener("mouseup", function () {
    painting = false;
    updateDepthTexture();
  });

  depthMapCanvas.addEventListener("mouseleave", function () {
    painting = false;
  });

  depthMapCanvas.addEventListener("mousemove", function (e) {
    if (painting) {
      draw(e);
    }
  });

  depthMapCanvas.addEventListener("touchstart", function (e) {
    painting = true;
    draw(e.touches[0]);
  });

  depthMapCanvas.addEventListener("touchend", function () {
    painting = false;
    updateDepthTexture();
  });

  depthMapCanvas.addEventListener("touchmove", function (e) {
    if (painting) {
      draw(e.touches[0]);
    }
  });

  // Cursor ring element
  const cursorRing = document.getElementById("cursorRing");

  /**
   * Updates the cursor ring based on tool, brush size, opacity, and feather.
   */
  function updateCursorRing() {
    const brushSize = parseInt(document.getElementById("brushSize").value);
    const feather = parseInt(document.getElementById("brushFeather").value);
    const tool = document.getElementById("toolSelect").value;
    const opacity = parseFloat(document.getElementById("brushOpacity").value);

    // Update cursor ring size and opacity
    cursorRing.style.width = `${brushSize}px`;
    cursorRing.style.height = `${brushSize}px`;
    cursorRing.style.borderColor =
      tool === "eraser" ? "red" : "rgba(255, 255, 255, 0.8)";
    cursorRing.style.boxShadow = `0 0 ${feather * 2}px rgba(255, 255, 255, ${
      opacity * 0.5
    })`;
  }

  // Call updateCursorRing whenever brush settings change
  document
    .getElementById("brushSize")
    .addEventListener("input", updateCursorRing);
  document
    .getElementById("brushOpacity")
    .addEventListener("input", updateCursorRing);
  document
    .getElementById("brushFeather")
    .addEventListener("input", updateCursorRing);
  document
    .getElementById("toolSelect")
    .addEventListener("change", updateCursorRing);

  /**
   * Position the cursor ring over the depthMapCanvas following mouse movements.
   */
  depthMapCanvas.addEventListener("mousemove", (e) => {
    cursorRing.style.display = "block"; // Show the ring
    const rect = depthMapCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    cursorRing.style.left = `${x + rect.left}px`;
    cursorRing.style.top = `${y + rect.top}px`;
  });

  /**
   * Hide the cursor ring when the mouse leaves the canvas.
   */
  depthMapCanvas.addEventListener("mouseleave", () => {
    cursorRing.style.display = "none"; // Hide the ring
  });

  // Drawing function with opacity and feather adjustments
  // Simple blur function
  function blurArea(x, y, size) {
    const startX = Math.max(0, x - size / 2);
    const startY = Math.max(0, y - size / 2);
    const width = Math.min(size, depthMapCanvas.width - startX);
    const height = Math.min(size, depthMapCanvas.height - startY);

    const imageData = depthMapCtx.getImageData(startX, startY, width, height);
    const data = imageData.data;
    let sum = 0;

    // Calculate the average grayscale value
    for (let i = 0; i < data.length; i += 4) {
      const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
      sum += brightness;
    }
    const avg = sum / (data.length / 4);

    // Set the pixels to the average value
    for (let i = 0; i < data.length; i += 4) {
      data[i] = avg;
      data[i + 1] = avg;
      data[i + 2] = avg;
    }

    depthMapCtx.putImageData(imageData, startX, startY);
  }

  // Update the depth texture in the Three.js scene
  function updateDepthTexture() {
    const depthTexture = getDepthTextureFromCanvas();
    appState.threeDScene.updateDepthTexture(depthTexture);
  }
  function draw(e) {
    const rect = depthMapCanvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * depthMapCanvas.width) / rect.width;
    const y = ((e.clientY - rect.top) * depthMapCanvas.height) / rect.height;

    if (tool === "brush") {
      depthMapCtx.beginPath();
      depthMapCtx.arc(x, y, brushSize / 2, 0, 2 * Math.PI);

      // Apply opacity
      depthMapCtx.globalAlpha = opacity;

      // Calculate inner and outer radii for feathering
      const innerRadius = Math.max(0, brushSize / 2 - feather);
      const outerRadius = brushSize / 2;

      // Create radial gradient for feathering effect
      const gradient = depthMapCtx.createRadialGradient(
        x,
        y,
        innerRadius,
        x,
        y,
        outerRadius
      );
      gradient.addColorStop(0, "white");
      gradient.addColorStop(1, "rgba(255, 255, 255, 0)"); // Transparent at the edge

      depthMapCtx.fillStyle = gradient;
      depthMapCtx.fill();

      // Reset global alpha to default
      depthMapCtx.globalAlpha = 1.0;
    } else if (tool === "eraser") {
      depthMapCtx.beginPath();
      depthMapCtx.arc(x, y, brushSize / 2, 0, 2 * Math.PI);
      depthMapCtx.fillStyle = "black";
      depthMapCtx.fill();
    } else if (tool === "blur") {
      blurArea(x, y, brushSize);
    }
  }

  function updateDepthTexture() {
    const depthTexture = getDepthTextureFromCanvas();
    appState.threeDScene.updateDepthTexture(depthTexture);
  }
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
    this.currentUpdateId = 0;
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
    this.camera.position.z = 2;

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
    const updateId = ++this.currentUpdateId;

    const aspectRatio = width / height;
    const planeWidth = 1.2;
    const planeHeight = planeWidth / aspectRatio;
    const segments = 256;

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
      displacementScale: -0.2,
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
      this.mesh.material.displacementScale = -scale * 0.2;
      this.mesh.material.needsUpdate = true;
    }
  }

  /**
   * Updates the depth texture of the mesh material.
   * @param {THREE.CanvasTexture} depthTexture - The new depth texture.
   */
  updateDepthTexture(depthTexture) {
    if (this.mesh) {
      this.mesh.material.displacementMap = depthTexture;
      this.mesh.material.alphaMap = depthTexture;
      this.mesh.material.needsUpdate = true;
    }
  }
}

// Start the application
initializeApp();
