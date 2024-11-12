import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-converter";
import "@tensorflow/tfjs-backend-webgl";
import * as depthEstimation from "@tensorflow-models/depth-estimation";

// Function to load the depth estimation model with optimizations
async function loadModel() {
  await tf.setBackend("webgl");
  tf.env().set("WEBGL_CPU_FORWARD", false); // Optimizing WebGL backend
  const model = depthEstimation.SupportedModels.ARPortraitDepth;
  const estimatorConfig = { outputDepthRange: [0, 1] }; // Adjusted depth range
  const estimator = await depthEstimation.createEstimator(
    model,
    estimatorConfig
  );
  return estimator;
}

const estimator = await loadModel();

document
  .getElementById("imageUpload")
  .addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file) {
      const img = new Image();
      img.src = URL.createObjectURL(file);

      // Show the original image preview
      img.onload = async () => {
        document.getElementById("imagePreview").src = img.src;

        // Estimate depth and display depth map on canvas
        const depthMap = await estimator.estimateDepth(img, {
          minDepth: 0,
          maxDepth: 1,
        });

        const depthTexture = await renderDepthMapToTexture(
          depthMap,
          img.width,
          img.height
        );
        initThreeJsScene(depthTexture, img.width, img.height, img);
      };
    }
  });

async function renderDepthMapToTexture(depthMap, imgWidth, imgHeight) {
  const depthCanvas = document.getElementById("depthMapCanvas");
  const ctx = depthCanvas.getContext("2d");

  depthCanvas.width = imgWidth;
  depthCanvas.height = imgHeight;

  const depthImage = await depthMap.toCanvasImageSource();
  ctx.drawImage(depthImage, 0, 0, imgWidth, imgHeight);

  // Apply smoothing filter for improved depth consistency
  applySmoothing(ctx, imgWidth, imgHeight);

  const depthTexture = new THREE.CanvasTexture(depthCanvas);
  normalizeBrightness(ctx, imgWidth, imgHeight);
  return depthTexture;
}

// Apply a simple blur to smooth out the depth map
function applySmoothing(ctx, imgWidth, imgHeight) {
  ctx.filter = "blur(2px)";
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  ctx.putImageData(imageData, 0, 0);
  ctx.filter = "none"; // Reset filter after applying blur
}

// Normalize brightness to enhance depth map consistency
function normalizeBrightness(ctx, imgWidth, imgHeight) {
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  const data = imageData.data;
  let min = 255;
  let max = 0;

  for (let i = 0; i < data.length; i += 4) {
    const depthValue = data[i];
    if (depthValue < min) min = depthValue;
    if (depthValue > max) max = depthValue;
  }

  const range = max - min || 1;
  for (let i = 0; i < data.length; i += 4) {
    const normalizedValue = ((data[i] - min) / range) * 255;
    data[i] = data[i + 1] = data[i + 2] = normalizedValue;
  }

  ctx.putImageData(imageData, 0, 0);
}

async function initThreeJsScene(depthTexture, imgWidth, imgHeight, img) {
  const aspectRatio = imgWidth / imgHeight;

  // Update the Three.js renderer to respect image aspect ratio
  const renderer = new THREE.WebGLRenderer();
  const renderWidth = 500;
  const renderHeight = renderWidth / aspectRatio;
  renderer.setSize(renderWidth, renderHeight);
  document.getElementById("threeContainer").appendChild(renderer.domElement);

  const scene = new THREE.Scene();

  // Set the camera aspect ratio based on image dimensions
  const camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
  camera.position.z = 5;

  // Adjust plane dimensions based on the aspect ratio
  const planeWidth = 3;
  const planeHeight = planeWidth / aspectRatio;

  const widthSegments = 256;
  const heightSegments = 256;
  const geometry = new THREE.PlaneGeometry(
    planeWidth,
    planeHeight,
    widthSegments,
    heightSegments
  );

  // Depth map normalization
  const ctx = document.getElementById("depthMapCanvas").getContext("2d");
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  const data = imageData.data;

  let minDepth = 255;
  let maxDepth = 0;
  for (let i = 0; i < data.length; i += 4) {
    const depthValue = data[i];
    if (depthValue < minDepth) minDepth = depthValue;
    if (depthValue > maxDepth) maxDepth = depthValue;
  }
  const depthRange = maxDepth - minDepth || 1;

  const positionAttribute = geometry.attributes.position;
  for (let i = 0; i < positionAttribute.count; i++) {
    const ix = i % (widthSegments + 1);
    const iy = Math.floor(i / (widthSegments + 1));

    const x = Math.floor((ix / widthSegments) * imgWidth);
    const y = Math.floor((iy / heightSegments) * imgHeight);
    const index = (y * imgWidth + x) * 4;

    const depthValue = data[index];
    if (isNaN(depthValue)) continue;

    const normalizedDepth = (depthValue - minDepth) / depthRange;
    const z = (1 - normalizedDepth) * 1.0;
    positionAttribute.setZ(i, z);
  }

  positionAttribute.needsUpdate = true;

  const textureLoader = new THREE.TextureLoader();
  textureLoader.load(img.src, (texture) => {
    const material = new THREE.MeshStandardMaterial({
      map: texture,
      displacementMap: depthTexture,
      displacementScale: -1.0,
    });

    const plane = new THREE.Mesh(geometry, material);
    scene.add(plane);

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(0, 1, 1).normalize();
    scene.add(light);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const depthSlider = document.getElementById("depthSlider");
    depthSlider.addEventListener("input", (event) => {
      material.displacementScale = -event.target.value;
    });

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }

    animate();
  });
}
