import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-converter";
import "@tensorflow/tfjs-backend-webgl";
import * as depthEstimation from "@tensorflow-models/depth-estimation";

// Continue with the rest of your code...

// Continue with the rest of the code as before...

// Continue with the rest of the code as before...

// Function to load the depth estimation model
async function loadModel() {
  await tf.setBackend("webgl");
  const model = depthEstimation.SupportedModels.ARPortraitDepth;
  const estimatorConfig = { outputDepthRange: [0, 0.5] };
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
      img.onload = async () => {
        const depthMap = await estimator.estimateDepth(img, {
          minDepth: 0,
          maxDepth: 1,
        });
        const depthTexture = await renderDepthMapToTexture(
          depthMap,
          img.width,
          img.height
        );
        initThreeJsScene(depthTexture, img.width, img.height);
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

  // Process image data to ensure the original depth map is not changed
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  const data = imageData.data;

  // Ensure no changes to the original depth map; we don't touch the black areas here
  // The depth map remains intact (black areas will remain black in the image)

  // Apply this unmodified depth map to a texture
  const depthTexture = new THREE.CanvasTexture(depthCanvas);

  // Normalize brightness (optional but improves visualization)
  normalizeBrightness(ctx, imgWidth, imgHeight);

  // Now we return the depth texture with no inversion applied yet
  return depthTexture;
}

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

async function initThreeJsScene(depthTexture, imgWidth, imgHeight) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    75,
    imgWidth / imgHeight,
    0.1,
    1000
  );
  camera.position.z = 5;

  const renderer = new THREE.WebGLRenderer();
  renderer.setSize(500, 500);
  document.getElementById("threeContainer").appendChild(renderer.domElement);

  const aspectRatio = imgWidth / imgHeight;
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

  const material = new THREE.MeshStandardMaterial({
    map: depthTexture,
    transparent: true,
  });

  const plane = new THREE.Mesh(geometry, material);
  scene.add(plane);

  const ctx = document.getElementById("depthMapCanvas").getContext("2d");
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  const data = imageData.data;

  // Calculate min and max depth values for normalization
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
    const normalizedDepth = (depthValue - minDepth) / depthRange;
    const z = (1 - normalizedDepth) * 1.0; // Invert depth for correct direction

    // Interpolate the Z position for smooth transitions
    positionAttribute.setZ(i, z);
  }

  positionAttribute.needsUpdate = true;

  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(0, 1, 1).normalize();
  scene.add(light);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  animate();
}
