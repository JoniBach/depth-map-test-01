import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-converter";
import "@tensorflow/tfjs-backend-webgl";
import * as depthEstimation from "@tensorflow-models/depth-estimation";

let renderer, scene, camera, plane;

async function loadModel() {
  await tf.setBackend("webgl");
  tf.env().set("WEBGL_CPU_FORWARD", false);
  const model = depthEstimation.SupportedModels.ARPortraitDepth;
  const estimatorConfig = { outputDepthRange: [0, 1] };
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
        document.getElementById("imagePreview").src = img.src;
        const lightestTone = findLightestTone(img);
        const depthMap = await estimator.estimateDepth(img, {
          minDepth: 0,
          maxDepth: 1,
        });

        const depthTexture = await renderDepthMapToTexture(
          depthMap,
          img.width,
          img.height,
          lightestTone
        );
        initThreeJsScene(depthTexture, img.width, img.height, img);
      };
    }
  });

function findLightestTone(img) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0, img.width, img.height);

  const imageData = ctx.getImageData(0, 0, img.width, img.height);
  const data = imageData.data;

  let maxBrightness = 0;
  let lightestColor = { r: 255, g: 255, b: 255 };

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const brightness = r + g + b;
    if (brightness > maxBrightness) {
      maxBrightness = brightness;
      lightestColor = { r, g, b };
    }
  }
  return lightestColor;
}

document
  .getElementById("toneRangeSlider")
  .addEventListener("input", async () => {
    const img = document.getElementById("imagePreview");
    if (img.src) {
      const lightestTone = findLightestTone(img);
      const depthMap = await estimator.estimateDepth(img, {
        minDepth: 0,
        maxDepth: 1,
      });
      const depthTexture = await renderDepthMapToTexture(
        depthMap,
        img.width,
        img.height,
        lightestTone
      );
      initThreeJsScene(depthTexture, img.width, img.height, img);
    }
  });

async function renderDepthMapToTexture(depthMap, imgWidth, imgHeight) {
  const depthCanvas = document.getElementById("depthMapCanvas");
  const ctx = depthCanvas.getContext("2d");

  depthCanvas.width = imgWidth;
  depthCanvas.height = imgHeight;

  const depthImage = await depthMap.toCanvasImageSource();
  ctx.drawImage(depthImage, 0, 0, imgWidth, imgHeight);

  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  const data = imageData.data;
  const toneRange = parseInt(document.getElementById("toneRangeSlider").value);

  // Set a depth threshold to define foreground vs. background
  const depthThreshold = toneRange * 3;

  // Apply mask to make background pixels transparent and keep foreground opaque
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const brightness = r + g + b;

    if (brightness < depthThreshold) {
      data[i + 3] = 0; // Make background pixels fully transparent
    } else {
      data[i + 3] = 255; // Keep foreground fully opaque
    }
  }

  ctx.putImageData(imageData, 0, 0);

  // Convert canvas to a texture that only shows the masked area
  const depthTexture = new THREE.CanvasTexture(depthCanvas);

  return depthTexture;
}

function blendColors(original, lightest, blendFactor = 0.5) {
  return Math.floor(original * (1 - blendFactor) + lightest * blendFactor);
}

// Stronger blur for general smoothing, with an additional selective smoothing pass
function applySmoothing(ctx, imgWidth, imgHeight) {
  ctx.filter = "blur(4px)"; // Increased blur strength
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  ctx.putImageData(imageData, 0, 0);
  ctx.filter = "none";
}

function adaptiveSmoothing(ctx, imgWidth, imgHeight) {
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  const data = imageData.data;

  // Apply selective smoothing around high-contrast areas
  for (let y = 1; y < imgHeight - 1; y++) {
    for (let x = 1; x < imgWidth - 1; x++) {
      const i = (y * imgWidth + x) * 4;
      const brightness = data[i] + data[i + 1] + data[i + 2];

      // Check contrast with neighboring pixels
      const contrast =
        Math.abs(brightness - (data[i - 4] + data[i - 3] + data[i - 2])) +
        Math.abs(brightness - (data[i + 4] + data[i + 5] + data[i + 6]));

      if (contrast > 50) {
        // Threshold for high contrast
        data[i] = (data[i] + data[i - 4] + data[i + 4]) / 3;
        data[i + 1] = (data[i + 1] + data[i - 3] + data[i + 5]) / 3;
        data[i + 2] = (data[i + 2] + data[i - 2] + data[i + 6]) / 3;
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
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
async function initThreeJsScene(depthTexture, imgWidth, imgHeight, img) {
  const aspectRatio = imgWidth / imgHeight;

  if (!renderer) {
    renderer = new THREE.WebGLRenderer({ alpha: true });
    renderer.setSize(500, 500 / aspectRatio);
    document.getElementById("threeContainer").appendChild(renderer.domElement);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
    camera.position.z = 5;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(0, 1, 1).normalize();
    scene.add(light);

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  }

  // Define a more compact plane size focused on the subject area
  const planeWidth = 1.2; // Slightly reduced width for focus
  const planeHeight = planeWidth / aspectRatio;
  const widthSegments = 128;
  const heightSegments = 128;

  if (!plane) {
    const geometry = new THREE.PlaneGeometry(
      planeWidth,
      planeHeight,
      widthSegments,
      heightSegments
    );

    const textureLoader = new THREE.TextureLoader();
    textureLoader.load(img.src, (texture) => {
      const material = new THREE.MeshStandardMaterial({
        map: texture,
        displacementMap: depthTexture,
        displacementScale: -0.3, // Lower displacement scale to reduce edge artifacts
        transparent: false, // Enable transparency for masking
        alphaMap: depthTexture, // Use depth texture as alpha map
        alphaTest: 0.3, // Set threshold to ignore fully transparent areas
        depthWrite: false, // Avoid depth conflicts with background
        side: THREE.DoubleSide, // Render both sides to improve visibility at angles
      });

      plane = new THREE.Mesh(geometry, material);
      scene.add(plane);
    });
  } else {
    plane.material.displacementMap = depthTexture;
    plane.material.needsUpdate = true;
  }

  // Adjust displacement with the slider
  const depthSlider = document.getElementById("depthSlider");
  depthSlider.addEventListener("input", (event) => {
    plane.material.displacementScale = -event.target.value * 0.3; // Scale for smoother adjustment
  });
}
