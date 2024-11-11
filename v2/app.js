// Confirm that the app.js file is loaded
console.log("app.js is loaded");

import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-converter";
import "@tensorflow/tfjs-backend-webgl";
import * as depthEstimation from "@tensorflow-models/depth-estimation";

// Function to load the depth estimation model
async function loadModel() {
  console.log("Loading the model...");
  await tf.setBackend("webgl");
  const model = depthEstimation.SupportedModels.ARPortraitDepth;

  // Configure the model with an extended depth range for more detail
  const estimatorConfig = { outputDepthRange: [0, 0.5] }; // Focus on closer objects
  const estimator = await depthEstimation.createEstimator(
    model,
    estimatorConfig
  );
  console.log("Model loaded successfully:", estimator);
  return estimator;
}

// Load the model
const estimator = await loadModel();

// Event listener for file input
document
  .getElementById("imageUpload")
  .addEventListener("change", async (event) => {
    console.log("Image file selected");
    const file = event.target.files[0];
    if (file) {
      console.log("File details:", file);
      const img = new Image();
      img.src = URL.createObjectURL(file);
      img.onload = async () => {
        console.log(
          "Image loaded successfully with dimensions:",
          img.width,
          img.height
        );

        // Estimate depth with the model
        const depthMap = await estimator.estimateDepth(img, {
          minDepth: 0,
          maxDepth: 1,
        });
        console.log("Depth estimation completed:", depthMap);

        // Render the depth map with enhanced quality
        renderDepthMap(depthMap, img.width, img.height);
      };
    } else {
      console.log("No file selected");
    }
  });

// Function to render the depth map with normalized brightness
async function renderDepthMap(depthMap, imgWidth, imgHeight) {
  console.log("Rendering depth map...");

  const depthCanvas = document.getElementById("depthMapCanvas");
  const ctx = depthCanvas.getContext("2d");

  depthCanvas.width = imgWidth;
  depthCanvas.height = imgHeight;

  try {
    const depthImage = await depthMap.toCanvasImageSource();
    ctx.clearRect(0, 0, depthCanvas.width, depthCanvas.height);
    ctx.drawImage(depthImage, 0, 0, imgWidth, imgHeight);

    // Normalize brightness across the image
    normalizeBrightness(ctx, imgWidth, imgHeight);

    console.log("Depth map rendered successfully with normalized brightness.");
  } catch (error) {
    console.error("Error rendering depth map:", error);
  }
}

// Function to normalize brightness across the image
function normalizeBrightness(ctx, imgWidth, imgHeight) {
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  const data = imageData.data;

  let min = 255;
  let max = 0;

  // Find the min and max depth values to normalize
  for (let i = 0; i < data.length; i += 4) {
    const depthValue = data[i]; // Assuming grayscale in red channel
    if (depthValue < min) min = depthValue;
    if (depthValue > max) max = depthValue;
  }

  // Normalize each pixel value to full grayscale range
  const range = max - min || 1; // Avoid division by zero
  for (let i = 0; i < data.length; i += 4) {
    const normalizedValue = ((data[i] - min) / range) * 255;
    data[i] = data[i + 1] = data[i + 2] = normalizedValue; // Set all channels
  }

  ctx.putImageData(imageData, 0, 0);
}
