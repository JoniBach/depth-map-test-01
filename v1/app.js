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
  const estimatorConfig = { outputDepthRange: [0, 0.5] }; // Narrower depth range for finer detail
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

// Function to render the depth map with depth gradient mapping and perspective shading
async function renderDepthMap(depthMap, imgWidth, imgHeight) {
  console.log("Rendering depth map...");

  const depthCanvas = document.getElementById("depthMapCanvas");
  const ctx = depthCanvas.getContext("2d");

  // Set canvas dimensions without scaling
  depthCanvas.width = imgWidth;
  depthCanvas.height = imgHeight;

  try {
    // Use `toCanvasImageSource()` to render the depth image
    const depthImage = await depthMap.toCanvasImageSource();

    // Clear the canvas to ensure no residual artifacts
    ctx.clearRect(0, 0, depthCanvas.width, depthCanvas.height);

    // Draw the depth image on the canvas at the same resolution as the image
    ctx.drawImage(depthImage, 0, 0, imgWidth, imgHeight);

    // Apply depth gradient mapping for better depth perception
    applyDepthGradient(ctx, imgWidth, imgHeight);

    // Apply perspective shading to simulate depth-based lighting
    applyPerspectiveShading(ctx, imgWidth, imgHeight);

    console.log(
      "Depth map rendered successfully on canvas with enhanced quality"
    );
  } catch (error) {
    console.error("Error rendering depth map:", error);
  }
}

// Function to apply depth gradient mapping to simulate depth perception
function applyDepthGradient(ctx, imgWidth, imgHeight) {
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  const data = imageData.data;

  for (let i = 0; i < data.length; i += 4) {
    const depthValue = data[i]; // Using the red channel as a base for depth

    // Map depth values to a gradient (e.g., closer objects more intense color)
    data[i] = 255 - depthValue; // Red for closer objects
    data[i + 1] = depthValue * 0.6; // Green based on depth
    data[i + 2] = depthValue * 0.3; // Blue based on depth
  }

  ctx.putImageData(imageData, 0, 0);
}

// Function to apply perspective shading to simulate depth-based lighting
function applyPerspectiveShading(ctx, imgWidth, imgHeight) {
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  const data = imageData.data;

  for (let i = 0; i < data.length; i += 4) {
    const depthValue = data[i]; // Use depth as a base for shading
    const shadeFactor = 1 - depthValue / 255; // Fading effect with depth

    // Adjust the brightness of each color channel for shading
    data[i] *= shadeFactor; // Red
    data[i + 1] *= shadeFactor; // Green
    data[i + 2] *= shadeFactor; // Blue
  }

  ctx.putImageData(imageData, 0, 0);
}
