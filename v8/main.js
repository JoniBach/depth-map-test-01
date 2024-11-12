import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as depthEstimation from "@tensorflow-models/depth-estimation";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import TRIANGULATION from "./TRIANGULATION.json";

// Define outer ring landmark indices (assuming this specific set of landmarks)
const OUTER_RING_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109,
];

async function main() {
  try {
    const detector = await faceLandmarksDetection.createDetector(
      faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
      {
        runtime: "tfjs",
        maxFaces: 1,
        refineLandmarks: false,
      }
    );

    const depthModel = depthEstimation.SupportedModels.ARPortraitDepth;
    const depthEstimator = await depthEstimation.createEstimator(depthModel, {
      outputDepthRange: [0, 1],
    });

    const img = document.getElementById("inputImage");
    const depthCanvas = document.getElementById("depthCanvas");
    const depthCtx = depthCanvas.getContext("2d");
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

    img.onload = async () => {
      const depthMap = await depthEstimator.estimateDepth(img, {
        minDepth: 0,
        maxDepth: 1,
      });

      const depthImage = await depthMap.toCanvasImageSource();
      depthCtx.drawImage(
        depthImage,
        0,
        0,
        depthCanvas.width,
        depthCanvas.height
      );

      const predictions = await detector.estimateFaces(img, {
        flipHorizontal: false,
      });

      // Draw the original image on the output canvas for reference
      outputCtx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);

      if (predictions.length > 0) {
        const prediction = predictions[0];
        const keypoints = prediction.keypoints;

        // Overlay keypoints on output canvas for troubleshooting
        outputCtx.fillStyle = "red";
        keypoints.forEach(({ x, y }) => {
          outputCtx.beginPath();
          outputCtx.arc(x, y, 2, 0, 2 * Math.PI);
          outputCtx.fill();
        });

        // Draw outer ring of points on outerRingCanvas
        outerRingCtx.clearRect(
          0,
          0,
          outerRingCanvas.width,
          outerRingCanvas.height
        );
        outerRingCtx.drawImage(
          img,
          0,
          0,
          outerRingCanvas.width,
          outerRingCanvas.height
        );
        outerRingCtx.strokeStyle = "blue";
        outerRingCtx.lineWidth = 2;
        outerRingCtx.beginPath();

        const startIdx = OUTER_RING_INDICES[0];
        const startPoint = keypoints[startIdx];
        outerRingCtx.moveTo(startPoint.x, startPoint.y);

        OUTER_RING_INDICES.slice(1).forEach((idx) => {
          const point = keypoints[idx];
          outerRingCtx.lineTo(point.x, point.y);
        });
        outerRingCtx.lineTo(startPoint.x, startPoint.y);
        outerRingCtx.stroke();

        // Draw masked depth map on maskedDepthCanvas with a clipping mask
        maskedDepthCtx.clearRect(
          0,
          0,
          maskedDepthCanvas.width,
          maskedDepthCanvas.height
        );
        maskedDepthCtx.save();
        maskedDepthCtx.beginPath();

        maskedDepthCtx.moveTo(startPoint.x, startPoint.y);
        OUTER_RING_INDICES.slice(1).forEach((idx) => {
          const point = keypoints[idx];
          maskedDepthCtx.lineTo(point.x, point.y);
        });
        maskedDepthCtx.closePath();
        maskedDepthCtx.clip();

        maskedDepthCtx.drawImage(
          depthImage,
          0,
          0,
          maskedDepthCanvas.width,
          maskedDepthCanvas.height
        );

        maskedDepthCtx.restore();

        // Invert the depth map on the invertedDepthCanvas
        invertedDepthCtx.drawImage(
          depthImage,
          0,
          0,
          invertedDepthCanvas.width,
          invertedDepthCanvas.height
        );
        const imageData = invertedDepthCtx.getImageData(
          0,
          0,
          invertedDepthCanvas.width,
          invertedDepthCanvas.height
        );
        for (let i = 0; i < imageData.data.length; i += 4) {
          imageData.data[i] = 255 - imageData.data[i];
          imageData.data[i + 1] = 255 - imageData.data[i + 1];
          imageData.data[i + 2] = 255 - imageData.data[i + 2];
        }
        invertedDepthCtx.putImageData(imageData, 0, 0);

        // Invert the masked depth map on the invertedMaskedDepthCanvas
        invertedMaskedDepthCtx.drawImage(
          maskedDepthCanvas,
          0,
          0,
          invertedMaskedDepthCanvas.width,
          invertedMaskedDepthCanvas.height
        );
        const maskedImageData = invertedMaskedDepthCtx.getImageData(
          0,
          0,
          invertedMaskedDepthCanvas.width,
          invertedMaskedDepthCanvas.height
        );
        for (let i = 0; i < maskedImageData.data.length; i += 4) {
          maskedImageData.data[i] = 255 - maskedImageData.data[i];
          maskedImageData.data[i + 1] = 255 - maskedImageData.data[i + 1];
          maskedImageData.data[i + 2] = 255 - maskedImageData.data[i + 2];
        }
        invertedMaskedDepthCtx.putImageData(maskedImageData, 0, 0);

        // Draw green triangulation overlay on triangulationCanvas
        triangulationCtx.clearRect(
          0,
          0,
          triangulationCanvas.width,
          triangulationCanvas.height
        );
        triangulationCtx.drawImage(
          img,
          0,
          0,
          triangulationCanvas.width,
          triangulationCanvas.height
        );
        triangulationCtx.strokeStyle = "green";
        triangulationCtx.lineWidth = 1;

        for (let i = 0; i < TRIANGULATION.length; i += 3) {
          const point1 = keypoints[TRIANGULATION[i]];
          const point2 = keypoints[TRIANGULATION[i + 1]];
          const point3 = keypoints[TRIANGULATION[i + 2]];

          triangulationCtx.beginPath();
          triangulationCtx.moveTo(point1.x, point1.y);
          triangulationCtx.lineTo(point2.x, point2.y);
          triangulationCtx.lineTo(point3.x, point3.y);
          triangulationCtx.closePath();
          triangulationCtx.stroke();
        }

        // Draw combined overlay on combinedOverlayCanvas
        combinedOverlayCtx.clearRect(
          0,
          0,
          combinedOverlayCanvas.width,
          combinedOverlayCanvas.height
        );
        combinedOverlayCtx.drawImage(
          img,
          0,
          0,
          combinedOverlayCanvas.width,
          combinedOverlayCanvas.height
        );

        // 1. Draw green triangulation
        combinedOverlayCtx.strokeStyle = "green";
        combinedOverlayCtx.lineWidth = 1;
        for (let i = 0; i < TRIANGULATION.length; i += 3) {
          const point1 = keypoints[TRIANGULATION[i]];
          const point2 = keypoints[TRIANGULATION[i + 1]];
          const point3 = keypoints[TRIANGULATION[i + 2]];

          combinedOverlayCtx.beginPath();
          combinedOverlayCtx.moveTo(point1.x, point1.y);
          combinedOverlayCtx.lineTo(point2.x, point2.y);
          combinedOverlayCtx.lineTo(point3.x, point3.y);
          combinedOverlayCtx.closePath();
          combinedOverlayCtx.stroke();
        }

        // 2. Draw blue outer ring
        combinedOverlayCtx.strokeStyle = "blue";
        combinedOverlayCtx.lineWidth = 2;
        combinedOverlayCtx.beginPath();
        combinedOverlayCtx.moveTo(startPoint.x, startPoint.y);
        OUTER_RING_INDICES.slice(1).forEach((idx) => {
          const point = keypoints[idx];
          combinedOverlayCtx.lineTo(point.x, point.y);
        });
        combinedOverlayCtx.lineTo(startPoint.x, startPoint.y);
        combinedOverlayCtx.stroke();

        // 3. Draw red keypoints
        combinedOverlayCtx.fillStyle = "red";
        keypoints.forEach(({ x, y }) => {
          combinedOverlayCtx.beginPath();
          combinedOverlayCtx.arc(x, y, 2, 0, 2 * Math.PI);
          combinedOverlayCtx.fill();
        });

        // Setup vertices and UVs for each renderer
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
          indices.push(
            TRIANGULATION[i],
            TRIANGULATION[i + 1],
            TRIANGULATION[i + 2]
          );
        }

        // Set up each renderer, scene, camera, and mesh independently

        // 1. Flat Wireframe
        const flatWireframeContainer = document.getElementById(
          "flatWireframeContainer"
        );
        const flatWireframeScene = new THREE.Scene();
        const flatWireframeCamera = new THREE.PerspectiveCamera(
          75,
          1,
          0.1,
          1000
        );
        flatWireframeCamera.position.z = 500;
        const flatWireframeRenderer = new THREE.WebGLRenderer();
        flatWireframeRenderer.setSize(400, 400);
        flatWireframeContainer.appendChild(flatWireframeRenderer.domElement);
        const flatWireframeControls = new OrbitControls(
          flatWireframeCamera,
          flatWireframeRenderer.domElement
        );
        flatWireframeControls.enableDamping = true;
        flatWireframeControls.dampingFactor = 0.25;
        flatWireframeControls.enableZoom = true;

        const flatGeometry = new THREE.BufferGeometry();
        flatGeometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(flatVertices, 3)
        );
        flatGeometry.setAttribute(
          "uv",
          new THREE.Float32BufferAttribute(uvCoordinates, 2)
        );
        flatGeometry.setIndex(indices);

        const flatWireframeMaterial = new THREE.MeshBasicMaterial({
          color: 0xff0000,
          wireframe: true,
        });
        const flatWireframeMesh = new THREE.Mesh(
          flatGeometry,
          flatWireframeMaterial
        );
        flatWireframeScene.add(flatWireframeMesh);

        // 2. Wireframe 3D Mesh
        const wireframeContainer =
          document.getElementById("wireframeContainer");
        const wireframeScene = new THREE.Scene();
        const wireframeCamera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        wireframeCamera.position.z = 500;
        const wireframeRenderer = new THREE.WebGLRenderer();
        wireframeRenderer.setSize(400, 400);
        wireframeContainer.appendChild(wireframeRenderer.domElement);
        const wireframeControls = new OrbitControls(
          wireframeCamera,
          wireframeRenderer.domElement
        );
        wireframeControls.enableDamping = true;
        wireframeControls.dampingFactor = 0.25;
        wireframeControls.enableZoom = true;

        const wireframeGeometry = new THREE.BufferGeometry();
        wireframeGeometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(vertices, 3)
        );
        wireframeGeometry.setAttribute(
          "uv",
          new THREE.Float32BufferAttribute(uvCoordinates, 2)
        );
        wireframeGeometry.setIndex(indices);

        const wireframeMaterial = new THREE.MeshBasicMaterial({
          color: 0x00ff00,
          wireframe: true,
        });
        const wireframeMesh = new THREE.Mesh(
          wireframeGeometry,
          wireframeMaterial
        );
        wireframeScene.add(wireframeMesh);

        // 3. Textured 3D Mesh
        const texturedContainer = document.getElementById("texturedContainer");
        const texturedScene = new THREE.Scene();
        const texturedCamera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        texturedCamera.position.z = 500;
        const texturedRenderer = new THREE.WebGLRenderer();
        texturedRenderer.setSize(400, 400);
        texturedContainer.appendChild(texturedRenderer.domElement);
        const texturedControls = new OrbitControls(
          texturedCamera,
          texturedRenderer.domElement
        );
        texturedControls.enableDamping = true;
        texturedControls.dampingFactor = 0.25;
        texturedControls.enableZoom = true;

        const texture = new THREE.Texture(img);
        texture.needsUpdate = true;

        const texturedGeometry = new THREE.BufferGeometry();
        texturedGeometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(vertices, 3)
        );
        texturedGeometry.setAttribute(
          "uv",
          new THREE.Float32BufferAttribute(uvCoordinates, 2)
        );
        texturedGeometry.setIndex(indices);

        const texturedMaterial = new THREE.MeshBasicMaterial({
          map: texture,
          side: THREE.DoubleSide,
        });
        const texturedMesh = new THREE.Mesh(texturedGeometry, texturedMaterial);
        texturedScene.add(texturedMesh);

        // Animation loop for all scenes
        function animate() {
          requestAnimationFrame(animate);

          flatWireframeControls.update();
          wireframeControls.update();
          texturedControls.update();

          flatWireframeRenderer.render(flatWireframeScene, flatWireframeCamera);
          wireframeRenderer.render(wireframeScene, wireframeCamera);
          texturedRenderer.render(texturedScene, texturedCamera);
        }
        animate();
      } else {
        console.log("No faces detected.");
      }
    };

    if (img.complete && img.naturalHeight !== 0) {
      img.onload();
    }
  } catch (error) {
    console.error("An error occurred:", error);
  }
}

main();
