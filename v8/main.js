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
          outputCtx.arc(x, y, 2, 0, 2 * Math.PI); // Draw a small circle at each keypoint
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

        // Move to the first outer ring point
        const startIdx = OUTER_RING_INDICES[0];
        const startPoint = keypoints[startIdx];
        outerRingCtx.moveTo(startPoint.x, startPoint.y);

        // Draw lines connecting outer ring points
        OUTER_RING_INDICES.slice(1).forEach((idx) => {
          const point = keypoints[idx];
          outerRingCtx.lineTo(point.x, point.y);
        });

        // Close the loop back to the starting point
        outerRingCtx.lineTo(startPoint.x, startPoint.y);
        outerRingCtx.stroke();

        // Draw masked depth map on maskedDepthCanvas using a clipping mask
        maskedDepthCtx.clearRect(
          0,
          0,
          maskedDepthCanvas.width,
          maskedDepthCanvas.height
        );
        maskedDepthCtx.save(); // Save the current state before clipping
        maskedDepthCtx.beginPath();

        // Move to the first outer ring point for clipping
        maskedDepthCtx.moveTo(startPoint.x, startPoint.y);
        OUTER_RING_INDICES.slice(1).forEach((idx) => {
          const point = keypoints[idx];
          maskedDepthCtx.lineTo(point.x, point.y);
        });
        maskedDepthCtx.closePath();
        maskedDepthCtx.clip();

        // Draw the depth map within the clipping path
        maskedDepthCtx.drawImage(
          depthImage,
          0,
          0,
          maskedDepthCanvas.width,
          maskedDepthCanvas.height
        );

        // Restore to remove clipping for future drawings
        maskedDepthCtx.restore();

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
