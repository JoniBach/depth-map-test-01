import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import TRIANGULATION from "./TRIANGULATION.json";

async function main() {
  try {
    // Load the face landmarks detection model
    const detector = await faceLandmarksDetection.createDetector(
      faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
      {
        runtime: "tfjs",
        maxFaces: 1,
        refineLandmarks: false,
      }
    );

    const img = document.getElementById("inputImage");

    // Setup for wireframe 3D visualization
    const wireframeContainer = document.getElementById("wireframeContainer");
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

    // Setup for textured 3D visualization
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

    // Set up the 2D canvas context
    const canvas = document.getElementById("outputCanvas");
    const ctx = canvas.getContext("2d");

    img.onload = async () => {
      const predictions = await detector.estimateFaces(img, {
        flipHorizontal: false,
      });
      if (predictions.length > 0) {
        const prediction = predictions[0];
        const keypoints = prediction.keypoints;

        // Render the 2D point map on the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "red";
        keypoints.forEach(({ x, y }) => {
          ctx.beginPath();
          ctx.arc(x, y, 1, 0, 2 * Math.PI);
          ctx.fill();
        });

        // Set up vertices and UV coordinates for 3D models
        const vertices = [];
        const uvCoordinates = [];
        const depthScale = 1;

        keypoints.forEach(({ x, y, z }) => {
          vertices.push(x - img.width / 2, -y + img.height / 2, z * depthScale);
          uvCoordinates.push(x / img.width, 1 - y / img.height);
        });

        // Configure geometry and indices based on TRIANGULATION
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(vertices, 3)
        );
        geometry.setAttribute(
          "uv",
          new THREE.Float32BufferAttribute(uvCoordinates, 2)
        );

        const indices = [];
        for (let i = 0; i < TRIANGULATION.length; i += 3) {
          indices.push(
            TRIANGULATION[i],
            TRIANGULATION[i + 1],
            TRIANGULATION[i + 2]
          );
        }
        geometry.setIndex(indices);

        // Wireframe mesh
        const wireframeMaterial = new THREE.MeshBasicMaterial({
          color: 0x00ff00,
          wireframe: true,
        });
        const wireframeMesh = new THREE.Mesh(geometry, wireframeMaterial);
        wireframeScene.add(wireframeMesh);

        // Textured mesh
        const texture = new THREE.Texture(img);
        texture.needsUpdate = true;
        const texturedMaterial = new THREE.MeshBasicMaterial({
          map: texture,
          side: THREE.DoubleSide,
        });
        const texturedMesh = new THREE.Mesh(geometry, texturedMaterial);
        texturedScene.add(texturedMesh);

        // Render loop for both scenes
        function animate() {
          requestAnimationFrame(animate);
          wireframeControls.update();
          texturedControls.update();
          wireframeRenderer.render(wireframeScene, wireframeCamera);
          texturedRenderer.render(texturedScene, texturedCamera);
        }
        animate();
      } else {
        console.log("No faces detected.");
      }
    };

    // Ensure the image triggers the load event if already cached
    if (img.complete && img.naturalHeight !== 0) {
      img.onload();
    }
  } catch (error) {
    console.error("An error occurred:", error);
  }
}

main();
