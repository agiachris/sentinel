let scene, camera, renderer, pointCloud, controls;
let viewWidth, viewHeight;

function initPointCloud() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    // Camera setup
    viewWidth = window.innerWidth / 2 - 66;
    viewHeight = window.innerHeight - 132;
    camera = new THREE.PerspectiveCamera(75, viewWidth / viewHeight, 0.1, 1000);
    camera.position.set(1.5, 0.5, 0.5);
    camera.lookAt(new THREE.Vector3(0, 0, 0));
    camera.up.set(0, 0, 1);

    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(viewWidth, viewHeight);
    const div = document.getElementById('canvas-wrapper');
    div.appendChild(renderer.domElement);

    controls = new THREE_ADDONS.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enableZoom = true;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const axesHelper = new THREE.AxesHelper(1);
    scene.add(axesHelper);

    // Point cloud setup
    let geometry = new THREE.BufferGeometry();
    pointCloud = new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0x111111, size: 0.03 }));
    scene.add(pointCloud);

    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);

    // Initial render
    animate();
}

function onWindowResize() {
    viewWidth = window.innerWidth / 2 - 66;
    viewHeight = window.innerHeight - 132;
    renderer.setSize(viewWidth, viewHeight); // Adjust size as needed
    camera.aspect = viewWidth / viewHeight;
    camera.updateProjectionMatrix();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function updatePointCloud(points) {
    // Convert points to Float32Array for BufferGeometry
    let vertices = new Float32Array(points.length * 3);
    points.forEach((point, i) => {
        vertices[i * 3] = point[0];
        vertices[i * 3 + 1] = point[1];
        vertices[i * 3 + 2] = point[2];
    });

    // Update geometry
    pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    pointCloud.geometry.computeBoundingSphere();
}

// Call this function to initialize the point cloud viewer
initPointCloud();

function updateImage(elementId, path) {
    var img = new Image();
    var imageElement = document.getElementById(elementId);

    // Preload the image
    img.onload = function() {
        imageElement.src = this.src;
    };
    img.src = path;
}

function updateData() {
    fetch('http://127.0.0.1:8000/data') // Replace with your server's data endpoint
        .then(response => response.json())
        .then(data => {
            var timestamp = new Date().getTime();
            const suffix = '?t=' + timestamp;

            // Update the point cloud
            updatePointCloud(data.pointCloud);

            // Update the image and mask
            updateImage('camera-image', '/uploads/' + data.image + suffix);
            updateImage('segmentation-mask', '/uploads/' + data.mask + suffix);

            // Update the elapsed time
            document.getElementById('elapsed-time').textContent = data.elapsedTime.toFixed(2);
            document.getElementById('process-elapsed-time').textContent = data.processElapsedTime.toFixed(2);
        })
        .catch(error => console.error('Error fetching data:', error));
}

// Update the data every 1/3 second
setInterval(updateData, 333);
