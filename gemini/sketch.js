let hexRadius = 100;
let hexHeight = 150;
let sphereRadius = 20;
let angle = 0; // for sphere animation
let cam;

function setup() {
    createCanvas(windowWidth, windowHeight, WEBGL);
    // Enable orbiting with the mouse
    orbitControl();

    // Set up initial camera position (optional, orbitControl overrides this on interaction)
    cam = createCamera();
    cam.setPosition(300, 200, 300);
    cam.lookAt(0, 0, 0);
}

function draw() {
    background(50); // Dark gray background

    // Add some lighting
    ambientLight(60);
    pointLight(255, 255, 255, 0, 0, 200);

    // Draw the hexagonal prism
    push();
    noStroke();
    ambientMaterial(100, 100, 150); // Bluish material for the box
    drawHexagonalPrism(hexRadius, hexHeight);
    pop();

    // Draw the sphere
    push();
    // Animate the sphere's position in a circle within the box
    let sphereX = cos(angle) * (hexRadius - sphereRadius - 10); // keep sphere slightly within the box
    let sphereY = sin(angle) * (hexRadius - sphereRadius - 10);
    let sphereZ = 0; // stay in the middle vertically for this example

    translate(sphereX, sphereY, sphereZ);
    noStroke();
    ambientMaterial(255, 50, 50); // Red material for the sphere
    sphere(sphereRadius);
    pop();

    // Increase the angle for animation
    angle += 0.02;
}

// Function to draw a hexagonal prism
function drawHexagonalPrism(radius, height) {
    let angleStep = TWO_PI / 6;

    // Top and bottom faces
    beginShape();
    for (let i = 0; i < 6; i++) {
        let x = cos(i * angleStep) * radius;
        let y = sin(i * angleStep) * radius;
        vertex(x, y, -height / 2);
    }
    endShape(CLOSE);

    beginShape();
    for (let i = 0; i < 6; i++) {
        let x = cos(i * angleStep) * radius;
        let y = sin(i * angleStep) * radius;
        vertex(x, y, height / 2);
    }
    endShape(CLOSE);

    // Side faces
    beginShape(TRIANGLE_STRIP);
    for (let i = 0; i <= 6; i++) {
        let currentAngle = i * angleStep;
        let nextAngle = (i + 1) * angleStep;
        let x1 = cos(currentAngle) * radius;
        let y1 = sin(currentAngle) * radius;
        let x2 = cos(nextAngle) * radius;
        let y2 = sin(nextAngle) * radius;

        vertex(x1, y1, -height / 2);
        vertex(x1, y1, height / 2);
        // To close the strip, repeat the first vertices at the end
        if (i === 6) {
             vertex(cos(0) * radius, sin(0) * radius, -height / 2);
             vertex(cos(0) * radius, sin(0) * radius, height / 2);
        }
    }
    endShape();
}


function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
}