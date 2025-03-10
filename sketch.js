// function setup() {
//   createCanvas(400, 400);
// }

// function draw() {
//   background(220);
// }

let boids = [];
const numBoids = 100;

// Simulation parameters
const visualRange = 50;
const centeringFactor = 0.005;
const matchingFactor = 0.05;
const avoidFactor = 0.05;
const turnFactor = 1.0;
const minSpeed = 2;
const maxSpeed = 4;
const margin = 50;

class Boid {
  constructor() {
    this.x = random(width);
    this.y = random(height);
    this.vx = random(-2, 2);
    this.vy = random(-2, 2);
    this.biasval = 0;
  }

  update(boids) {
    let xPosAvg = 0;
    let yPosAvg = 0;
    let xVelAvg = 0;
    let yVelAvg = 0;
    let neighboringBoids = 0;
    let closeDx = 0;
    let closeDy = 0;

    // Find average positions and velocities of nearby boids
    for (let other of boids) {
      if (other === this) continue;

      let dx = other.x - this.x;
      let dy = other.y - this.y;
      let distance = sqrt(dx * dx + dy * dy);

      if (distance < visualRange) {
        // Separation: avoid getting too close
        if (distance < visualRange / 2) {
          closeDx -= dx;
          closeDy -= dy;
        }

        // Alignment and cohesion
        xPosAvg += other.x;
        yPosAvg += other.y;
        xVelAvg += other.vx;
        yVelAvg += other.vy;
        neighboringBoids++;
      }
    }

    // Update velocity based on neighboring boids
    if (neighboringBoids > 0) {
      xPosAvg = xPosAvg / neighboringBoids;
      yPosAvg = yPosAvg / neighboringBoids;
      xVelAvg = xVelAvg / neighboringBoids;
      yVelAvg = yVelAvg / neighboringBoids;

      // Update velocity based on center of mass and velocity matching
      this.vx += (xPosAvg - this.x) * centeringFactor +
                 (xVelAvg - this.vx) * matchingFactor;
      this.vy += (yPosAvg - this.y) * centeringFactor +
                 (yVelAvg - this.vy) * matchingFactor;

      // Add separation force
      this.vx += closeDx * avoidFactor;
      this.vy += closeDy * avoidFactor;
    }

    // Keep boids within bounds
    if (this.y < margin) this.vy += turnFactor;
    if (this.x > width - margin) this.vx -= turnFactor;
    if (this.x < margin) this.vx += turnFactor;
    if (this.y > height - margin) this.vy -= turnFactor;

    // Limit speed
    let speed = sqrt(this.vx * this.vx + this.vy * this.vy);
    if (speed < minSpeed) {
      this.vx = (this.vx / speed) * minSpeed;
      this.vy = (this.vy / speed) * minSpeed;
    }
    if (speed > maxSpeed) {
      this.vx = (this.vx / speed) * maxSpeed;
      this.vy = (this.vy / speed) * maxSpeed;
    }

    // Update position
    this.x += this.vx;
    this.y += this.vy;
  }

  draw() {
    // Calculate heading angle
    let angle = atan2(this.vy, this.vx);
    
    push();
    translate(this.x, this.y);
    rotate(angle);
    
    // Draw triangle for boid
    fill(200, 100, 100);
    noStroke();
    triangle(-10, -5, -10, 5, 10, 0);
    
    pop();
  }
}

function setup() {
  createCanvas(800, 600);
  
  // Initialize boids
  for (let i = 0; i < numBoids; i++) {
    boids.push(new Boid());
  }
}

function draw() {
  background(51);
  
  // Update and draw all boids
  for (let boid of boids) {
    boid.update(boids);
    boid.draw();
  }
}