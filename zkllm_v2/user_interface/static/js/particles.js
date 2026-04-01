/* =====================================================================
   VerifAI — Interactive Particle Constellation System
   A GPU-accelerated canvas layer with mouse-reactive floating nodes,
   dynamic connection lines, and pulsing energy orbs.
   ===================================================================== */

(function () {
  'use strict';

  // ── Configuration ──────────────────────────────────────────────────
  const CONFIG = {
    particleCount: 80,
    connectionDistance: 160,
    mouseInfluenceRadius: 220,
    mouseRepelStrength: 0.04,
    baseSpeed: 0.25,
    particleMinRadius: 1.2,
    particleMaxRadius: 3.0,
    glowOrbCount: 5,
    glowOrbMinRadius: 60,
    glowOrbMaxRadius: 180,
    fps: 60,
  };

  // ── State ──────────────────────────────────────────────────────────
  let canvas, ctx;
  let width, height;
  let mouseX = -1000, mouseY = -1000;
  let particles = [];
  let glowOrbs = [];
  let animFrameId;
  let lastTime = 0;
  const interval = 1000 / CONFIG.fps;

  // ── Color palette (matches CSS vars) ──────────────────────────────
  const COLORS = {
    indigo:  { r: 99,  g: 102, b: 241 },
    violet:  { r: 139, g: 92,  b: 246 },
    sky:     { r: 56,  g: 189, b: 248 },
    emerald: { r: 16,  g: 185, b: 129 },
    amber:   { r: 245, g: 158, b: 11  },
  };
  const COLOR_KEYS = Object.keys(COLORS);

  // ── Utility ────────────────────────────────────────────────────────
  function randomBetween(a, b) {
    return a + Math.random() * (b - a);
  }
  function pickColor() {
    return COLORS[COLOR_KEYS[Math.floor(Math.random() * COLOR_KEYS.length)]];
  }
  function lerp(a, b, t) {
    return a + (b - a) * t;
  }
  function dist(x1, y1, x2, y2) {
    const dx = x1 - x2, dy = y1 - y2;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // ── Particle class ────────────────────────────────────────────────
  class Particle {
    constructor() {
      this.reset();
    }
    reset() {
      this.x = Math.random() * (width || window.innerWidth);
      this.y = Math.random() * (height || window.innerHeight);
      this.radius = randomBetween(CONFIG.particleMinRadius, CONFIG.particleMaxRadius);
      this.baseRadius = this.radius;
      this.color = pickColor();
      this.vx = randomBetween(-CONFIG.baseSpeed, CONFIG.baseSpeed);
      this.vy = randomBetween(-CONFIG.baseSpeed, CONFIG.baseSpeed);
      this.opacity = randomBetween(0.15, 0.55);
      this.baseOpacity = this.opacity;
      this.pulseOffset = Math.random() * Math.PI * 2;
      this.pulseSpeed = randomBetween(0.005, 0.02);
    }
    update(dt) {
      // Pulse
      this.pulseOffset += this.pulseSpeed * dt;
      const pulse = Math.sin(this.pulseOffset);
      this.radius = this.baseRadius + pulse * 0.6;
      this.opacity = this.baseOpacity + pulse * 0.08;

      // Mouse interaction: gentle repel + brightening
      const d = dist(this.x, this.y, mouseX, mouseY);
      if (d < CONFIG.mouseInfluenceRadius) {
        const factor = 1 - d / CONFIG.mouseInfluenceRadius;
        const angle = Math.atan2(this.y - mouseY, this.x - mouseX);
        this.vx += Math.cos(angle) * CONFIG.mouseRepelStrength * factor * dt;
        this.vy += Math.sin(angle) * CONFIG.mouseRepelStrength * factor * dt;
        // Brighten near cursor
        this.opacity = Math.min(this.baseOpacity + factor * 0.5, 1);
        this.radius = this.baseRadius + factor * 2.5;
      }

      // Soft velocity damping
      this.vx *= 0.998;
      this.vy *= 0.998;

      this.x += this.vx * dt;
      this.y += this.vy * dt;

      // Wrap around edges
      if (this.x < -20) this.x = width + 20;
      if (this.x > width + 20) this.x = -20;
      if (this.y < -20) this.y = height + 20;
      if (this.y > height + 20) this.y = -20;
    }
    draw() {
      const { r, g, b } = this.color;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r},${g},${b},${this.opacity})`;
      ctx.fill();

      // Soft glow halo
      if (this.radius > 2) {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius * 3, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(
          this.x, this.y, this.radius * 0.5,
          this.x, this.y, this.radius * 3
        );
        grad.addColorStop(0, `rgba(${r},${g},${b},${this.opacity * 0.25})`);
        grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
        ctx.fillStyle = grad;
        ctx.fill();
      }
    }
  }

  // ── Glow Orb (large soft light that drifts and reacts to mouse) ───
  class GlowOrb {
    constructor() {
      this.x = Math.random() * (width || window.innerWidth);
      this.y = Math.random() * (height || window.innerHeight);
      this.radius = randomBetween(CONFIG.glowOrbMinRadius, CONFIG.glowOrbMaxRadius);
      this.color = pickColor();
      this.vx = randomBetween(-0.15, 0.15);
      this.vy = randomBetween(-0.15, 0.15);
      this.opacity = randomBetween(0.015, 0.04);
      this.baseOpacity = this.opacity;
      this.phaseX = Math.random() * Math.PI * 2;
      this.phaseY = Math.random() * Math.PI * 2;
      this.driftSpeed = randomBetween(0.001, 0.004);
    }
    update(dt) {
      this.phaseX += this.driftSpeed * dt;
      this.phaseY += this.driftSpeed * dt * 0.7;
      this.x += (Math.sin(this.phaseX) * 0.3 + this.vx) * dt * 0.3;
      this.y += (Math.cos(this.phaseY) * 0.3 + this.vy) * dt * 0.3;

      // Attract towards mouse gently
      const d = dist(this.x, this.y, mouseX, mouseY);
      if (d < 500) {
        const factor = (1 - d / 500) * 0.005;
        this.x += (mouseX - this.x) * factor * dt;
        this.y += (mouseY - this.y) * factor * dt;
        this.opacity = lerp(this.baseOpacity, this.baseOpacity * 3, 1 - d / 500);
      } else {
        this.opacity = this.baseOpacity;
      }

      // Wrap
      if (this.x < -this.radius * 2) this.x = width + this.radius;
      if (this.x > width + this.radius * 2) this.x = -this.radius;
      if (this.y < -this.radius * 2) this.y = height + this.radius;
      if (this.y > height + this.radius * 2) this.y = -this.radius;
    }
    draw() {
      const { r, g, b } = this.color;
      const grad = ctx.createRadialGradient(
        this.x, this.y, 0,
        this.x, this.y, this.radius
      );
      grad.addColorStop(0, `rgba(${r},${g},${b},${this.opacity})`);
      grad.addColorStop(0.4, `rgba(${r},${g},${b},${this.opacity * 0.5})`);
      grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    }
  }

  // ── Draw connections between nearby particles ─────────────────────
  function drawConnections() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i], b = particles[j];
        const d = dist(a.x, a.y, b.x, b.y);
        if (d < CONFIG.connectionDistance) {
          const alpha = (1 - d / CONFIG.connectionDistance) * 0.12;

          // Brighten connections near mouse
          const midX = (a.x + b.x) / 2, midY = (a.y + b.y) / 2;
          const mouseDist = dist(midX, midY, mouseX, mouseY);
          const mouseBoost = mouseDist < CONFIG.mouseInfluenceRadius
            ? (1 - mouseDist / CONFIG.mouseInfluenceRadius) * 0.25
            : 0;

          const r = Math.round(lerp(a.color.r, b.color.r, 0.5));
          const g = Math.round(lerp(a.color.g, b.color.g, 0.5));
          const bl = Math.round(lerp(a.color.b, b.color.b, 0.5));

          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(${r},${g},${bl},${alpha + mouseBoost})`;
          ctx.lineWidth = 0.5 + mouseBoost * 2;
          ctx.stroke();
        }
      }
    }
  }

  // ── Mouse trail ring effect ───────────────────────────────────────
  let mouseTrails = [];
  function addMouseTrail() {
    if (mouseX < 0 || mouseY < 0) return;
    mouseTrails.push({
      x: mouseX, y: mouseY,
      radius: 4, maxRadius: 50,
      opacity: 0.2, color: pickColor()
    });
    if (mouseTrails.length > 6) mouseTrails.shift();
  }
  function updateMouseTrails(dt) {
    for (let i = mouseTrails.length - 1; i >= 0; i--) {
      const t = mouseTrails[i];
      t.radius += 0.8 * dt;
      t.opacity -= 0.004 * dt;
      if (t.opacity <= 0 || t.radius >= t.maxRadius) {
        mouseTrails.splice(i, 1);
      }
    }
  }
  function drawMouseTrails() {
    mouseTrails.forEach(t => {
      const { r, g, b } = t.color;
      ctx.beginPath();
      ctx.arc(t.x, t.y, t.radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${r},${g},${b},${t.opacity})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }

  // ── Main Loop ─────────────────────────────────────────────────────
  function animate(timestamp) {
    animFrameId = requestAnimationFrame(animate);

    const delta = timestamp - lastTime;
    if (delta < interval) return;
    lastTime = timestamp - (delta % interval);

    const dt = Math.min(delta / 16.67, 3); // Normalize to ~60fps, cap at 3x

    ctx.clearRect(0, 0, width, height);

    // Glow orbs (behind everything)
    glowOrbs.forEach(o => { o.update(dt); o.draw(); });

    // Connections
    drawConnections();

    // Particles
    particles.forEach(p => { p.update(dt); p.draw(); });

    // Mouse trails
    updateMouseTrails(dt);
    drawMouseTrails();
  }

  // ── Init / Resize ─────────────────────────────────────────────────
  function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
  }

  function init() {
    canvas = document.createElement('canvas');
    canvas.id = 'particle-canvas';
    canvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;';
    document.body.prepend(canvas);
    ctx = canvas.getContext('2d');
    resize();

    // Create particles
    particles = [];
    for (let i = 0; i < CONFIG.particleCount; i++) {
      particles.push(new Particle());
    }
    // Create glow orbs
    glowOrbs = [];
    for (let i = 0; i < CONFIG.glowOrbCount; i++) {
      glowOrbs.push(new GlowOrb());
    }

    // Events
    window.addEventListener('resize', () => {
      resize();
      // Redistribute particles outside screen
      particles.forEach(p => {
        if (p.x > width || p.y > height) p.reset();
      });
    });

    let trailTimer = 0;
    document.addEventListener('mousemove', (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
      trailTimer++;
      if (trailTimer % 8 === 0) addMouseTrail();
    });
    document.addEventListener('mouseleave', () => {
      mouseX = -1000;
      mouseY = -1000;
    });

    // Click burst effect
    document.addEventListener('click', (e) => {
      // Don't create burst if clicking on interactive elements
      if (e.target.closest('a, button, input, textarea, select, label')) return;
      for (let i = 0; i < 8; i++) {
        const p = new Particle();
        p.x = e.clientX;
        p.y = e.clientY;
        const angle = (Math.PI * 2 * i) / 8 + randomBetween(-0.3, 0.3);
        const speed = randomBetween(1, 3);
        p.vx = Math.cos(angle) * speed;
        p.vy = Math.sin(angle) * speed;
        p.radius = randomBetween(1.5, 3.5);
        p.baseRadius = p.radius;
        p.opacity = 0.7;
        p.baseOpacity = 0.7;
        particles.push(p);
      }
      // Trim excess particles
      while (particles.length > CONFIG.particleCount + 40) {
        particles.shift();
      }
    });

    animFrameId = requestAnimationFrame(animate);
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
