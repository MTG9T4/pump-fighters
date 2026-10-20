// On-screen arcade controls: 8-way stick on the left, six attack buttons on the right
// (LP MP HP / LK MK HK), START and SELECT. Multi-touch through pointer events: every finger is
// tracked on its own, fingers can slide between buttons, and a finger between two buttons
// presses both (e.g. LP+LK for a throw).
//
// The state goes to the game as a bit mask (Web_SetTouchState in sdl_pad.c), where it acts as
// player 1's keyboard.

export const Bits = {
  UP: 1 << 0,
  DOWN: 1 << 1,
  LEFT: 1 << 2,
  RIGHT: 1 << 3,
  LP: 1 << 4,
  MP: 1 << 5,
  HP: 1 << 6,
  LK: 1 << 7,
  MK: 1 << 8,
  HK: 1 << 9,
  START: 1 << 10,
  SELECT: 1 << 11,
};

const DIRECTIONS = [
  Bits.RIGHT,
  Bits.RIGHT | Bits.DOWN,
  Bits.DOWN,
  Bits.DOWN | Bits.LEFT,
  Bits.LEFT,
  Bits.LEFT | Bits.UP,
  Bits.UP,
  Bits.UP | Bits.RIGHT,
];

export class TouchControls {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.stickZone   area that captures the stick finger
   * @param {HTMLElement} opts.stickBase   the drawn stick (its center is the neutral point)
   * @param {HTMLElement} opts.stickKnob
   * @param {HTMLElement} opts.buttonZone  area that captures attack fingers
   * @param {HTMLElement[]} opts.holdButtons  elements with data-bit (START, SELECT, attack buttons)
   * @param {(mask: number) => void} opts.onChange
   * @param {() => void} opts.onFirstTouch
   */
  constructor(opts) {
    this.opts = opts;
    this.stickMask = 0;
    this.stickPointer = null;
    this.buttonPointers = new Map(); // pointerId -> mask
    this.holdPointers = new Map(); // pointerId -> mask (START/SELECT)
    this.mask = 0;
    this.haptics = true;
    this.attackButtons = [...opts.buttonZone.querySelectorAll('[data-bit]')];
    this.rects = null;

    this._bindStick();
    this._bindButtons();
    this._bindHoldButtons();

    const relayout = () => {
      this.rects = null;
    };
    window.addEventListener('resize', relayout);
    window.addEventListener('orientationchange', relayout);
  }

  // ---- stick ---------------------------------------------------------------------------

  _bindStick() {
    const { stickZone } = this.opts;

    stickZone.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.opts.onFirstTouch?.();
      if (this.stickPointer !== null) return;
      this.stickPointer = e.pointerId;
      stickZone.setPointerCapture?.(e.pointerId);
      this._updateStick(e);
    });

    stickZone.addEventListener('pointermove', (e) => {
      if (e.pointerId !== this.stickPointer) return;
      e.preventDefault();
      this._updateStick(e);
    });

    const release = (e) => {
      if (e.pointerId !== this.stickPointer) return;
      this.stickPointer = null;
      this.stickMask = 0;
      this.opts.stickKnob.style.transform = 'translate(-50%, -50%)';
      this._render();
      this._emit();
    };

    stickZone.addEventListener('pointerup', release);
    stickZone.addEventListener('pointercancel', release);
    stickZone.addEventListener('lostpointercapture', release);
  }

  _updateStick(e) {
    const base = this.opts.stickBase.getBoundingClientRect();
    const cx = base.left + base.width / 2;
    const cy = base.top + base.height / 2;
    const radius = base.width / 2;
    const dx = e.clientX - cx;
    const dy = e.clientY - cy;
    const dist = Math.hypot(dx, dy);
    let mask = 0;

    if (dist > radius * 0.22) {
      // 8 sectors of 45 degrees, 0 = right, clockwise (screen y grows down)
      const angle = Math.atan2(dy, dx);
      const sector = (Math.round(angle / (Math.PI / 4)) + 8) % 8;
      mask = DIRECTIONS[sector];
    }

    const clamp = Math.min(dist, radius * 0.55);
    const kx = dist ? (dx / dist) * clamp : 0;
    const ky = dist ? (dy / dist) * clamp : 0;
    this.opts.stickKnob.style.transform = `translate(calc(-50% + ${kx}px), calc(-50% + ${ky}px))`;

    if (mask !== this.stickMask) {
      if (mask && (mask & ~this.stickMask)) this._buzz(6);
      this.stickMask = mask;
      this._render();
      this._emit();
    }
  }

  // ---- attack buttons --------------------------------------------------------------------

  _measure() {
    this.rects = this.attackButtons.map((el) => {
      const r = el.getBoundingClientRect();
      return { bit: Number(el.dataset.bit), x: r.left + r.width / 2, y: r.top + r.height / 2, r: r.width / 2 };
    });
  }

  _hit(x, y) {
    if (!this.rects) this._measure();
    let mask = 0;
    let nearest = null;
    let nearestDist = Infinity;

    for (const b of this.rects) {
      const d = Math.hypot(x - b.x, y - b.y);
      // Slightly larger than the drawn button: the lens between two neighbours presses both
      if (d <= b.r + 9) mask |= b.bit;
      if (d < nearestDist) {
        nearestDist = d;
        nearest = b;
      }
    }

    // Fingers in the zone that miss every circle still press the closest button
    if (!mask && nearest && nearestDist <= nearest.r * 1.9) mask = nearest.bit;
    return mask;
  }

  _bindButtons() {
    const zone = this.opts.buttonZone;

    zone.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.opts.onFirstTouch?.();
      zone.setPointerCapture?.(e.pointerId);
      const mask = this._hit(e.clientX, e.clientY);
      this.buttonPointers.set(e.pointerId, mask);
      if (mask) this._buzz(10);
      this._render();
      this._emit();
    });

    zone.addEventListener('pointermove', (e) => {
      if (!this.buttonPointers.has(e.pointerId)) return;
      e.preventDefault();
      const old = this.buttonPointers.get(e.pointerId);
      const mask = this._hit(e.clientX, e.clientY);
      if (mask === old) return;
      if (mask & ~old) this._buzz(8);
      this.buttonPointers.set(e.pointerId, mask);
      this._render();
      this._emit();
    });

    const release = (e) => {
      if (!this.buttonPointers.delete(e.pointerId)) return;
      this._render();
      this._emit();
    };

    zone.addEventListener('pointerup', release);
    zone.addEventListener('pointercancel', release);
    zone.addEventListener('lostpointercapture', release);
  }

  // ---- START / SELECT ------------------------------------------------------------------------

  _bindHoldButtons() {
    for (const el of this.opts.holdButtons) {
      const bit = Number(el.dataset.bit);

      el.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        this.opts.onFirstTouch?.();
        el.setPointerCapture?.(e.pointerId);
        this.holdPointers.set(e.pointerId, bit);
        this._buzz(10);
        this._render();
        this._emit();
      });

      const release = (e) => {
        if (!this.holdPointers.delete(e.pointerId)) return;
        this._render();
        this._emit();
      };

      el.addEventListener('pointerup', release);
      el.addEventListener('pointercancel', release);
      el.addEventListener('lostpointercapture', release);
    }
  }

  // ---- output --------------------------------------------------------------------------------

  /** Releases everything (page hidden, settings opened). */
  reset() {
    this.stickPointer = null;
    this.stickMask = 0;
    this.buttonPointers.clear();
    this.holdPointers.clear();
    this.opts.stickKnob.style.transform = 'translate(-50%, -50%)';
    this._render();
    this._emit();
  }

  _currentMask() {
    let mask = this.stickMask;
    for (const m of this.buttonPointers.values()) mask |= m;
    for (const m of this.holdPointers.values()) mask |= m;
    return mask;
  }

  _emit() {
    const mask = this._currentMask();
    if (mask === this.mask) return;
    this.mask = mask;
    this.opts.onChange(mask);
  }

  _render() {
    const mask = this._currentMask();

    for (const el of this.attackButtons) {
      el.classList.toggle('down', (mask & Number(el.dataset.bit)) !== 0);
    }

    for (const el of this.opts.holdButtons) {
      el.classList.toggle('down', (mask & Number(el.dataset.bit)) !== 0);
    }

    this.opts.stickBase.dataset.dir = String(this.stickMask);
  }

  _buzz(ms) {
    if (this.haptics && navigator.vibrate) {
      try {
        navigator.vibrate(ms);
      } catch (_) {
        // not allowed
      }
    }
  }
}
