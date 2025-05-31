import {
  LitElement,
  html,
  css,
} from 'https://cdn.jsdelivr.net/gh/lit/dist@3/core/lit-core.min.js';

class SmoothScrollComponent extends LitElement {
  static properties = {
    targetKey: {type: String},
    behavior: {type: String}, // "smooth", "auto", "instant"
    block: {type: String},    // "start", "center", "end", "nearest" 
    inline: {type: String},   // "start", "center", "end", "nearest"
    duration: {type: Number}, // Animation duration in ms
    easing: {type: String},   // "ease-in", "ease-out", "ease-in-out", "linear"
    showProgress: {type: Boolean}, // Show scroll progress indicator
    trigger: {type: String},  // Event name to trigger scroll
    ScrollStart: {type: String},
    ScrollProgress: {type: String}, 
    ScrollComplete: {type: String},
  };

  static styles = css`
    :host {
      display: none; /* Hidden component, only for functionality */
    }
    
    .scroll-progress-indicator {
      position: fixed;
      top: 0;
      left: 0;
      height: 2px;
      background: #00A884;
      z-index: 9999;
      transition: width 0.1s ease-out;
      pointer-events: none;
    }
    
    .scroll-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.1);
      z-index: 9998;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.2s ease;
    }
    
    .scroll-overlay.active {
      opacity: 1;
    }
  `;

  constructor() {
    super();
    this.targetKey = '';
    this.behavior = 'smooth';
    this.block = 'end';
    this.inline = 'nearest';
    this.duration = 300;
    this.easing = 'ease-out';
    this.showProgress = true;
    this.trigger = '';
    this.ScrollStart = '';
    this.ScrollProgress = '';
    this.ScrollComplete = '';
    
    this.isScrolling = false;
    this.scrollStartTime = 0;
    this.progressIndicator = null;
    this.scrollOverlay = null;
  }

  connectedCallback() {
    super.connectedCallback();
    // Listen for trigger events
    if (this.trigger) {
      document.addEventListener(this.trigger, this.handleTrigger.bind(this));
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this.trigger) {
      document.removeEventListener(this.trigger, this.handleTrigger.bind(this));
    }
    this.cleanup();
  }

  render() {
    return html`<!-- Smooth Scroll Component - Hidden -->`; 
  }

  updated(changedProperties) {
    super.updated(changedProperties);
    
    // Auto-trigger scroll when targetKey changes
    if (changedProperties.has('targetKey') && this.targetKey) {
      this.performScroll();
    }
  }

  handleTrigger(event) {
    if (event.detail && event.detail.targetKey) {
      this.targetKey = event.detail.targetKey;
    }
    this.performScroll();
  }

  async performScroll() {
    if (!this.targetKey || this.isScrolling) return;
    
    const targetElement = this.findTargetElement();
    if (!targetElement) {
      console.warn(`SmoothScrollComponent: Target element with key "${this.targetKey}" not found`);
      return;
    }

    this.isScrolling = true;
    this.scrollStartTime = performance.now();
    
    // Emit scroll start event
    this.emitScrollEvent('ScrollStart', { targetKey: this.targetKey });
    
    // Create progress indicator if enabled
    if (this.showProgress) {
      this.createProgressIndicator();
    }

    if (this.behavior === 'instant') {
      this.scrollInstant(targetElement);
    } else if (this.behavior === 'auto') {
      this.scrollAuto(targetElement);
    } else {
      await this.scrollSmooth(targetElement);
    }
  }

  findTargetElement() {
    // Find element by Mesop key attribute
    return document.querySelector(`[key="${this.targetKey}"]`);
  }

  scrollInstant(targetElement) {
    targetElement.scrollIntoView({
      behavior: 'auto',
      block: this.block,
      inline: this.inline
    });
    this.completeScroll();
  }

  scrollAuto(targetElement) {
    targetElement.scrollIntoView({
      behavior: 'auto', 
      block: this.block,
      inline: this.inline
    });
    // Small delay for consistency
    setTimeout(() => this.completeScroll(), 50);
  }

  async scrollSmooth(targetElement) {
    if ('scrollBehavior' in document.documentElement.style) {
      // Use native smooth scrolling if supported
      targetElement.scrollIntoView({
        behavior: 'smooth',
        block: this.block, 
        inline: this.inline
      });
      
      // Monitor scroll completion
      await this.monitorNativeScroll(targetElement);
    } else {
      // Fallback to custom animation
      await this.animateScroll(targetElement);
    }
  }

  async monitorNativeScroll(targetElement) {
    return new Promise((resolve) => {
      const startTime = performance.now();
      const maxDuration = this.duration + 1000; // Max timeout
      
      const checkScroll = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / this.duration, 1);
        
        this.updateProgress(progress);
        
        if (progress >= 1 || elapsed > maxDuration) {
          this.completeScroll();
          resolve();
        } else {
          requestAnimationFrame(checkScroll);
        }
      };
      
      requestAnimationFrame(checkScroll);
    });
  }

  async animateScroll(targetElement) {
    const startPosition = window.pageYOffset;
    const targetPosition = this.getTargetScrollPosition(targetElement);
    const distance = targetPosition - startPosition;
    
    return new Promise((resolve) => {
      const startTime = performance.now();
      
      const animateStep = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / this.duration, 1);
        
        const easedProgress = this.applyEasing(progress);
        const currentPosition = startPosition + (distance * easedProgress);
        
        window.scrollTo(0, currentPosition);
        this.updateProgress(progress);
        
        if (progress < 1) {
          requestAnimationFrame(animateStep);
        } else {
          this.completeScroll();
          resolve();
        }
      };
      
      requestAnimationFrame(animateStep);
    });
  }

  getTargetScrollPosition(targetElement) {
    const rect = targetElement.getBoundingClientRect();
    const scrollTop = window.pageYOffset;
    
    switch (this.block) {
      case 'start':
        return scrollTop + rect.top;
      case 'center':
        return scrollTop + rect.top + rect.height / 2 - window.innerHeight / 2;
      case 'end':
        return scrollTop + rect.top + rect.height - window.innerHeight;
      case 'nearest':
      default:
        const elementTop = scrollTop + rect.top;
        const elementBottom = elementTop + rect.height;
        const viewportTop = scrollTop;
        const viewportBottom = scrollTop + window.innerHeight;
        
        if (elementTop < viewportTop) {
          return elementTop;
        } else if (elementBottom > viewportBottom) {
          return elementBottom - window.innerHeight;
        }
        return scrollTop; // Already in view
    }
  }

  applyEasing(progress) {
    switch (this.easing) {
      case 'ease-in':
        return progress * progress;
      case 'ease-out':
        return 1 - Math.pow(1 - progress, 2);
      case 'ease-in-out':
        return progress < 0.5 
          ? 2 * progress * progress 
          : 1 - Math.pow(-2 * progress + 2, 2) / 2;
      case 'linear':
      default:
        return progress;
    }
  }

  createProgressIndicator() {
    if (!this.showProgress) return;
    
    this.progressIndicator = document.createElement('div');
    this.progressIndicator.className = 'scroll-progress-indicator';
    this.progressIndicator.style.width = '0%';
    document.body.appendChild(this.progressIndicator);
    
    // Optional overlay
    this.scrollOverlay = document.createElement('div');
    this.scrollOverlay.className = 'scroll-overlay';
    document.body.appendChild(this.scrollOverlay);
    
    // Trigger overlay animation
    requestAnimationFrame(() => {
      this.scrollOverlay.classList.add('active');
    });
  }

  updateProgress(progress) {
    if (this.progressIndicator) {
      this.progressIndicator.style.width = `${progress * 100}%`;
    }
    
    // Emit progress event
    this.emitScrollEvent('ScrollProgress', { 
      targetKey: this.targetKey,
      progress: progress,
      elapsed: performance.now() - this.scrollStartTime 
    });
  }

  completeScroll() {
    this.isScrolling = false;
    
    // Emit completion event
    this.emitScrollEvent('ScrollComplete', { 
      targetKey: this.targetKey,
      totalDuration: performance.now() - this.scrollStartTime
    });
    
    // Cleanup UI
    setTimeout(() => this.cleanup(), 200);
  }

  cleanup() {
    if (this.progressIndicator) {
      this.progressIndicator.remove();
      this.progressIndicator = null;
    }
    
    if (this.scrollOverlay) {
      this.scrollOverlay.classList.remove('active');
      setTimeout(() => {
        if (this.scrollOverlay) {
          this.scrollOverlay.remove();
          this.scrollOverlay = null;
        }
      }, 200);
    }
  }

  emitScrollEvent(eventName, data) {
    const handlerName = this[eventName];
    if (handlerName) {
      this.dispatchEvent(
        new MesopEvent(handlerName, data)
      );
    }
  }
}

customElements.define('smooth-scroll-component', SmoothScrollComponent);
