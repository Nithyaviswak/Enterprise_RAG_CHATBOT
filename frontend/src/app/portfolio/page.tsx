'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import './portfolio.css';

/* ── Project Data ──────────────────────────────────────────── */
interface Project {
  id: number;
  title: string;
  tagline: string;
  description: string;
  image: string;
  tech: string[];
  metrics: { value: string; label: string; color: string }[];
  accent: string;
  gradient: string;
  github: string;
  demo: string;
  caseStudy: string;
}

const PROJECTS: Project[] = [
  {
    id: 1,
    title: 'Enterprise RAG Chatbot',
    tagline: 'AI-Powered Document Intelligence at Scale',
    description:
      'Production-grade conversational AI system with retrieval-augmented generation, powered by Google Gemini and fine-tuned embeddings. Processes 10K+ documents with sub-second retrieval and context-aware responses.',
    image: '/images/project-rag-chatbot.png',
    tech: ['Google Gemini', 'RAGFlow', 'FastAPI', 'Next.js', 'ChromaDB', 'Docker'],
    metrics: [
      { value: '<200ms', label: 'Retrieval', color: '#38BDF8' },
      { value: '94.7%', label: 'Accuracy', color: '#8B5CF6' },
      { value: '10K+', label: 'Documents', color: '#10B981' },
    ],
    accent: '#8B5CF6',
    gradient: 'linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(56,189,248,0.04) 100%)',
    github: '#',
    demo: '#',
    caseStudy: '#',
  },
  {
    id: 2,
    title: 'Retail Vision Intelligence',
    tagline: 'Real-Time Computer Vision for Smart Retail',
    description:
      'End-to-end computer vision pipeline for retail analytics with YOLOv8 object detection, customer tracking, heatmap generation, and real-time RTSP stream processing across multiple store locations.',
    image: '/images/project-retail-vision.png',
    tech: ['YOLOv8', 'OpenCV', 'TensorRT', 'React', 'FastAPI', 'Kafka'],
    metrics: [
      { value: '30 FPS', label: 'Throughput', color: '#38BDF8' },
      { value: '96.2%', label: 'Detection', color: '#10B981' },
      { value: '12', label: 'Cameras', color: '#8B5CF6' },
    ],
    accent: '#38BDF8',
    gradient: 'linear-gradient(135deg, rgba(56,189,248,0.08) 0%, rgba(16,185,129,0.04) 100%)',
    github: '#',
    demo: '#',
    caseStudy: '#',
  },
  {
    id: 3,
    title: 'Clinical Voice Copilot',
    tagline: 'AI-Powered Clinical Documentation & Triage',
    description:
      'Voice-driven clinical assistant automating patient triage and medical documentation. Integrates real-time speech recognition, NLP-based symptom extraction, and structured clinical note generation.',
    image: '/images/project-clinical-copilot.png',
    tech: ['Whisper', 'GPT-4', 'FastAPI', 'PostgreSQL', 'WebSocket', 'React'],
    metrics: [
      { value: '3.2s', label: 'Latency', color: '#10B981' },
      { value: '91%', label: 'Triage Acc.', color: '#8B5CF6' },
      { value: '60%', label: 'Time Saved', color: '#38BDF8' },
    ],
    accent: '#10B981',
    gradient: 'linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(139,92,246,0.04) 100%)',
    github: '#',
    demo: '#',
    caseStudy: '#',
  },
  {
    id: 4,
    title: 'Heart Disease Risk Predictor',
    tagline: 'Clinical-Grade ML Diagnostics with XGBoost',
    description:
      'Production ML pipeline for cardiovascular risk assessment using XGBoost with ONNX Runtime inference. Features SHAP explainability, clinical-grade UI, and real-time risk scoring with confidence intervals.',
    image: '/images/project-heart-prediction.png',
    tech: ['XGBoost', 'ONNX Runtime', 'SHAP', 'FastAPI', 'Tailwind CSS', 'MLflow'],
    metrics: [
      { value: '<50ms', label: 'Inference', color: '#8B5CF6' },
      { value: '93.8%', label: 'AUC-ROC', color: '#38BDF8' },
      { value: '0.91', label: 'F1 Score', color: '#10B981' },
    ],
    accent: '#8B5CF6',
    gradient: 'linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(248,113,113,0.04) 100%)',
    github: '#',
    demo: '#',
    caseStudy: '#',
  },
  {
    id: 5,
    title: 'MLOps Production Pipeline',
    tagline: 'End-to-End ML Lifecycle Management System',
    description:
      'Comprehensive MLOps platform with automated model training, drift detection via Evidently AI, experiment tracking with MLflow, and Kubernetes-based deployment with canary rollouts and A/B testing.',
    image: '/images/project-mlops-pipeline.png',
    tech: ['MLflow', 'Evidently AI', 'Kubernetes', 'Airflow', 'Prometheus', 'Grafana'],
    metrics: [
      { value: '99.9%', label: 'Uptime', color: '#10B981' },
      { value: '<15min', label: 'Deploy', color: '#38BDF8' },
      { value: '24/7', label: 'Monitoring', color: '#8B5CF6' },
    ],
    accent: '#38BDF8',
    gradient: 'linear-gradient(135deg, rgba(56,189,248,0.08) 0%, rgba(251,191,36,0.04) 100%)',
    github: '#',
    demo: '#',
    caseStudy: '#',
  },
  {
    id: 6,
    title: 'Customer Intelligence Platform',
    tagline: 'Enterprise Analytics & Predictive Insights',
    description:
      'Data-dense analytics platform with customer segmentation, churn prediction, sentiment analysis, and revenue forecasting. Built with Apache ECharts for real-time visualization of business KPIs.',
    image: '/images/project-customer-intel.png',
    tech: ['Scikit-learn', 'Apache ECharts', 'FastAPI', 'Redis', 'React', 'Pandas'],
    metrics: [
      { value: '87%', label: 'Churn Pred.', color: '#8B5CF6' },
      { value: '2.4x', label: 'ROI Uplift', color: '#10B981' },
      { value: '50K+', label: 'Users', color: '#38BDF8' },
    ],
    accent: '#10B981',
    gradient: 'linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(139,92,246,0.04) 100%)',
    github: '#',
    demo: '#',
    caseStudy: '#',
  },
];

const TOTAL = PROJECTS.length;

/* ── SVG Icons ─────────────────────────────────────────────── */
const GithubIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
    <path d="M9 18c-4.51 2-5-2-7-2" />
  </svg>
);

const ExternalIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    <polyline points="15 3 21 3 21 9" />
    <line x1="10" y1="14" x2="21" y2="3" />
  </svg>
);

const ArrowIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="5" y1="12" x2="19" y2="12" />
    <polyline points="12 5 19 12 12 19" />
  </svg>
);

/* ── Portfolio Page Component ──────────────────────────────── */
export default function PortfolioPage() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [navScrolled, setNavScrolled] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [projectsNavVisible, setProjectsNavVisible] = useState(false);

  const sectionRef = useRef<HTMLDivElement>(null);
  const isScrolling = useRef(false);
  const lastScrollTime = useRef(0);
  const touchStartY = useRef(0);

  /* ── Scroll-based card navigation ──────────────────────── */
  const goToProject = useCallback((index: number) => {
    const clamped = Math.max(0, Math.min(TOTAL - 1, index));
    setActiveIndex(clamped);
  }, []);

  /* Handle wheel events on the projects section */
  useEffect(() => {
    const section = sectionRef.current;
    if (!section) return;

    const handleWheel = (e: WheelEvent) => {
      const rect = section.getBoundingClientRect();
      const inView = rect.top <= 100 && rect.bottom >= window.innerHeight - 100;
      if (!inView) return;

      const now = Date.now();
      if (now - lastScrollTime.current < 600) return;

      if (Math.abs(e.deltaY) > 30) {
        e.preventDefault();
        lastScrollTime.current = now;
        if (e.deltaY > 0 && activeIndex < TOTAL - 1) {
          goToProject(activeIndex + 1);
        } else if (e.deltaY < 0 && activeIndex > 0) {
          goToProject(activeIndex - 1);
        }
      }
    };

    const handleTouchStart = (e: TouchEvent) => {
      touchStartY.current = e.touches[0].clientY;
    };

    const handleTouchEnd = (e: TouchEvent) => {
      const rect = section.getBoundingClientRect();
      const inView = rect.top <= 100 && rect.bottom >= window.innerHeight - 100;
      if (!inView) return;

      const deltaY = touchStartY.current - e.changedTouches[0].clientY;
      const now = Date.now();
      if (now - lastScrollTime.current < 600) return;

      if (Math.abs(deltaY) > 50) {
        lastScrollTime.current = now;
        if (deltaY > 0 && activeIndex < TOTAL - 1) {
          goToProject(activeIndex + 1);
        } else if (deltaY < 0 && activeIndex > 0) {
          goToProject(activeIndex - 1);
        }
      }
    };

    section.addEventListener('wheel', handleWheel, { passive: false });
    section.addEventListener('touchstart', handleTouchStart, { passive: true });
    section.addEventListener('touchend', handleTouchEnd, { passive: true });

    return () => {
      section.removeEventListener('wheel', handleWheel);
      section.removeEventListener('touchstart', handleTouchStart);
      section.removeEventListener('touchend', handleTouchEnd);
    };
  }, [activeIndex, goToProject]);

  /* Keyboard navigation */
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!projectsNavVisible) return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        e.preventDefault();
        goToProject(activeIndex + 1);
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        e.preventDefault();
        goToProject(activeIndex - 1);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeIndex, goToProject, projectsNavVisible]);

  /* Scroll progress + nav state */
  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      setScrollProgress(docHeight > 0 ? (scrollY / docHeight) * 100 : 0);
      setNavScrolled(scrollY > 60);

      if (sectionRef.current) {
        const rect = sectionRef.current.getBoundingClientRect();
        setProjectsNavVisible(rect.top <= 100 && rect.bottom >= window.innerHeight - 100);
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  /* ── Card transform calculations ───────────────────────── */
  const getCardStyle = (index: number): React.CSSProperties => {
    const diff = index - activeIndex;

    if (diff < 0) {
      // Already passed — slid up and out
      return {
        transform: `translateY(${diff * 80}px) scale(${0.92 + diff * 0.02})`,
        opacity: 0,
        filter: 'blur(4px)',
        zIndex: TOTAL - Math.abs(diff),
        pointerEvents: 'none',
      };
    }

    if (diff === 0) {
      // Active card
      return {
        transform: 'translateY(0) scale(1)',
        opacity: 1,
        filter: 'blur(0px)',
        zIndex: TOTAL + 1,
      };
    }

    // Behind / upcoming cards — stack underneath with offset
    const yOffset = diff * 35;
    const scaleVal = Math.max(0.88, 1 - diff * 0.04);
    const opacityVal = Math.max(0.2, 1 - diff * 0.25);
    const blurVal = diff * 1.5;

    return {
      transform: `translateY(${yOffset}px) scale(${scaleVal})`,
      opacity: opacityVal,
      filter: `blur(${blurVal}px)`,
      zIndex: TOTAL - diff,
      pointerEvents: 'none' as const,
    };
  };

  return (
    <div className="portfolio-root">
      {/* Scroll Progress */}
      <div className="scroll-progress" style={{ width: `${scrollProgress}%` }} />

      {/* Navigation */}
      <nav className={`portfolio-nav ${navScrolled ? 'scrolled' : ''}`}>
        <div className="nav-logo">NV.</div>
        <ul className="nav-links">
          <li><a href="#hero">Home</a></li>
          <li><a href="#projects">Projects</a></li>
          <li><a href="#about">About</a></li>
          <li><a href="#contact">Contact</a></li>
        </ul>
        <a href="#contact" className="nav-cta">Get in Touch</a>
      </nav>

      {/* Hero Section */}
      <section className="hero-section" id="hero">
        <div className="hero-bg">
          <img
            src="/images/landing-hero.jpg"
            alt="Scenic landscape"
            loading="eager"
          />
          <div className="hero-bg-overlay" />
          <div className="hero-bg-gradient" />
        </div>
        <div className="hero-content">
          <div className="hero-badge">
            <span className="hero-badge-dot" />
            Available for collaboration
          </div>
          <h1 className="hero-title">
            Building the Future with{' '}
            <span className="gradient-text">AI & Machine Learning</span>
          </h1>
          <p className="hero-subtitle">
            AI/ML Engineer crafting production-grade intelligent systems — from
            enterprise RAG pipelines and computer vision to clinical AI and
            MLOps infrastructure.
          </p>
          <div className="hero-actions">
            <button className="hero-btn-primary" onClick={() => document.getElementById('projects')?.scrollIntoView({ behavior: 'smooth' })}>
              <span>Explore Projects</span>
            </button>
            <a href="#contact" className="hero-btn-secondary">
              Download Resume
            </a>
          </div>
        </div>
        <div className="hero-scroll-indicator">
          <div className="scroll-mouse" />
          <span>Scroll</span>
        </div>
      </section>

      {/* Projects Section */}
      <section
        className="projects-section"
        id="projects"
        ref={sectionRef}
        style={{ minHeight: '100vh' }}
      >
        <div className="projects-sticky-wrapper">
          <div className="section-header">
            <div className="section-label">
              <span className="section-label-line" />
              Featured Work
              <span className="section-label-line" />
            </div>
            <h2 className="section-title">Selected Projects</h2>
            <p className="section-description">
              Each project represents a journey from concept to production —
              engineered for impact, designed for scale.
            </p>
          </div>

          <div className="projects-stack-container">
            {PROJECTS.map((project, index) => (
              <div
                key={project.id}
                className={`project-card ${index === activeIndex ? 'active' : 'behind'}`}
                style={getCardStyle(index)}
              >
                <div
                  className="project-card-gradient"
                  style={{ background: project.gradient }}
                />
                <div className="project-card-inner">
                  <div className="project-card-content">
                    <span className="project-number">
                      {String(project.id).padStart(2, '0')} / {String(TOTAL).padStart(2, '0')}
                    </span>
                    <h3 className="project-title">{project.title}</h3>
                    <p className="project-tagline">{project.tagline}</p>
                    <p className="project-description">{project.description}</p>
                    <div className="project-tech-stack">
                      {project.tech.map((t) => (
                        <span key={t} className="tech-chip">{t}</span>
                      ))}
                    </div>
                    <div className="project-metrics">
                      {project.metrics.map((m) => (
                        <div key={m.label} className="metric">
                          <span className="metric-value" style={{ color: m.color }}>
                            {m.value}
                          </span>
                          <span className="metric-label">{m.label}</span>
                        </div>
                      ))}
                    </div>
                    <div className="project-actions">
                      <a href={project.github} className="project-btn project-btn-secondary">
                        <GithubIcon /> GitHub
                      </a>
                      <a href={project.demo} className="project-btn project-btn-primary">
                        <ExternalIcon /> Live Demo
                      </a>
                      <a href={project.caseStudy} className="project-btn project-btn-ghost">
                        View Case Study <ArrowIcon />
                      </a>
                    </div>
                  </div>
                  <div className="project-card-image">
                    <img
                      src={project.image}
                      alt={project.title}
                      loading="lazy"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Vertical Nav */}
      <div className={`projects-nav ${projectsNavVisible ? 'visible' : ''}`}>
        <span className="projects-nav-counter">
          {String(activeIndex + 1).padStart(2, '0')} / {String(TOTAL).padStart(2, '0')}
        </span>
        <div className="projects-nav-track">
          <div
            className="projects-nav-fill"
            style={{ height: `${((activeIndex + 1) / TOTAL) * 100}%` }}
          />
        </div>
        <div className="projects-nav-dots">
          {PROJECTS.map((_, i) => (
            <button
              key={i}
              className={`projects-nav-dot ${i === activeIndex ? 'active' : ''}`}
              onClick={() => goToProject(i)}
              aria-label={`Go to project ${i + 1}`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
