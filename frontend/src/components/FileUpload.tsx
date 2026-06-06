'use client';

import { useState, useCallback, useRef } from 'react';
import { uploadDocument } from '@/lib/api';

interface FileUploadProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadComplete?: () => void;
}

export default function FileUpload({ isOpen, onClose, onUploadComplete }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = useCallback(async (file: File) => {
    setUploading(true);
    setProgress(20);
    setStatus(`Uploading ${file.name}...`);

    try {
      setProgress(50);
      const result = await uploadDocument(file);
      setProgress(100);
      setStatus(`✅ ${file.name} uploaded and processing`);

      setTimeout(() => {
        onUploadComplete?.();
        onClose();
        setUploading(false);
        setProgress(0);
        setStatus('');
      }, 1500);
    } catch (error: any) {
      setStatus(`❌ Upload failed: ${error.message}`);
      setUploading(false);
      setProgress(0);
    }
  }, [onClose, onUploadComplete]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  }, [handleUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  }, [handleUpload]);

  if (!isOpen) return null;

  return (
    <div className="file-upload-overlay" onClick={onClose}>
      <div className="file-upload-modal" onClick={(e) => e.stopPropagation()}>
        <h3>Upload Document</h3>
        <p>Upload a document to add it to the knowledge base for retrieval.</p>

        <div
          className={`drop-zone ${isDragging ? 'dragging' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <div className="label">
            {isDragging ? 'Drop file here' : 'Click or drag file to upload'}
          </div>
          <div className="formats">PDF, DOCX, TXT, MD, CSV</div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md,.csv"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />

        {uploading && (
          <div className="upload-progress">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)' }}>
              {status}
            </div>
          </div>
        )}

        {!uploading && status && (
          <div style={{ marginTop: 'var(--space-3)', fontSize: 'var(--text-sm)', color: 'var(--color-text-secondary)' }}>
            {status}
          </div>
        )}
      </div>
    </div>
  );
}
