'use client';

import React, { useState, useRef } from 'react';
import { Upload, File, Film, Image as ImageIcon, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface DragDropUploadProps {
  onUploadStart: () => void;
  onUploadSuccess: (fileId: string, filename: string, fileType: string) => void;
  onUploadError: (error: string) => void;
  disabled?: boolean;
}

export default function DragDropUpload({
  onUploadStart,
  onUploadSuccess,
  onUploadError,
  disabled
}: DragDropUploadProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (disabled) return;
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return;
    if (e.target.files && e.target.files[0]) {
      await processFile(e.target.files[0]);
    }
  };

  const processFile = async (file: File) => {
    const validExtensions = ['.mp4', '.avi', '.png', '.jpg', '.jpeg', '.pdf'];
    const suffix = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    
    if (!validExtensions.includes(suffix)) {
      onUploadError(`Unsupported file format. Please upload Video (${validExtensions.slice(0,2).join('/')}), Image (${validExtensions.slice(2,5).join('/')}), or Document (${validExtensions.slice(5).join('/')}).`);
      return;
    }

    setSelectedFile(file);
    onUploadStart();
    setUploadProgress(0);

    // Identify file category
    let fileType = 'image';
    if (suffix === '.mp4' || suffix === '.avi') fileType = 'video';
    else if (suffix === '.pdf') fileType = 'document';

    // Prepare FormData
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Create XML Http Request to track upload progress
      const xhr = new XMLHttpRequest();
      xhr.open('POST', 'http://localhost:8000/api/v1/upload', true);
      
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percentComplete = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(percentComplete);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            onUploadSuccess(data.file_id, file.name, fileType);
          } catch (err) {
            onUploadError("Failed to parse server response.");
          }
        } else {
          onUploadError(`Upload failed with status ${xhr.status}`);
        }
        setUploadProgress(null);
      };

      xhr.onerror = () => {
        onUploadError("Network error during file upload. Make sure the FastAPI server is running.");
        setUploadProgress(null);
      };

      xhr.send(formData);
    } catch (err: any) {
      onUploadError(err.message || "An unexpected error occurred during upload.");
      setUploadProgress(null);
    }
  };

  const triggerFileInput = () => {
    if (disabled) return;
    fileInputRef.current?.click();
  };

  const getFileIcon = () => {
    if (!selectedFile) return <Upload className="w-12 h-12 text-gray-500 group-hover:text-primary transition-colors duration-300" />;
    
    const name = selectedFile.name.toLowerCase();
    if (name.endsWith('.mp4') || name.endsWith('.avi')) {
      return <Film className="w-12 h-12 text-primary" />;
    }
    if (name.endsWith('.pdf')) {
      return <File className="w-12 h-12 text-secondary" />;
    }
    return <ImageIcon className="w-12 h-12 text-success" />;
  };

  return (
    <div className="w-full">
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={triggerFileInput}
        className={`relative w-full h-56 border-2 border-dashed rounded-xl flex flex-col items-center justify-center cursor-pointer transition-all duration-300 group ${
          isDragActive 
            ? 'border-primary bg-primary/5 scale-[0.99]' 
            : 'border-white/10 hover:border-primary/50 hover:bg-white/2 bg-white/1'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={handleFileChange}
          accept=".mp4,.avi,.png,.jpg,.jpeg,.pdf"
          disabled={disabled}
        />
        
        <div className="flex flex-col items-center text-center px-6">
          <div className="mb-3 p-3 bg-white/5 rounded-full group-hover:scale-110 transition-transform duration-300">
            {getFileIcon()}
          </div>
          
          {selectedFile ? (
            <div>
              <p className="font-semibold text-white line-clamp-1 max-w-md">{selectedFile.name}</p>
              <p className="text-xs text-gray-400 mt-1">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
          ) : (
            <div>
              <p className="font-semibold text-white text-sm md:text-base">Drag & drop your file here, or <span className="text-primary group-hover:underline">browse</span></p>
              <p className="text-xs text-gray-500 mt-2">Supports MP4, AVI (video), PNG, JPG (image), or PDF (document) up to 100MB</p>
            </div>
          )}
        </div>

        {/* Progress Overlay */}
        {uploadProgress !== null && (
          <div className="absolute inset-0 bg-background-dark/90 rounded-xl flex flex-col items-center justify-center px-8">
            <div className="flex items-center justify-between w-full max-w-xs mb-2">
              <span className="text-xs text-primary font-mono glow-text-cyan">UPLOADING MEDIA...</span>
              <span className="text-xs text-primary font-mono">{uploadProgress}%</span>
            </div>
            <div className="w-full max-w-xs h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div 
                className="h-full bg-primary shadow-[0_0_8px_#00f0ff] transition-all duration-200" 
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
