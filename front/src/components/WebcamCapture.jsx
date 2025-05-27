import React, { useRef, useEffect, useState } from 'react';

const WebcamCapture = ({ onCapture }) => {
  const videoRef = useRef(null);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const streamRef = useRef(null); // Para almacenar el stream y detenerlo

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' }, // Cambiar a 'user' para la cámara frontal
          audio: false,
        });

        streamRef.current = stream; // Guardar el stream para detenerlo después

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            setIsVideoReady(true);
          };
        }
      } catch (err) {
        console.error('Error accediendo a la cámara:', err);
        setError('No se pudo acceder a la cámara. Asegúrate de que esté habilitada y no esté en uso por otra aplicación.');
      }
    };

    startCamera();

    // Cleanup: Detener el stream cuando el componente se desmonta
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const handleCapture = async () => {
    const video = videoRef.current;
    if (!video || !isVideoReady) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    setIsLoading(true);
    setError('');

    canvas.toBlob(async (blob) => {
      if (!blob) {
        console.error('No se pudo capturar imagen del canvas');
        setError('Error al capturar la imagen');
        setIsLoading(false);
        return;
      }

      try {
        await onCapture(blob); // Llamar a la función onCapture pasada desde el padre
      } catch (err) {
        console.error('Error al procesar la captura:', err);
        setError('Error al procesar la captura');
      } finally {
        setIsLoading(false);
      }
    }, 'image/jpeg');
  };

  return (
    <div className="webcam-container">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="rounded shadow mx-auto"
        width="300"
      />
      {error && <p className="text-red-500 mt-2">{error}</p>}
      <button
        onClick={handleCapture}
        disabled={!isVideoReady || isLoading}
        className="bg-blue-500 text-white p-3 rounded-lg mt-4 disabled:bg-gray-400"
      >
        {isLoading ? 'Procesando...' : 'Capturar y Predecir'}
      </button>
    </div>
  );
};

export default WebcamCapture;